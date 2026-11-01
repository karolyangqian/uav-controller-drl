/*
 * AMD Kria KR260 Hardware-in-the-Loop (HIL) C Client.
 *
 * Designed for deployment on AMD Kria KR260 (Ubuntu 24.04 on ARM Cortex-A53 PS).
 * Executes Q8.8 fixed-point integer neural network inference (< 2 microseconds).
 * Zero external dependencies: pure standard C and POSIX sockets.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "weights.h"

#define STATE_MAGIC  0x55415653  // "UAVS"
#define ACTION_MAGIC 0x55415641  // "UAVA"

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t seq_id;
    float obs[16];
} StatePacket;

typedef struct {
    uint32_t magic;
    uint32_t seq_id;
    float action[4];
} ActionPacket;
#pragma pack(pop)

static volatile int g_running = 1;

void handle_sigint(int sig) {
    (void)sig;
    g_running = 0;
}

// Q8.8 Fixed-Point Forward Inference Pass
void quadrotor_forward(const int16_t x[16], int16_t out[4]) {
    // Layer 1: 16 -> 64 with ReLU
    int32_t h1[64];
    for (int i = 0; i < 64; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 16; j++) {
            acc += (int32_t)W1[i][j] * (int32_t)x[j];
        }
        acc = (acc >> Q_SHIFT) + (int32_t)B1[i];
        h1[i] = acc > 0 ? acc : 0;
    }

    // Layer 2: 64 -> 64 with ReLU
    int32_t h2[64];
    for (int i = 0; i < 64; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 64; j++) {
            acc += (int32_t)W2[i][j] * h1[j];
        }
        acc = (acc >> Q_SHIFT) + (int32_t)B2[i];
        h2[i] = acc > 0 ? acc : 0;
    }

    // Output Layer: 64 -> 4 with clamping to [-1.0, 1.0] in Q8.8
    for (int i = 0; i < 4; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 64; j++) {
            acc += (int32_t)W3[i][j] * h2[j];
        }
        acc = (acc >> Q_SHIFT) + (int32_t)B3[i];
        if (acc > Q_SCALE) acc = Q_SCALE;
        if (acc < -Q_SCALE) acc = -Q_SCALE;
        out[i] = (int16_t)acc;
    }
}

int main(int argc, char *argv[]) {
    int client_port = 5005;
    int server_port = 5006;
    const char *server_ip = "127.0.0.1";
    int max_steps = 0;

    // Parse CLI options
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            client_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            server_ip = argv[++i];
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            server_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-p client_port] [-s server_ip] [-r server_port] [-m max_steps]\n", argv[0]);
            return 0;
        }
    }

    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    printf("============================================================\n");
    printf("  AMD Kria KR260 Fixed-Point C HIL Inference Client\n");
    printf("============================================================\n");
    printf("  Listening on Port:   %d\n", client_port);
    printf("  Server Target:       %s:%d\n", server_ip, server_port);
    printf("  Arithmetic Mode:     Signed 16-Bit Q8.8 Fixed-Point\n");
    printf("============================================================\n");

    // Create UDP Socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("[ERROR] Failed to create UDP socket");
        return 1;
    }

    struct sockaddr_in client_addr, server_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    client_addr.sin_port = htons(client_port);

    if (bind(sockfd, (struct sockaddr *)&client_addr, sizeof(client_addr)) < 0) {
        perror("[ERROR] Failed to bind client socket");
        close(sockfd);
        return 1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("[ERROR] Invalid server IP address");
        close(sockfd);
        return 1;
    }

    // Set 100ms socket timeout so loop can check g_running
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    StatePacket in_pkt;
    ActionPacket out_pkt;
    out_pkt.magic = ACTION_MAGIC;

    uint32_t step_count = 0;
    double total_infer_us = 0.0;
    struct timespec t0, t1;

    printf("\n[READY] Listening for simulation states... (Press Ctrl+C to terminate)\n\n");

    while (g_running) {
        ssize_t n = recv(sockfd, &in_pkt, sizeof(StatePacket), 0);
        if (n < 0) {
            // Timeout or interrupt
            continue;
        }

        if (n != sizeof(StatePacket) || in_pkt.magic != STATE_MAGIC) {
            continue;
        }

        // Measure microsecond inference latency
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Convert incoming float observations to Q8.8 fixed-point int16_t
        int16_t q_x[16];
        for (int i = 0; i < 16; i++) {
            float val = in_pkt.obs[i];
            int32_t q = (int32_t)(val * (float)Q_SCALE);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            q_x[i] = (int16_t)q;
        }

        // Execute integer forward pass
        int16_t q_out[4];
        quadrotor_forward(q_x, q_out);

        // Convert back to floating point action for transmission
        out_pkt.seq_id = in_pkt.seq_id;
        for (int i = 0; i < 4; i++) {
            out_pkt.action[i] = (float)q_out[i] / (float)Q_SCALE;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double infer_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
        total_infer_us += infer_us;

        // Transmit ActionPacket back to server
        sendto(sockfd, &out_pkt, sizeof(ActionPacket), 0, (struct sockaddr *)&server_addr, sizeof(server_addr));

        step_count++;
        if (step_count % 50 == 0) {
            printf("[KRIA PL/C] Step %5u | Seq %5u | Latency: %5.2f µs | Action: [%5.2f, %5.2f, %5.2f, %5.2f]\n",
                   step_count, in_pkt.seq_id, infer_us,
                   out_pkt.action[0], out_pkt.action[1], out_pkt.action[2], out_pkt.action[3]);
        }

        if (max_steps > 0 && step_count >= (uint32_t)max_steps) {
            break;
        }
    }

    close(sockfd);

    printf("\n============================================================\n");
    printf("  Kria KR260 Fixed-Point C Client Benchmark Summary\n");
    printf("============================================================\n");
    printf("  Total Inferences:      %u\n", step_count);
    if (step_count > 0) {
        double avg_lat = total_infer_us / step_count;
        printf("  Average Latency:       %.3f microseconds / inference\n", avg_lat);
        printf("  Throughput Capacity:   %.1f kHz (%.0f inferences/sec)\n", 1000.0 / avg_lat, 1e6 / avg_lat);
    }
    printf("============================================================\n");

    return 0;
}
