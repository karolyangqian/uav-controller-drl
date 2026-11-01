/*
Standalone C testbench for Q8.8 Fixed-Point Forward Pass.
Matches the hardware architecture to be synthesized into AMD Kria KR260 FPGA.
*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "weights.h"

void quadrotor_forward(const int16_t x[16], int16_t out[4]) {
    int32_t h1[64];
    for (int i = 0; i < 64; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 16; j++) {
            acc += (int32_t)W1[i][j] * (int32_t)x[j];
        }
        acc = (acc >> Q_SHIFT) + (int32_t)B1[i];
        h1[i] = acc > 0 ? acc : 0;
    }

    int32_t h2[64];
    for (int i = 0; i < 64; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 64; j++) {
            acc += (int32_t)W2[i][j] * h1[j];
        }
        acc = (acc >> Q_SHIFT) + (int32_t)B2[i];
        h2[i] = acc > 0 ? acc : 0;
    }

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

int main() {
    printf("============================================================\n");
    printf("Testing Embedded C Q8.8 Fixed-Point Forward Inference\n");
    printf("============================================================\n");

    // Sample observation input: [0.5, 0.0, 0.2, ... zeros]
    int16_t test_x[16] = {0};
    test_x[0] = (int16_t)(0.5 * Q_SCALE);
    test_x[2] = (int16_t)(0.2 * Q_SCALE);
    test_x[6] = (int16_t)(1.0 * Q_SCALE); // z_body = 1.0 upright

    int16_t test_out[4] = {0};

    // Benchmark inference latency in C
    clock_t start = clock();
    int iterations = 100000;
    for (int iter = 0; iter < iterations; iter++) {
        quadrotor_forward(test_x, test_out);
    }
    clock_t end = clock();
    double total_time_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double latency_us = (total_time_sec / iterations) * 1e6;

    printf("[OK] Forward pass executed successfully %d times.\n", iterations);
    printf("     C Inference Latency:   %.3f microseconds / sample\n", latency_us);
    printf("     Achievable Frequency:  %.1f kHz\n", 1000.0 / latency_us);
    printf("     Raw Commanded Action:  [%d, %d, %d, %d]\n", test_out[0], test_out[1], test_out[2], test_out[3]);
    printf("     Normalized Actions:    [%.3f, %.3f, %.3f, %.3f]\n",
           (float)test_out[0] / Q_SCALE,
           (float)test_out[1] / Q_SCALE,
           (float)test_out[2] / Q_SCALE,
           (float)test_out[3] / Q_SCALE);
    printf("============================================================\n");

    return 0;
}
