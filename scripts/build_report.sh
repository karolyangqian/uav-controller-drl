#!/usr/bin/env bash
# Script to compile Markdown report to PDF and DOCX using Pandoc
set -e

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../docs" && pwd)"
INPUT_MD="${1:-$DOCS_DIR/report_template.md}"
OUTPUT_PDF="${2:-$DOCS_DIR/report.pdf}"
OUTPUT_DOCX="${3:-$DOCS_DIR/report.docx}"

echo "============================================================"
echo "Building Report via Pandoc"
echo "Input:       $INPUT_MD"
echo "Output PDF:  $OUTPUT_PDF"
echo "Output DOCX: $OUTPUT_DOCX"
echo "============================================================"

# Compile to PDF
pandoc "$INPUT_MD" \
  -o "$OUTPUT_PDF" \
  -H "$DOCS_DIR/preamble.tex" \
  --number-sections \
  -V fontsize=12pt \
  --pdf-engine=pdflatex

echo "[OK] PDF generated successfully: $OUTPUT_PDF"

# Compile to DOCX
pandoc "$INPUT_MD" \
  -o "$OUTPUT_DOCX" \
  --number-sections

echo "[OK] DOCX generated successfully: $OUTPUT_DOCX"
echo "============================================================"
