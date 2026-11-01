#!/usr/bin/env bash
# Script to compile Markdown report to PDF and DOCX using Pandoc
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
INPUT_MD="${1:-$DOCS_DIR/report_template.md}"
OUTPUT_PDF="${2:-$DOCS_DIR/report.pdf}"
OUTPUT_DOCX="${3:-$DOCS_DIR/report.docx}"

echo "============================================================"
echo "Building Report via Pandoc"
echo "Project Root: $PROJECT_ROOT"
echo "Input:        $INPUT_MD"
echo "Output PDF:   $OUTPUT_PDF"
echo "Output DOCX:  $OUTPUT_DOCX"
echo "============================================================"

# Compile to PDF
pandoc "$INPUT_MD" \
  -o "$OUTPUT_PDF" \
  -H "$DOCS_DIR/preamble.tex" \
  --resource-path="$PROJECT_ROOT:$DOCS_DIR:$PROJECT_ROOT/figures:$PROJECT_ROOT/media" \
  --number-sections \
  -V fontsize=12pt \
  --pdf-engine=pdflatex

echo "[OK] PDF generated successfully: $OUTPUT_PDF"

# Compile to DOCX
pandoc "$INPUT_MD" \
  -o "$OUTPUT_DOCX" \
  --resource-path="$PROJECT_ROOT:$DOCS_DIR:$PROJECT_ROOT/figures:$PROJECT_ROOT/media" \
  --number-sections

echo "[OK] DOCX generated successfully: $OUTPUT_DOCX"
echo "============================================================"

