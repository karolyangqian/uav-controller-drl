# Report Formatting Guidelines & Pandoc Configuration

This document defines the report drafting and formatting standards for the project, configured for **Pandoc** compilation to PDF and Word DOCX.

---

## 📐 Formatting Specifications

| Element | Specification | Pandoc / LaTeX Implementation |
| :--- | :--- | :--- |
| **Font Family** | **Times New Roman** | `\usepackage{mathptmx}` (pdflatex) or `mainfont: Times New Roman` (lualatex) |
| **Normal Text Size** | **12 pt** | `-V fontsize=12pt` / `\documentclass[12pt]{article}` |
| **Text Alignment** | **Justified** | LaTeX default full justification (`\microtype`) |
| **Document Title** | **Centered** (18 pt, Bold, Black) | `\usepackage{titling}` custom `\pretitle` block |
| **Section Headings** | **Left-Aligned** (0 pt left indent) | `\usepackage{titlesec}` with `\titlespacing*{...}{0pt}{...}{...}` |
| **Section Hierarchy** | Distinguished by font size: <br>• **Level 1 (`#`)**: 16 pt, Bold <br>• **Level 2 (`##`)**: 14 pt, Bold <br>• **Level 3 (`###`)**: 12 pt, Bold | Defined via `\titleformat` in `docs/preamble.tex` |
| **Section Numbering** | **Multi-level numeric**: `1`, `1.1`, `1.1.1` | Flag `--number-sections` or YAML `numbersections: true` |
| **Color Scheme** | **Strictly Black (`#000000`)** | `\color{black}`, links borderless & uncolored (`hidelinks`) |
| **Margins** | 1 inch ($2.54\text{ cm}$) all sides | `\usepackage[margin=1in]{geometry}` |

---

## 🛠️ How to Compile Using Pandoc

### 1. Compile to PDF (Recommended)

To compile a Markdown report file (e.g. `docs/report_template.md`) into a formatted PDF using the pre-configured preamble:

```bash
pandoc docs/report_template.md \
  -o docs/report.pdf \
  -H docs/preamble.tex \
  --number-sections \
  -V fontsize=12pt \
  --pdf-engine=pdflatex
```

Or run the automated helper script:
```bash
./scripts/build_report.sh
```

### 2. Compile to Microsoft Word (.docx)

Pandoc can also output directly to Microsoft Word `.docx` with preserved section numbering:

```bash
pandoc docs/report_template.md \
  -o docs/report.docx \
  --number-sections
```

---

## 📄 File Reference

* **[`docs/preamble.tex`](file:///home/karol/ITB/kuliah/uav-controller/docs/preamble.tex)**: The LaTeX styling preamble enforcing Times New Roman 12pt, left-aligned headings (0pt indent), centered title, and pure black text.
* **[`docs/report_template.md`](file:///home/karol/ITB/kuliah/uav-controller/docs/report_template.md)**: A complete, pre-structured academic report template tailored for this DRL UAV project.
* **[`scripts/build_report.sh`](file:///home/karol/ITB/kuliah/uav-controller/scripts/build_report.sh)**: Executable bash script for one-command report building.
