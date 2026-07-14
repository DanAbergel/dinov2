#!/bin/bash
# Rebuild the dark-background (editor-style) Code Review PDF from CODE_REVIEW.md.
# Needs: pandoc, weasyprint, and Homebrew pango (`brew install pango`).
set -euo pipefail
cd "$(dirname "$0")"

pandoc CODE_REVIEW.md -f gfm -t html5 -s \
  --syntax-highlighting=breezedark \
  --embed-resources --css review_dark.css \
  --metadata title="fMRI × DINOv2 — Code Review" \
  -o .review.html

DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}" \
  weasyprint .review.html CODE_REVIEW.pdf

rm -f .review.html
echo "-> $(pwd)/CODE_REVIEW.pdf"
