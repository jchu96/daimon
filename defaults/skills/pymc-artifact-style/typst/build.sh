#!/usr/bin/env bash
# Compile a PyMC Labs report. Wires in the bundled brand fonts (Inter, Archivo)
# and the local Fira Math. Usage: ./build.sh examples/report.typ [out.pdf]
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-examples/report.typ}"
OUT="${2:-${SRC%.typ}.pdf}"
typst compile "$SRC" "$OUT" \
  --root "." \
  --font-path "./fonts" \
  --font-path "./PyMC-Labs-New-Brand/Fonts/Inter/static" \
  --font-path "./PyMC-Labs-New-Brand/Fonts/Archivo/static"
echo "→ $OUT"
