#!/bin/bash
# Build index on fast machine (M2), ship to VPS
# Usage: ./build_index.sh [vps_host]
# Contoh: ./build_index.sh root@vmi3097120

set -e

VPS="${1:-root@vmi3097120}"
FOLDER="${2:-files}"

echo "=== Building index locally (M2)... ==="
cd core
cargo build --release 2>/dev/null
./target/release/onod benchmark "../${FOLDER}/" --no-cache 2>&1 | grep -E "Index:|Recall"
cd ..

echo ""
echo "=== Shipping index.bin to ${VPS}... ==="
scp "${FOLDER}/index.bin" "${VPS}:~/onod/${FOLDER}/"

echo ""
echo "=== Done! Index deployed. ==="
echo "Di VPS, langsung run:"
echo "  cd ~/onod/core && ./target/release/onod benchmark ../${FOLDER}/"
