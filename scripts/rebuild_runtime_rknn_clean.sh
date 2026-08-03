#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${VISIONOPS_EDGE_ROOT:-/opt/visionops_v3}"
BUILD_DIR="${VISIONOPS_RUNTIME_BUILD_DIR:-${EDGE_ROOT}/build-rknn}"
JOBS="${VISIONOPS_BUILD_JOBS:-4}"

cd "${EDGE_ROOT}"
echo "[INFO] removing stale runtime build: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISIONOPS_ENABLE_RKNN=ON \
  -DVISIONOPS_ENABLE_OPENCV=ON \
  -DVISIONOPS_ENABLE_RGA=ON \
  -DVISIONOPS_RKNN_INCLUDE_DIR="${VISIONOPS_RKNN_INCLUDE_DIR:-/usr/include}" \
  -DVISIONOPS_RKNN_LIBRARY="${VISIONOPS_RKNN_LIBRARY:-/usr/lib/librknnrt.so}" \
  -DVISIONOPS_RGA_INCLUDE_DIR="${VISIONOPS_RGA_INCLUDE_DIR:-/usr/include}" \
  -DVISIONOPS_RGA_LIBRARY="${VISIONOPS_RGA_LIBRARY:-/usr/lib/librga.so}"

cmake --build "${BUILD_DIR}" -j"${JOBS}"
echo "[OK] runtime clean rebuild completed"
