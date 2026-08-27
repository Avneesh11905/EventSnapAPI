#!/usr/bin/env bash
# generate.sh — Regenerate Python stubs from inference.proto
# Run this from the inference_api/ root directory.
# Requires grpcio-tools (already in pyproject.toml dependencies).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_API_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROTO_DIR="$SCRIPT_DIR"
OUT_DIR="$SCRIPT_DIR"

echo "Generating gRPC stubs from inference.proto..."
echo "  Source : $PROTO_DIR/inference.proto"
echo "  Output : $OUT_DIR"

python -m grpc_tools.protoc \
    --proto_path="$MAIN_API_ROOT" \
    --python_out="$MAIN_API_ROOT" \
    --grpc_python_out="$MAIN_API_ROOT" \
    "infrastructure/inference/proto/inference.proto"

echo "✅ Done. Generated files:"
echo "   $OUT_DIR/inference_pb2.py"
echo "   $OUT_DIR/inference_pb2_grpc.py"
