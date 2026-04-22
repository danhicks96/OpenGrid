#!/usr/bin/env bash
# OpenGrid Quick Start — get a daemon running in 60 seconds
#
# Usage:
#   ./quickstart.sh                     # CPU-only node
#   ./quickstart.sh --gpu               # GPU node (NVIDIA)
#   ./quickstart.sh --model /path.gguf  # Load specific model on startup
#   ./quickstart.sh --orchestrator      # Orchestrator mode (coordinator + micro model)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
OPENGRID_HOME="${OPENGRID_HOME:-$HOME/.opengrid}"

echo "=== OpenGrid Quick Start ==="
echo ""

# Step 1: Python venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment exists."
fi

# Step 2: Install
echo "[2/4] Installing OpenGrid..."
"$VENV_DIR/bin/pip" install --quiet -e "$SCRIPT_DIR"

# Step 3: Install inference backend
GPU_FLAG=""
MODEL_FLAG=""
ORCH_FLAG=""
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --gpu)
            GPU_FLAG="yes"
            ;;
        --orchestrator)
            ORCH_FLAG="--orchestrator"
            ;;
        --model)
            # Next arg is the model path
            ;;
        *)
            if [ "$MODEL_FLAG" = "pending" ]; then
                MODEL_FLAG="--model $arg"
            fi
            ;;
    esac
    if [ "$arg" = "--model" ]; then
        MODEL_FLAG="pending"
    fi
done

# Handle --model /path properly
for i in $(seq 1 $#); do
    if [ "${!i}" = "--model" ]; then
        next=$((i+1))
        MODEL_FLAG="--model ${!next}"
        break
    fi
done

if [ "$GPU_FLAG" = "yes" ]; then
    echo "[3/4] Installing GPU backend (vLLM)..."
    "$VENV_DIR/bin/pip" install --quiet vllm 2>/dev/null || {
        echo "    vLLM install failed. Installing llama-cpp-python with GPU offload..."
        CMAKE_ARGS="-DGGML_CUDA=on" "$VENV_DIR/bin/pip" install --quiet llama-cpp-python
    }
else
    echo "[3/4] Installing CPU backend (llama.cpp)..."
    "$VENV_DIR/bin/pip" install --quiet llama-cpp-python 2>/dev/null || {
        echo "    llama-cpp-python install failed. Will use stub backend."
    }
fi

# Step 4: Run benchmark and start
echo "[4/4] Starting OpenGrid daemon..."
echo ""
mkdir -p "$OPENGRID_HOME"

"$VENV_DIR/bin/opengrid" benchmark > /dev/null 2>&1 || true

echo "    Node ID: $(cat "$OPENGRID_HOME/profile.json" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("node_id","unknown"))' 2>/dev/null || echo 'generating...')"
echo "    Tier:    $(cat "$OPENGRID_HOME/profile.json" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tier","unknown"))' 2>/dev/null || echo 'benchmarking...')"
echo "    API:     http://localhost:8080"
echo "    Worker:  ws://localhost:7600"
echo "    Gossip:  tcp://localhost:7601"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec "$VENV_DIR/bin/opengrid" serve $ORCH_FLAG $MODEL_FLAG $EXTRA_ARGS
