# Deploying an OpenGrid Daemon

## Quick Start (60 seconds)

```bash
git clone https://github.com/andromaliusai/OpenGrid.git
cd OpenGrid
./quickstart.sh
```

That's it. Your machine is now an OpenGrid node.

## With a GPU (NVIDIA)

```bash
./quickstart.sh --gpu
```

This installs vLLM (if CUDA available) or llama.cpp with GPU offload.

## With a specific model

```bash
# Download a small model for testing
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models

# Start with that model loaded
./quickstart.sh --model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Connecting to an orchestrator

Set the `OPENGRID_BOOTSTRAP_PEERS` env var to point at the orchestrator:

```bash
OPENGRID_BOOTSTRAP_PEERS="orchestrator-ip:7600" ./quickstart.sh --gpu
```

## Docker

```bash
cd docker

# CPU only
docker compose up

# With GPU
docker compose --profile gpu up

# With a model mounted in
docker compose up -e OPENGRID_MODEL_PATH=/root/.opengrid/models/my-model.gguf
```

## Configuration

On first run, OpenGrid creates `~/.opengrid/config.toml`. Edit it to control:

```toml
[resources]
max_gpu_fraction     = 0.50       # How much GPU to donate
max_vram_gb          = 6.0        # VRAM budget for model shards
max_ram_gb           = 4.0        # RAM budget for KV cache
max_disk_gb          = 40.0       # Disk budget for shard storage

[schedule]
active_hours_start   = "08:00"    # Only serve during these hours
active_hours_end     = "17:00"
pause_on_battery     = true       # Laptops: stop when unplugged
pause_when_gaming    = true       # Pause if GPU load > 80%
```

## Checking your node

```bash
# See your hardware profile
opengrid status

# Hit the API
curl http://localhost:8080/health
curl http://localhost:8080/v1/network/status
curl http://localhost:8080/v1/credits/balance \
  -H "Authorization: Bearer dev-key-change-me"
```
