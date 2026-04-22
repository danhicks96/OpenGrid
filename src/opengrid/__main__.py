"""CLI entry point: `opengrid`"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="opengrid", description="OpenGrid node CLI")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Start the API server + worker node")
    serve.add_argument("--orchestrator", action="store_true",
                       help="Enable local orchestrator micro-model")
    serve.add_argument("--port", type=int, default=None,
                       help="API port (default: 8080)")
    serve.add_argument("--model", type=str, default=None,
                       help="Path to GGUF model file to load on startup")
    serve.add_argument("--gpu-layers", type=int, default=0,
                       help="Number of layers to offload to GPU (llama.cpp)")

    sub.add_parser("benchmark", help="Run hardware benchmark and save profile")
    sub.add_parser("status", help="Show network status and local node info")

    dl = sub.add_parser("download", help="Download a model from HuggingFace")
    dl.add_argument("model_url", help="HuggingFace model URL or repo ID")
    dl.add_argument("--output", "-o", type=str, default=None,
                    help="Output path (default: ~/.opengrid/models/)")

    args = parser.parse_args()

    if args.cmd == "serve":
        import os
        import uvicorn
        from opengrid.daemon.config import load_config
        cfg = load_config()
        if args.orchestrator:
            os.environ["OPENGRID_MODE"] = "orchestrator"
        if args.model:
            os.environ["OPENGRID_MODEL_PATH"] = args.model
        if args.gpu_layers:
            os.environ["OPENGRID_GPU_LAYERS"] = str(args.gpu_layers)
        port = args.port or cfg.network.api_port
        uvicorn.run("opengrid.api.server:app", host="0.0.0.0", port=port, reload=False)

    elif args.cmd == "benchmark":
        import json, dataclasses
        from opengrid.daemon.benchmark import run_benchmark
        print(json.dumps(dataclasses.asdict(run_benchmark()), indent=2))

    elif args.cmd == "status":
        import json, dataclasses
        from opengrid.daemon.config import load_config
        from opengrid.daemon.benchmark import load_or_run
        cfg = load_config()
        profile = load_or_run(cfg.profile_path)
        print(json.dumps(dataclasses.asdict(profile), indent=2))

    elif args.cmd == "download":
        from pathlib import Path
        from opengrid.daemon.config import load_config, ensure_dirs
        cfg = load_config()
        ensure_dirs(cfg)
        models_dir = Path(args.output) if args.output else cfg.home / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore
            print(f"Downloading {args.model_url} to {models_dir}...")
            path = snapshot_download(args.model_url, local_dir=str(models_dir / args.model_url.split("/")[-1]))
            print(f"Downloaded to: {path}")
        except ImportError:
            print("huggingface_hub not installed. Run: pip install huggingface_hub")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
