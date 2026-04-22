"""CLI entry point: `opengrid`"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="opengrid", description="OpenGrid node CLI")
    sub = parser.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="Start the API server")
    serve.add_argument("--orchestrator", action="store_true",
                       help="Enable local orchestrator micro-model")
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("benchmark", help="Run hardware benchmark and save profile")
    sub.add_parser("status", help="Show network status and local node info")

    args = parser.parse_args()

    if args.cmd == "serve":
        import os
        import uvicorn
        from opengrid.daemon.config import load_config
        cfg = load_config()
        if args.orchestrator:
            os.environ["OPENGRID_MODE"] = "orchestrator"
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
        print(json.dumps(dataclasses.asdict(load_or_run(cfg.profile_path)), indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
