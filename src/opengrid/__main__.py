"""
CLI entry point: `opengrid`
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="opengrid", description="OpenGrid node CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="Start the API server")
    bench = sub.add_parser("benchmark", help="Run hardware benchmark and save profile")
    sub.add_parser("status", help="Show network status and local node info")

    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn
        from opengrid.daemon.config import load_config
        cfg = load_config()
        uvicorn.run("opengrid.api.server:app", host="0.0.0.0",
                    port=cfg.network.api_port, reload=False)

    elif args.cmd == "benchmark":
        import json
        from opengrid.daemon.benchmark import run_benchmark
        profile = run_benchmark()
        import dataclasses
        print(json.dumps(dataclasses.asdict(profile), indent=2))

    elif args.cmd == "status":
        import json
        from opengrid.daemon.config import load_config
        from opengrid.daemon.benchmark import load_or_run
        cfg = load_config()
        profile = load_or_run(cfg.profile_path)
        import dataclasses
        print(json.dumps(dataclasses.asdict(profile), indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
