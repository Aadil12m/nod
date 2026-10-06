"""`nod-gateway` console script.

    nod-gateway --mock                 # CPU only: MockRunner, FakePlanner, FakeASR, FakeCosmos, FakeHome
    nod-gateway                        # NOD_MODE from the environment (default mock)
    NOD_MODE=live nod-gateway --host 0.0.0.0

Any `NOD_*` environment variable overrides a config field (see config.py).
"""

from __future__ import annotations

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nod-gateway", description="Nod gateway (FastAPI)")
    ap.add_argument("--mock", action="store_true", help="mock mode: no GPU, no API keys (sets NOD_MODE=mock)")
    ap.add_argument("--live", action="store_true", help="live mode (sets NOD_MODE=live)")
    ap.add_argument("--host", default=os.environ.get("NOD_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("NOD_PORT", "8000")))
    ap.add_argument("--pwa-dir", help="directory with index.html and presenter.html")
    ap.add_argument("--runs-dir", help="where run traces are written (default ./runs)")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)
    if args.mock and args.live:
        ap.error("--mock and --live are exclusive")
    if args.mock:
        os.environ["NOD_MODE"] = "mock"
    if args.live:
        os.environ["NOD_MODE"] = "live"
    if args.pwa_dir:
        os.environ["NOD_PWA_DIR"] = args.pwa_dir
    if args.runs_dir:
        os.environ["NOD_RUNS_DIR"] = args.runs_dir
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from .app import create_app
    from .config import Config

    cfg = Config()
    logging.getLogger(__name__).info(
        "starting nod-gateway mode=%s planner=%s asr=%s cosmos=%s home=%s",
        cfg.mode,
        cfg.planner_kind,
        cfg.asr_kind,
        cfg.cosmos_kind,
        cfg.home_kind,
    )
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level=args.log_level, ws_ping_interval=None)


if __name__ == "__main__":
    main()
