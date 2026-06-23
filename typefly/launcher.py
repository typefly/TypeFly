"""Single-command launcher for TypeFly.

Starts the two TypeFly processes in the right order with one command:
  1. the vision service (`python -m typefly.serving`),
  2. waits until its HTTP gateway reports healthy (covers the first-run YOLO
     weight download),
  3. the web UI (`python -m typefly.webui`).

Ctrl-C (or SIGTERM) shuts both down cleanly. The two-process architecture is
kept intact — this just orchestrates it.

Tunables (env):
  EDGE_SERVICE_IP        host to probe for gateway health (default 127.0.0.1)
  TYPEFLY_HEALTH_TIMEOUT seconds to wait for the gateway (default 180)
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request

from dotenv import load_dotenv

from typefly.serving.config import EDGE_SERVICE_PORT
from typefly.utils import print_t


def _gateway_up(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print_t("[TypeFly] OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
                "key (cp .env.example .env), or run: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    host = os.environ.get("EDGE_SERVICE_IP", "127.0.0.1")
    timeout = float(os.environ.get("TYPEFLY_HEALTH_TIMEOUT", "180"))

    procs: list[subprocess.Popen] = []

    def shutdown(*_):
        print_t("[TypeFly] Shutting down ...")
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 1. Vision service.
    serving = subprocess.Popen([sys.executable, "-m", "typefly.serving"])
    procs.append(serving)

    # 2. Wait for the gateway to become healthy.
    print_t(f"[TypeFly] Starting vision service; waiting for http://{host}:{EDGE_SERVICE_PORT}/health "
            f"(up to {int(timeout)}s; first run downloads YOLO weights) ...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if serving.poll() is not None:
            print_t("[TypeFly] Vision service exited before becoming healthy. See logs above.")
            sys.exit(1)
        if _gateway_up(host, EDGE_SERVICE_PORT):
            break
        time.sleep(1.0)
    else:
        print_t("[TypeFly] Vision service did not become healthy in time. "
                "Set TYPEFLY_HEALTH_TIMEOUT to wait longer.")
        shutdown()

    # 3. Web UI.
    print_t("[TypeFly] Vision service ready. Starting web UI on http://localhost:50000 ...")
    webui = subprocess.Popen([sys.executable, "-m", "typefly.webui"])
    procs.append(webui)

    # 4. Supervise: if either process exits, take the other down too.
    while True:
        for p in procs:
            if p.poll() is not None:
                print_t(f"[TypeFly] A TypeFly process exited (code {p.returncode}); shutting down.")
                shutdown()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
