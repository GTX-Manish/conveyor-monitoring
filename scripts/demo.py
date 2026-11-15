"""Run the demo: start the backend, wait until it answers, start the replayer, stop both on Ctrl-C.

    python -m scripts.demo            # then open http://127.0.0.1:8000
"""
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]


def wait_for_health(url: str, seconds: float = 60.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/api/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"the backend did not answer at {url}/api/health within {seconds:.0f} s")


def main(argv=None):
    settings = yaml.safe_load((REPO / "configs" / "app.yaml").read_text())
    server = settings["server"]
    ap = argparse.ArgumentParser(description="Run the demo.")
    ap.add_argument("--no-inject", action="store_true", help="do not add duplicate, out-of-range or out-of-order input")
    ap.add_argument("--speed", type=float, help="simulated hours per real second (default from configs/app.yaml)")
    args = ap.parse_args(argv)

    scenario_dir = REPO / server["data_dir"] / server["scenario"] / "public"
    if not scenario_dir.is_dir():
        print(f"no data for scenario {server['scenario']}; run: python -m simulator.scenario_set --only "
              f"{server['scenario']}", file=sys.stderr)
        return 1
    url = f"http://{server['host']}:{server['port']}"
    def interrupt(*_):          # a plain `kill` stops the children too, like Ctrl-C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    procs = [subprocess.Popen([sys.executable, "-m", "backend.app"], cwd=REPO)]
    try:
        health = wait_for_health(url)
        print(f"backend ready ({health['model_version'] or 'no model'}). Dashboard: {url}", flush=True)
        cmd = [sys.executable, "-m", "replayer.run", "--scenario", server["scenario"], "--url", url]
        if not args.no_inject:
            cmd.append("--inject-faults")
        if args.speed:
            cmd += ["--speed", str(args.speed)]
        procs.append(subprocess.Popen(cmd, cwd=REPO))
        procs[1].wait()
        print(f"replay finished. The dashboard stays up at {url}; press Ctrl-C to stop.", flush=True)
        procs[0].wait()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
