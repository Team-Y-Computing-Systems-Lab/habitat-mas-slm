"""Serve a SimHost over HTTP/JSON.

    conda activate habitat-mas
    python -m mas.sim_host.server --benchmark hssd_fetch_stretch --port 8765

POST /rpc  {"method": "run_skill", "params": {"robot_id": "fetch_0", "skill": "pick", "args": {"object": "bowl_0"}}}
  -> {"result": ...}  or  {"error": {"code": "...", "message": ...}}

Requests are handled on threads so several robots' `run_skill` calls can be
open at once; the simulator itself only runs on the main thread (scheduler.py).
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .host import SimHost
from .scheduler import Scheduler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_handler(scheduler: Scheduler):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/rpc":
                return self._reply(404, {"error": {"code": "NOT_FOUND", "message": self.path}})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reply = scheduler.submit(req["method"], req.get("params", {}))
            except (ValueError, KeyError) as e:
                reply = {"error": {"code": "BAD_REQUEST", "message": repr(e)}}
            self._reply(500 if reply.get("error", {}).get("code") == "INTERNAL" else 200, reply)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="hssd_fetch_stretch")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-steps", type=int, default=3000,
                   help="episode step budget; EMOS uses 750, too few for multi-skill plans")
    p.add_argument("overrides", nargs="*", help="hydra overrides for the habitat config")
    args = p.parse_args()

    os.chdir(PROJECT_ROOT)  # habitat configs use paths relative to data/
    overrides = [
        f"habitat.environment.max_episode_steps={args.max_steps}",
    ] + args.overrides
    host = SimHost(args.benchmark, overrides)
    scheduler = Scheduler(host)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scheduler))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[sim_host] {args.benchmark}: pool={[r['id'] for r in host.robots]} "
          f"episodes={len(host.env.episodes)} listening on {args.host}:{args.port}", flush=True)
    scheduler.run_forever()  # the simulator stays on the main thread


if __name__ == "__main__":
    main()
