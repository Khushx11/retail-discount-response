"""Serve the retail study dashboard on your computer."""
import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "docs/figures"
ASSETS = ROOT / "portfolio"
RESULTS = ROOT / "docs/results"
FILES = {
    "/": (ASSETS / "index.html", "text/html"),
    "/style.css": (ASSETS / "style.css", "text/css"),
    "/app.js": (ASSETS / "app.js", "text/javascript"),
    "/figures/ablations.png": (FIGURES / "ablations.png", "image/png"),
    "/figures/response.png": (FIGURES / "response.png", "image/png"),
}


def read_rows(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def payload(evidence=None):
    support = [
        {"discount": round(float(row["discount"]) * 100),
         "outside": float(row["outside_validation_distance_share"])}
        for row in read_rows("joint_support.csv")
    ]
    summary = [
        {"model": row["model"], "wape": float(row["WAPE"]), "runs": int(row["runs"])}
        for row in read_rows("summary.csv")
    ]
    verification = json.loads((RESULTS / "verification.json").read_text())
    scenarios = None
    if evidence is not None:
        runner = ROOT / "work/final_report/robust/scenario_grid.py"
        process = subprocess.run(
            [sys.executable, str(runner)], cwd=evidence,
            check=True, capture_output=True, text=True, timeout=180,
        )
        scenarios = json.loads(process.stdout)
    return {
        "rows": verification["test_rows"], "summary": summary, "support": support,
        "scenarios": scenarios,
    }


def serve(port, report):
    response = json.dumps(report, allow_nan=False).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/study":
                kind, body = "application/json", response
            elif self.path in FILES:
                path, kind = FILES[self.path]
                body = path.read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, help="Local extracted COMP5064_Evidence directory")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    try:
        report = payload(args.evidence)
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr or "Unable to replay the local evidence.\n")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Unable to load the study: {error}\n")
    serve(args.port, report)


if __name__ == "__main__":
    main()
