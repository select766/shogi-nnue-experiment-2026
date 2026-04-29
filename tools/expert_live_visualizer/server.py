#!/usr/bin/env python3
"""
Expert Live Visualizer server.

Reads YaneuraOu stdout log and serves the latest blending weights via HTTP.
Document root is the project root (two levels up from this script).

Usage:
    python server.py --log /tmp/yaneuraou.log [--port 8765]
"""

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tools/expert_live_visualizer -> project root

WEIGHT_RE = re.compile(r'info string blending_weight=\[([^\]]+)\]')

CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.svg':  'image/svg+xml',
    '.png':  'image/png',
    '.js':   'text/javascript',
    '.css':  'text/css',
    '.json': 'application/json',
}


def read_latest_weights(log_path: str) -> list[float] | None:
    """Scan the tail of the log file for the most recent blending_weight line."""
    try:
        p = Path(log_path)
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode('utf-8', errors='replace')
        matches = list(WEIGHT_RE.finditer(tail))
        if not matches:
            return None
        values_str = matches[-1].group(1)
        return [float(v.strip()) for v in values_str.split(',')]
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    log_path: str | None = None

    def do_GET(self):
        path = self.path.split('?', 1)[0]  # strip query string
        if path == '/api/weights':
            self._serve_weights()
        elif path in ('/', '/index.html'):
            self._serve_file(SCRIPT_DIR / 'index.html')
        else:
            self._serve_static(path)

    def _serve_weights(self):
        weights = read_latest_weights(self.log_path) if self.log_path else None
        body = json.dumps({'weights': weights, 'timestamp': time.time()}).encode()
        self._send(200, 'application/json', body, extra_headers={'Access-Control-Allow-Origin': '*'})

    def _serve_static(self, url_path: str):
        target = (PROJECT_ROOT / url_path.lstrip('/')).resolve()
        # Prevent directory traversal outside project root
        if not str(target).startswith(str(PROJECT_ROOT)):
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        ct = CONTENT_TYPES.get(target.suffix.lower(), 'application/octet-stream')
        self._serve_file(target, ct)

    def _serve_file(self, path: Path, ct: str = 'text/html; charset=utf-8'):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self._send(200, ct, data)

    def _send(self, code: int, ct: str, body: bytes, extra_headers: dict | None = None):
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', len(body))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress repetitive polling noise; only log non-200 responses
        if args and len(args) >= 2 and not str(args[1]).startswith('2'):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description='Expert Live Visualizer server')
    parser.add_argument('--log', default=None, metavar='FILE',
                        help='YaneuraOu stdout log file (omit for demo-only mode)')
    parser.add_argument('--port', type=int, default=8765,
                        help='HTTP port (default: 8765)')
    args = parser.parse_args()

    Handler.log_path = args.log

    server = HTTPServer(('localhost', args.port), Handler)
    log_desc = args.log if args.log else '(none — browser runs in demo mode)'
    print(f'Expert Live Visualizer')
    print(f'  URL : http://localhost:{args.port}/')
    print(f'  Log : {log_desc}')
    print(f'  Root: {PROJECT_ROOT}')
    print('Ctrl-C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
