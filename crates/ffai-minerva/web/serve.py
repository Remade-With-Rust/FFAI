"""Serve Minerva's browser build locally: python web/serve.py MODEL.gguf [--prompts DIR] [--port 8765]

Serves web/www, with /model.gguf streamed from MODEL.gguf (read-only) and /p64.ids from --prompts.
A page that POSTs to /result has its body printed; with --once the server then exits (the
headless-browser check uses that).
"""
import argparse, http.server, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--prompts', default=os.environ.get('BITNET_PROMPTS'))
ap.add_argument('--port', type=int, default=8765)
ap.add_argument('--once', action='store_true')
a = ap.parse_args()
WWW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'www')
TYPES = {'.js': 'text/javascript', '.wasm': 'application/wasm', '.html': 'text/html; charset=utf-8',
         '.ids': 'text/plain', '.css': 'text/css', '.json': 'application/json'}


class H(http.server.BaseHTTPRequestHandler):
    def send_file(self, path, ctype):
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        # Cross-origin isolation, so the threaded build can share memory between workers.
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.end_headers()
        with open(path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def do_GET(self):
        p = self.path.split('?')[0].lstrip('/') or 'index.html'
        if p == 'model.gguf':
            return self.send_file(a.model, 'application/octet-stream')
        if p.endswith('.ids') and a.prompts:
            f = os.path.join(a.prompts, os.path.basename(p))
            if os.path.isfile(f):
                return self.send_file(f, 'text/plain')
        f = os.path.normpath(os.path.join(WWW, p))
        if not f.startswith(WWW) or not os.path.isfile(f):
            self.send_error(404)
            return
        self.send_file(f, TYPES.get(os.path.splitext(f)[1], 'application/octet-stream'))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.send_response(204)
        self.end_headers()
        if self.path.startswith('/log'):
            print('LOG ' + body.decode('utf-8', 'replace'), flush=True)
        if self.path.startswith('/result'):
            print('RESULT ' + body.decode('utf-8', 'replace'), flush=True)
            if a.once:
                os._exit(0)

    def log_message(self, *args):
        pass


print(f'serving http://127.0.0.1:{a.port}/ (model {a.model})', flush=True)
http.server.ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
