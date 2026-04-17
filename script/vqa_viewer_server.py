import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rotate VLM Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    img { max-width: 320px; margin: 8px; border: 1px solid #ccc; }
    pre { white-space: pre-wrap; background: #f5f5f5; padding: 12px; }
    .sample { margin-bottom: 48px; }
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def render_samples(samples):
    chunks = []
    for idx, sample in enumerate(samples):
        images = "".join(f'<img src="/file?path={Path(path).as_posix()}">' for path in sample.get("images", []))
        assistant = sample.get("messages", [{}])[-1].get("content", "")
        metadata = json.dumps(sample.get("metadata", {}), ensure_ascii=False, indent=2)
        chunks.append(
            f'<div class="sample"><h3>Sample {idx}</h3>{images}<pre>{assistant}</pre><pre>{metadata}</pre></div>'
        )
    return HTML_TEMPLATE.format(body="\n".join(chunks))


class ViewerHandler(BaseHTTPRequestHandler):
    samples = []

    def do_GET(self):
        if self.path.startswith("/file?path="):
            raw_path = self.path.split("=", 1)[1]
            target = Path(raw_path)
            if not target.exists():
                self.send_error(404)
                return
            payload = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        body = render_samples(self.samples)
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="Serve rotate VLM samples in a simple browser viewer.")
    parser.add_argument("--samples-path", required=True, type=str)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ViewerHandler.samples = json.loads(Path(args.samples_path).read_text(encoding="utf-8"))
    server = HTTPServer(("0.0.0.0", int(args.port)), ViewerHandler)
    print(f"serving {args.samples_path} on http://0.0.0.0:{int(args.port)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
