"""Loopback HTTP app: browse source images, run matcher, inspect exported predictions."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import shutil
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .data import Catalog, DEFAULT_CONFIG, DEFAULT_IMAGES, DEFAULT_RESULTS, FILTERS, ROOT, artifact_path
from .jobs import Jobs

STATIC = Path(__file__).parent / 'static'


def make_server(catalog, jobs=None, host='127.0.0.1', port=7876):
    jobs = jobs or Jobs(catalog)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
            try:
                if url.path == '/api/summary':
                    return self.send_json(dict(catalog.summary(), token=token, job=jobs.snapshot()))
                if url.path == '/api/items':
                    return self.send_json(catalog.query(**{k: v for k, v in params.items()
                                                           if k in FILTERS | {'offset', 'limit'}}))
                if url.path.startswith('/api/item/'):
                    return self.send_json(catalog.detail(url.path.rsplit('/', 1)[-1]))
                if url.path.startswith('/preview/'):
                    identifier = url.path.rsplit('/', 1)[-1]
                    return self.send_bytes(catalog.preview(identifier, **params), 'image/jpeg')
                if url.path.startswith('/artifact/'):
                    parts = url.path.split('/')
                    if len(parts) != 4:
                        raise ValueError('无效文件请求')
                    with catalog.lock:
                        row = catalog.rows[parts[2]]
                        if not row['_manifest']:
                            raise ValueError('尚无推理结果')
                        path = artifact_path(row['_manifest'], row['_metadata'], parts[3])
                    return self.send_file(path)
                if url.path in ('/', '/app.js', '/style.css'):
                    return self.send_file(STATIC / ('index.html' if url.path == '/' else url.path[1:]))
                self.send_json({'error': '不存在该资源'}, 404)
            except (ValueError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except (KeyError, OSError) as exc:
                self.send_json({'error': str(exc)}, 404)

        def do_POST(self):
            # Custom token plus same-origin check prevents other sites submitting local jobs.
            origin = self.headers.get('Origin')
            expected = f'http://{self.headers.get("Host", "")}'
            if self.headers.get('X-Viewer-Token') != token or (origin and origin != expected):
                return self.send_json({'error': '请求来源或会话令牌无效'}, 403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    raise ValueError('无效请求大小')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('请求必须为 JSON 对象')
                path = urlsplit(self.path).path
                if path == '/api/jobs':
                    if set(body) - {'ids', 'filters', 'skip_completed'}:
                        raise ValueError('未知任务参数')
                    return self.send_json(jobs.start(**body))
                if path == '/api/stop':
                    return self.send_json(jobs.stop())
                if path == '/api/refresh':
                    if jobs.snapshot()['active']:
                        raise RuntimeError('请在任务结束后重新扫描')
                    return self.send_json(dict(catalog.refresh(), token=token, job=jobs.snapshot()))
                self.send_json({'error': '不存在该操作'}, 404)
            except (ValueError, TypeError, OSError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except RuntimeError as exc:
                self.send_json({'error': str(exc)}, 409)

        def send_json(self, data, status=200):
            self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'),
                            'application/json; charset=utf-8', status)

        def send_bytes(self, body, content_type, status=200):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store' if content_type.startswith('application/json') else 'private, max-age=30')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_file(self, path):
            with path.open('rb') as stream:
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    shutil.copyfileobj(stream, self.wfile)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def log_message(self, fmt, *args):
            if not self.path.startswith('/preview/'):
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.jobs = jobs
    return server


def main():
    parser = argparse.ArgumentParser(description='SD-1 matcher 批量推理 / 多图结果浏览器')
    parser.add_argument('--images', type=Path, default=DEFAULT_IMAGES)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--import-result', type=Path, action='append', default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7876)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    imports = args.import_result if args.import_result is not None else [ROOT / 'periodic_template_matcher/results/sd1_opencv']
    print(f'正在扫描图片：{args.images}', flush=True)
    try:
        catalog = Catalog(args.images, args.config, args.results, imports)
        server = make_server(catalog, host=args.host, port=args.port)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    address = f'http://{args.host}:{server.server_port}'
    print(f'已索引 {len(catalog.rows)} 张图片。打开 {address}，Ctrl+C 退出。', flush=True)
    print('只在点击批量匹配后运行推理；原图、当前配置及历史结果不会被覆盖。', flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.stop()
        # Complete the active image before interpreter teardown; queued images are cancelled.
        if server.jobs.thread and server.jobs.thread.is_alive():
            print('正在完成当前图片后退出……', flush=True)
            server.jobs.thread.join()
        server.server_close()
