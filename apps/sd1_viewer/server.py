"""Dependency-free, read-only VOC dataset server, bound to loopback by default."""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlsplit
import webbrowser
import xml.etree.ElementTree as ET

STATIC = Path(__file__).parent / 'static'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'source' / 'SD-1'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class Dataset:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not all((self.root / name).is_dir() for name in ('imgs', 'Annotations')):
            raise ValueError(f'数据集目录必须包含 imgs 和 Annotations：{self.root}')
        images, annotations = {}, {}
        for folder, extensions, target in (
            ('imgs', IMAGE_EXTENSIONS, images), ('Annotations', {'.xml'}, annotations)
        ):
            for path in sorted((self.root / folder).rglob('*')):
                if path.is_file() and path.suffix.lower() in extensions:
                    # Pair by relative path, never by basename alone.
                    key = path.relative_to(self.root / folder).with_suffix('').as_posix()
                    if key in target:
                        raise ValueError(f'存在同名的多个文件，无法唯一配对：{key}')
                    target[key] = path
        self.rows = []
        for index, key in enumerate(sorted(images.keys() | annotations.keys())):
            image, annotation = images.get(key), annotations.get(key)
            row = dict(id=index, key=key, group=key.split('/')[0], image=image,
                       annotation=annotation, width=None, height=None, objects=[], error=None,
                       status='paired' if image and annotation else 'image_only' if image else 'annotation_only')
            if annotation:
                try:
                    tree = ET.parse(annotation).getroot()
                    for dimension in ('width', 'height'):
                        value = tree.findtext(f'size/{dimension}')
                        row[dimension] = int(value) if value else None
                    for obj in tree.findall('object'):
                        box = [float(obj.findtext(f'bndbox/{n}', 'nan'))
                               for n in ('xmin', 'ymin', 'xmax', 'ymax')]
                        import math
                        if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                            raise ValueError('无效的矩形坐标')
                        row['objects'].append(dict(label=obj.findtext('name') or '未命名', box=box,
                                                   difficult=obj.findtext('difficult', '0')))
                except (ET.ParseError, ValueError, OSError) as exc:
                    row['error'] = str(exc)
            row['labels'] = sorted({obj['label'] for obj in row['objects']})
            self.rows.append(row)

    @staticmethod
    def public(row):
        return {key: value for key, value in row.items() if key not in ('image', 'annotation', 'objects')}

    def summary(self):
        counts = Counter(row['status'] for row in self.rows)
        return dict(root=str(self.root), images=sum(row['image'] is not None for row in self.rows),
                    annotations=sum(row['annotation'] is not None for row in self.rows),
                    paired=counts['paired'], image_only=counts['image_only'],
                    annotation_only=counts['annotation_only'], invalid=sum(bool(row['error']) for row in self.rows),
                    labels=dict(sorted(Counter(obj['label'] for row in self.rows for obj in row['objects']).items())),
                    groups=sorted({row['group'] for row in self.rows}))

    def query(self, q='', label='', group='', status='', offset=0, limit=80):
        rows = [row for row in self.rows if
                (not q or q.casefold() in row['key'].casefold()) and
                (not label or label in row['labels']) and
                (not group or group == row['group']) and
                (not status or (bool(row['error']) if status == 'invalid' else row['status'] == status))]
        offset, limit = max(0, int(offset)), max(1, min(200, int(limit)))
        return dict(total=len(rows), items=[self.public(row) for row in rows[offset:offset + limit]])

    def detail(self, index):
        row = self.rows[int(index)]
        return dict(self.public(row), objects=row['objects'],
                    image_url=f'/image/{index}' if row['image'] else None,
                    image_path=str(row['image']) if row['image'] else None,
                    annotation_path=str(row['annotation']) if row['annotation'] else None)


def make_server(dataset, host='127.0.0.1', port=7875):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == '/api/summary':
                    return self.send_json(dataset.summary())
                if url.path == '/api/items':
                    params = {key: values[0] for key, values in parse_qs(url.query).items()
                              if key in {'q', 'label', 'group', 'status', 'offset', 'limit'}}
                    return self.send_json(dataset.query(**params))
                if url.path.startswith('/api/item/'):
                    return self.send_json(dataset.detail(self.index(url.path)))
                if url.path.startswith('/image/'):
                    path = dataset.rows[self.index(url.path)]['image']
                    if path:
                        return self.send_file(path)
                elif url.path in ('/', '/app.js', '/style.css'):
                    return self.send_file(STATIC / ('index.html' if url.path == '/' else url.path[1:]))
                self.send_error(404)
            except (ValueError, IndexError):
                self.send_error(404)
            except OSError:
                self.send_error(404, 'File unavailable')

        @staticmethod
        def index(path):
            token = path.rsplit('/', 1)[-1]
            if not token.isdecimal():
                raise ValueError('Invalid ID')
            return int(token)

        def send_json(self, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path):
            with path.open('rb') as stream:
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='SD-1 图片 / VOC 标注只读浏览器')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7875)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    print(f'正在扫描数据集：{args.dataset}', flush=True)
    try:
        dataset = Dataset(args.dataset)
        server = make_server(dataset, args.host, args.port)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    address = f'http://{args.host}:{server.server_port}'
    print(f'已索引 {len(dataset.rows)} 条记录。打开 {address} （Ctrl+C 退出）', flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
