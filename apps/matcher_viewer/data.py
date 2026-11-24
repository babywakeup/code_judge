"""Image catalog and immutable, configuration-aware inference cache."""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading

from apps.sd1_viewer.server import IMAGE_EXTENSIONS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = ROOT / 'source' / 'SD-1' / 'imgs'
DEFAULT_CONFIG = ROOT / 'periodic_template_matcher' / 'config.sd1.labelme.opencv.json'
DEFAULT_RESULTS = ROOT / 'periodic_template_matcher' / 'results' / 'sd1_gallery'
FILTERS = {'q', 'group', 'status', 'template', 'station', 'severity', 'date_from',
           'date_to', 'min_score', 'sort'}
PATH_FIELDS = ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid')


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('JSON 必须为对象')
    return value


def finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def artifact_path(manifest: Path, metadata: dict, kind: str) -> Path:
    name = metadata.get('artifacts', {}).get(kind)
    if not isinstance(name, str) or not name:
        raise ValueError('不存在该结果文件')
    path = (manifest.parent / name).resolve()
    if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
        raise ValueError('结果文件越界或不存在')
    return path


def inspect_manifest(manifest: Path) -> dict:
    metadata = read_json(manifest)
    ranking, matches = metadata.get('ranking'), metadata.get('matches')
    if (metadata.get('format_version') != 1 or not isinstance(metadata.get('artifacts'), dict)
            or not isinstance(matches, list) or not isinstance(ranking, list) or not ranking
            or type(ranking[0]) is not int or not 0 <= ranking[0] < len(matches)
            or not isinstance(matches[ranking[0]], dict)
            or finite(matches[ranking[0]].get('match_score')) is None
            or metadata.get('prediction_status') not in
            ('available', 'match_rejected', 'no_template_annotations')):
        raise ValueError('不支持或损坏的 matcher 结果')
    if metadata['prediction_status'] == 'available' and not metadata.get('accepted'):
        raise ValueError('拒绝的匹配不能包含可用预测')
    artifact_path(manifest, metadata, 'input')
    if metadata['prediction_status'] == 'available':
        artifact_path(manifest, metadata, 'prediction_mask')
        artifact_path(manifest, metadata, 'prediction_valid')
    return metadata


class Catalog:
    def __init__(self, images=DEFAULT_IMAGES, config=DEFAULT_CONFIG, results=DEFAULT_RESULTS,
                 imports=()):
        self.images, self.config, self.results = (Path(p).resolve() for p in (images, config, results))
        if not self.images.is_dir():
            raise ValueError(f'图片目录不存在：{self.images}')
        # Cache writes must never land in the source tree or template inputs.
        if self.results.is_relative_to(self.images) or self.images.is_relative_to(self.results):
            raise ValueError('结果目录与图片目录不能互相包含')
        self.imports = tuple(Path(p).resolve() for p in imports)
        self.lock = threading.RLock()
        self.preview_lock = threading.Lock()
        self.render_slots = threading.BoundedSemaphore(2)
        self.previews = OrderedDict()
        self.rows = {}
        self.warnings = []
        self.spec = read_json(self.config)
        if not isinstance(self.spec.get('templates'), list) or not self.spec['templates']:
            raise ValueError('配置必须包含模板')
        digest = hashlib.sha256(self.config.read_bytes())
        for template in self.spec['templates']:
            for field in PATH_FIELDS:
                if template.get(field):
                    path = (self.config.parent / template[field]).resolve()
                    if path.is_relative_to(self.results):
                        raise ValueError('模板输入不能位于结果目录')
                    digest.update(str(path).encode())
                    digest.update(path.read_bytes())
        # A per-target mask in a single-image config must not be reused for every image.
        if self.spec.get('target_valid_mask'):
            raise ValueError('批量配置不能包含单图 target_valid_mask，请在 options.step1 设置公共有效区边界')
        self.config_digest = digest.hexdigest()
        self.refresh()

    def fingerprint(self, path: Path) -> str:
        stat = path.stat()
        return hashlib.sha256(f'{self.config_digest}|{path}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()

    @staticmethod
    def public(row):
        return {k: v for k, v in row.items() if not k.startswith('_')}

    def refresh(self):
        with self.lock:
            previous = self.rows
            self.rows, self.warnings = {}, []
            for path in sorted(self.images.rglob('*')):
                if (not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS
                        or not path.resolve().is_relative_to(self.images)):
                    continue
                key = path.relative_to(self.images).as_posix()
                identifier = hashlib.sha256(key.encode()).hexdigest()[:24]
                tokens = path.stem.split('_')
                row = dict(id=identifier, key=key, group=key.split('/')[0] if '/' in key else '(根目录)',
                           station=tokens[2] if len(tokens) >= 10 else '',
                           severity=tokens[-3] if len(tokens) >= 10 else '',
                           date=tokens[-2] if len(tokens) >= 10 else '',
                           status='pending', score=None, template='', coverage=None, error=None,
                           result=False, elapsed=None, timing_scope=None, provenance=None,
                           _image=path.resolve(), _manifest=None, _metadata=None,
                           _fingerprint=self.fingerprint(path))
                old = previous.get(identifier)
                if old and old['_fingerprint'] == row['_fingerprint']:
                    row = old
                self.rows[identifier] = row
            by_image = {str(r['_image']): r for r in self.rows.values()}
            if self.results.is_dir():
                entries = list(self.results.glob('*/*/receipt.json')) + list(self.results.glob('*/*/error.json'))
                for receipt in sorted(entries):
                    if not receipt.resolve().is_relative_to(self.results):
                        continue
                    try:
                        entry = read_json(receipt)
                        row = self.rows.get(entry.get('id'))
                        if row is None or entry.get('fingerprint') != row['_fingerprint'] or row['status'] == 'running':
                            continue
                        if receipt.name == 'error.json':
                            row.update(status='error', error=str(entry.get('error', '推理失败')),
                                       result=False, _manifest=None, _metadata=None, score=None,
                                       template='', coverage=None, elapsed=finite(entry.get('elapsed')),
                                       timing_scope=entry.get('timing_scope', 'including_image_save'))
                            continue
                        manifest = receipt.parent / 'result.json'
                        if not manifest.resolve().is_relative_to(self.results):
                            raise ValueError('结果清单越界')
                        self.attach(row, manifest, entry.get('elapsed'), 'cache',
                                    timing_scope=entry.get('timing_scope', 'including_image_save'))
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        self.warnings.append(f'{receipt}: {exc}')
            for root in self.imports:
                manifests = [root] if root.is_file() else [root / 'result.json']
                for manifest in manifests:
                    try:
                        metadata = inspect_manifest(manifest)
                        target = metadata.get('target')
                        if not isinstance(target, str):
                            continue
                        row = by_image.get(str(Path(target).resolve()))
                        if row is None or row['result'] or row['status'] in ('running', 'error'):
                            continue
                        # Only associate legacy output if frozen input pixels equal today's source.
                        import numpy as np
                        from periodic_template_matcher.utils import read_image
                        if not np.array_equal(read_image(row['_image']),
                                              read_image(artifact_path(manifest, metadata, 'input'))):
                            self.warnings.append(f'忽略与当前源图不同的历史结果：{manifest}')
                            continue
                        self.attach(row, manifest, None, 'imported')
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        self.warnings.append(f'{manifest}: {exc}')
            return self.summary()

    def attach(self, row, manifest, elapsed=None, provenance='cache', *, timing_scope=None):
        manifest = manifest.resolve()
        metadata = inspect_manifest(manifest)
        best = metadata['matches'][metadata['ranking'][0]]
        names = {'available': 'available', 'match_rejected': 'rejected',
                 'no_template_annotations': 'no_annotations'}
        row.update(status=names[metadata['prediction_status']], score=finite(best['match_score']),
                   template=best.get('template_name', ''),
                   coverage=finite(metadata.get('label_coverage_full_image')), result=True,
                   error=None, elapsed=finite(elapsed), timing_scope=timing_scope, provenance=provenance,
                   _manifest=manifest, _metadata=metadata)

    def summary(self):
        with self.lock:
            rows = list(self.rows.values())
            return dict(total=len(rows), counts=dict(Counter(r['status'] for r in rows)),
                        groups=sorted({r['group'] for r in rows}),
                        templates=sorted({s.get('name', f'template_{i + 1}')
                                          for i, s in enumerate(self.spec['templates'])}
                                         | {r['template'] for r in rows if r['template']}),
                        stations=sorted({r['station'] for r in rows if r['station']}),
                        severities=sorted({r['severity'] for r in rows if r['severity']}),
                        warnings=list(self.warnings), images=str(self.images),
                        config=str(self.config), results=str(self.results))

    def select(self, filters):
        if set(filters) - FILTERS:
            raise ValueError('未知筛选条件')
        if any(not isinstance(v, str) for v in filters.values()):
            raise ValueError('筛选值必须是字符串')
        minimum = float(filters['min_score']) if filters.get('min_score') else None
        if minimum is not None and not math.isfinite(minimum):
            raise ValueError('无效的分数')
        rows = []
        for row in self.rows.values():
            if filters.get('q', '').casefold() not in row['key'].casefold():
                continue
            if any(filters.get(k) and filters[k] != row[k]
                   for k in ('group', 'status', 'template', 'station', 'severity')):
                continue
            if filters.get('date_from') and row['date'] < filters['date_from'].replace('-', ''):
                continue
            if filters.get('date_to') and row['date'] > filters['date_to'].replace('-', ''):
                continue
            if minimum is not None and (row['score'] is None or row['score'] < minimum):
                continue
            rows.append(row)
        sort = filters.get('sort', 'key')
        if sort not in ('key', 'score_desc', 'score_asc'):
            raise ValueError('未知排序方式')
        rows.sort(key=lambda r: (r['score'] is None,
                                (-r['score'] if sort == 'score_desc' else r['score'])
                                if r['score'] is not None else 0, r['key'])
                  if sort != 'key' else (r['key'],))
        return rows

    def query(self, offset=0, limit=80, **filters):
        offset, limit = max(0, int(offset)), max(1, min(200, int(limit)))
        with self.lock:
            rows = self.select(filters)
            return dict(total=len(rows), offset=offset, limit=limit,
                        items=[self.public(r) for r in rows[offset:offset + limit]])

    def detail(self, identifier):
        with self.lock:
            row = self.rows[identifier]
            metadata = row['_metadata']
            regions = (metadata or {}).get('valid_regions')
            target_region = regions.get('target') if isinstance(regions, dict) else None
            bounds = target_region.get('valid_bounds') if isinstance(target_region, dict) else None
            return dict(self.public(row), metadata=metadata, valid_bounds=bounds,
                        image_url=f'/preview/{identifier}?mode=input',
                        result_path=str(row['_manifest']) if row['_manifest'] else None,
                        artifacts=row['_metadata'].get('artifacts', {}) if row['_metadata'] else {})

    def preview(self, identifier, mode='overlay', width='480', alpha='.5', source='all', unknown='1', **extra):
        if mode not in ('input', 'overlay', 'mask', 'split') or source not in ('all', 'fused', 'direct', 'periodic', 'objects'):
            raise ValueError('无效预览模式')
        width, alpha = max(64, min(1920, int(width))), float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('透明度须为 0..1')
        with self.lock:
            row = dict(self.rows[identifier])
        stamp = row['_manifest'].stat().st_mtime_ns if row['_manifest'] else row['_fingerprint']
        key = (identifier, stamp, mode, width, alpha, source, unknown)
        with self.preview_lock:
            if key in self.previews:
                self.previews.move_to_end(key)
                return self.previews[key]
        with self.render_slots:
            import cv2
            from periodic_template_matcher.app.data import load_result, render_view
            from periodic_template_matcher.utils import read_image
            if row['_manifest']:
                prediction = load_result(row['_manifest'])
                image = prediction.image
            else:
                prediction = None
                image = read_image(row['_image'])
            scale = min(1., width / (image.shape[1] * (2 if mode == 'split' and prediction else 1)))
            size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
            if prediction:
                if mode == 'split':
                    size = (size[0] * 2, size[1])
                image = render_view(prediction, size, scale, (0., 0.), mode=mode,
                                    alpha=alpha, source=source, show_unknown=unknown == '1')
            else:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError('预览编码失败')
            body = encoded.tobytes()
        with self.preview_lock:
            self.previews[key] = body
            while len(self.previews) > 160:
                self.previews.popitem(last=False)
        return body

    def batch_spec(self, image: Path) -> dict:
        spec = deepcopy(self.spec)
        spec['target'] = str(image)
        for template in spec['templates']:
            for field in PATH_FIELDS:
                if template.get(field):
                    template[field] = str((self.config.parent / template[field]).resolve())
        return spec
