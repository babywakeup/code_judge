"""Serial background inference with resumable, immutable cache entries."""
from __future__ import annotations

import json
import threading
import time
import uuid

from .data import Catalog


class Jobs:
    def __init__(self, catalog: Catalog, runner=None):
        self.catalog = catalog
        self.runner = runner
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = dict(active=False, total=0, done=0, errors=0, current=None, cancelled=False)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self, ids=None, filters=None, skip_completed=True):
        if type(skip_completed) is not bool:
            raise ValueError('skip_completed 必须为布尔值')
        if ids is not None and filters is not None:
            raise ValueError('ids 与 filters 不能同时指定')
        with self.lock, self.catalog.lock:
            if self.state['active']:
                raise RuntimeError('已有推理任务正在运行')
            if ids is None:
                if not isinstance(filters, dict):
                    raise ValueError('请提供 ids 或 filters')
                selected = self.catalog.select(filters)
            else:
                if not isinstance(ids, list) or any(not isinstance(i, str) or i not in self.catalog.rows for i in ids):
                    raise ValueError('无效图片 ID')
                selected = [self.catalog.rows[i] for i in dict.fromkeys(ids)]
            selected = [r['id'] for r in selected
                        if not (skip_completed and r['result'] and r['provenance'] == 'cache')]
            self.stop_event.clear()
            self.state = dict(active=bool(selected), total=len(selected), done=0, errors=0,
                              current=None, cancelled=False)
            if selected:
                self.thread = threading.Thread(target=self._run, args=(selected,), daemon=True)
                self.thread.start()
            return dict(self.state)

    def stop(self):
        with self.lock:
            if self.state['active']:
                self.stop_event.set()
                self.state['cancelled'] = True
            return dict(self.state)

    def _run(self, identifiers):
        try:
            matcher = None
            if self.runner is None:
                from periodic_template_matcher import PeriodicTemplateMatcher
            for identifier in identifiers:
                if self.stop_event.is_set():
                    break
                started = time.perf_counter()
                with self.catalog.lock:
                    row = self.catalog.rows.get(identifier)
                    if row is None:
                        continue
                    image = row['_image']
                    fingerprint = row['_fingerprint']
                    row.update(status='running', error=None, result=False, _manifest=None,
                               _metadata=None, score=None, template='', coverage=None)
                with self.lock:
                    self.state['current'] = row['key']
                folder = None
                timings = {}
                timing_scope = 'excluding_image_save' if self.runner is None else 'including_image_save'
                try:
                    fingerprint = self.catalog.fingerprint(image)
                    # UUID avoids touching any previous result, including forced reruns.
                    folder = self.catalog.results / identifier / f'{time.time_ns()}-{uuid.uuid4().hex[:8]}'
                    folder.mkdir(parents=True)
                    config = folder / 'inference_config.json'
                    config.write_text(json.dumps(self.catalog.batch_spec(image), ensure_ascii=False, indent=2), encoding='utf-8')
                    if self.runner is not None:
                        self.runner(config, folder)
                    else:
                        if matcher is None:
                            matcher = PeriodicTemplateMatcher(config)
                        matcher.infer_to(folder, target=image, timings=timings)
                    if self.catalog.fingerprint(image) != fingerprint:
                        raise ValueError('推理过程中源图发生改变；结果不作为当前缓存')
                    elapsed = time.perf_counter() - started - timings.get('image_save_seconds', 0.)
                    # Publish the receipt last. Partial/failed runs are never resumed as complete.
                    receipt = dict(id=identifier, fingerprint=fingerprint, elapsed=elapsed, timing_scope=timing_scope)
                    with self.catalog.lock:
                        self.catalog.attach(row, folder / 'result.json', elapsed, timing_scope=timing_scope)
                        row['_fingerprint'] = fingerprint
                        (folder / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
                except Exception as exc:
                    with self.catalog.lock:
                        # A failure never serves an older mask as the new prediction.
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}', result=False,
                                   _manifest=None, _metadata=None, score=None, template='', coverage=None,
                                   elapsed=time.perf_counter() - started - timings.get('image_save_seconds', 0.),
                                   timing_scope=timing_scope)
                    if folder is not None:
                        try:
                            (folder / 'error.json').write_text(json.dumps(
                                dict(id=identifier, fingerprint=fingerprint, error=row['error'],
                                     elapsed=row['elapsed'], timing_scope=timing_scope),
                                ensure_ascii=False), encoding='utf-8')
                        except OSError:
                            pass
                    with self.lock:
                        self.state['errors'] += 1
                finally:
                    with self.lock:
                        self.state['done'] += 1
        except Exception as exc:
            with self.lock:
                self.state['error'] = f'{type(exc).__name__}: {exc}'
                self.state['errors'] += 1
        finally:
            with self.lock:
                self.state.update(active=False, current=None)
