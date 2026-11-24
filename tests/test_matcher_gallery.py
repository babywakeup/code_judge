"""Offline contracts for the multi-image matcher gallery."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from apps.matcher_viewer.data import Catalog, artifact_path
from apps.matcher_viewer.jobs import Jobs
from apps.matcher_viewer.server import make_server
from periodic_template_matcher.utils import read_image, write_image


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / 'imgs'
        self.images.mkdir()
        self.config = self.root / 'config.json'
        template = self.root / 'template.png'
        self.image = np.full((40, 32, 3), 90, np.uint8)
        write_image(template, self.image)
        self.spec = dict(target='ignored.png', templates=[dict(name='T1', image='template.png')],
                         target_valid_height=40, options={})
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')
        for group, station, severity, date in [('G1', 'AOI1', 'M', '20260119'),
                                               ('G1', 'AOI2', 'S', '20260220'),
                                               ('G2', 'AOI1', 'L', '20260321')]:
            path = self.images / group / f'18851_AFE_{station}_42_-625.130_66.034_O_{severity}_{date}_124238.png'
            path.parent.mkdir(exist_ok=True)
            write_image(path, self.image)
        self.catalog = Catalog(self.images, self.config, self.root / 'results')

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def runner(config, output):
        spec = json.loads(config.read_text(encoding='utf-8'))
        assert Path(spec['templates'][0]['image']).is_absolute()
        assert spec['target_valid_height'] == 40
        write_image(output / 'input.png', np.full((40, 32, 3), 90, np.uint8))
        write_image(output / 'labels.png', np.ones((40, 32), np.uint16))
        write_image(output / 'valid.png', np.full((40, 32), 255, np.uint8))
        metadata = dict(format_version=1, target=spec['target'], accepted=True,
                        prediction_status='available', ranking=[0],
                        matches=[dict(match_score=82.5, template_name='T1')],
                        class_ids=[1], class_pixels={'1': 1280}, label_coverage_full_image=1.,
                        artifacts=dict(input='input.png', prediction_mask='labels.png', prediction_valid='valid.png'))
        (output / 'result.json').write_text(json.dumps(metadata), encoding='utf-8')
        return metadata

    def run_jobs(self, **kwargs):
        jobs = Jobs(self.catalog, runner=self.runner)
        jobs.start(**kwargs)
        if jobs.thread:
            jobs.thread.join(10)
            self.assertFalse(jobs.thread.is_alive())
        return jobs

    def test_catalog_filters_pagination_and_stable_ids(self):
        self.assertEqual(self.catalog.summary()['total'], 3)
        rows = self.catalog.query(group='G1', limit=1)
        self.assertEqual(rows['total'], 2)
        self.assertEqual(len(rows['items']), 1)
        self.assertEqual(self.catalog.query(station='AOI1', severity='L')['total'], 1)
        self.assertEqual(self.catalog.query(date_from='2026-02-01', date_to='2026-02-28')['total'], 1)
        self.assertEqual(self.catalog.query(q='-625.130')['total'], 3)
        self.assertEqual(self.catalog.query(offset=200)['items'], [])
        ids = list(self.catalog.rows)
        self.catalog.refresh()
        self.assertEqual(ids, list(self.catalog.rows))
        with self.assertRaises(ValueError):
            self.catalog.query(min_score='nan')
        with self.assertRaises(ValueError):
            self.catalog.query(sort='unknown')

    def test_batch_cache_skip_force_and_original_config_preserved(self):
        before = self.config.read_bytes()
        jobs = self.run_jobs(filters={'group': 'G1'})
        self.assertEqual(jobs.snapshot()['done'], 2)
        self.assertEqual(jobs.snapshot()['errors'], 0)
        self.assertEqual(self.catalog.query(status='available', template='T1', min_score='80')['total'], 2)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(len(list(self.catalog.results.rglob('receipt.json'))), 2)
        resumed = Catalog(self.images, self.config, self.catalog.results)
        self.assertEqual(resumed.summary()['counts']['available'], 2)
        self.assertEqual(Jobs(resumed).start(filters={'group': 'G1'})['total'], 0)
        self.run_jobs(filters={'group': 'G1'}, skip_completed=False)
        self.assertEqual(len(list(self.catalog.results.rglob('receipt.json'))), 4)
        self.assertEqual(self.catalog.query(sort='score_desc')['items'][-1]['status'], 'pending')

    def test_real_batch_infers_each_target_with_shared_templates(self):
        rows = list(self.catalog.rows.values())[:2]
        write_image(rows[1]['_image'], self.image + 20)
        jobs = Jobs(self.catalog)
        jobs.start(ids=[row['id'] for row in rows])
        jobs.thread.join(10)
        self.assertFalse(jobs.thread.is_alive())
        self.assertEqual(jobs.snapshot()['errors'], 0)
        self.assertEqual(jobs.snapshot()['done'], 2)
        for row, expected in zip(rows, (self.image, self.image + 20)):
            self.assertEqual(row['status'], 'rejected')
            self.assertEqual(row['_metadata']['target'], str(row['_image']))
            np.testing.assert_array_equal(
                read_image(artifact_path(row['_manifest'], row['_metadata'], 'input')), expected)

    def test_rectangular_roi_reaches_matcher_and_keeps_full_size_preview(self):
        self.spec.pop('target_valid_height')
        step1 = dict(valid_y_start=2, valid_y_end=38)
        self.spec['options']['step1'] = step1
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')
        catalog = Catalog(self.images, self.config, self.catalog.results)
        identifier = next(iter(catalog.rows))
        jobs = Jobs(catalog)
        jobs.start(ids=[identifier])
        jobs.thread.join(10)
        self.assertFalse(jobs.thread.is_alive())
        self.assertEqual(jobs.snapshot()['errors'], 0)

        detail = catalog.detail(identifier)
        self.assertEqual(detail['valid_bounds'], dict(x_start=0, x_end=32, y_start=2, y_end=38))
        self.assertEqual(detail['metadata']['matches'][0]['working_target_size'], [32, 36])
        self.assertEqual(detail['timing_scope'], 'excluding_image_save')
        snapshot = Path(detail['result_path']).parent/'inference_config.json'
        self.assertEqual(json.loads(snapshot.read_text(encoding='utf-8'))['options']['step1'], step1)
        preview = cv2.imdecode(np.frombuffer(catalog.preview(identifier, mode='input'), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(preview.shape, self.image.shape)

    def test_skip_completed_reruns_an_imported_result_with_current_config(self):
        folder = self.root/'legacy'
        folder.mkdir()
        row = next(iter(self.catalog.rows.values()))
        config = folder/'config.json'
        config.write_text(json.dumps(self.catalog.batch_spec(row['_image'])), encoding='utf-8')
        self.runner(config, folder)
        imported = Catalog(self.images, self.config, self.catalog.results, [folder])
        self.assertEqual(imported.rows[row['id']]['provenance'], 'imported')

        jobs = Jobs(imported, runner=self.runner)
        self.assertEqual(jobs.start(ids=[row['id']])['total'], 1)
        jobs.thread.join(10)
        self.assertFalse(jobs.thread.is_alive())
        self.assertEqual(jobs.snapshot()['errors'], 0)
        self.assertEqual(imported.rows[row['id']]['provenance'], 'cache')
        self.assertTrue((folder/'result.json').is_file())

    def test_cache_invalidation_by_config_template_and_source(self):
        self.run_jobs(filters={})
        self.spec['target_valid_height'] = 39
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')
        changed = Catalog(self.images, self.config, self.catalog.results)
        self.assertEqual(changed.summary()['counts'], {'pending': 3})
        self.config.write_text(json.dumps(dict(self.spec, target_valid_height=40)), encoding='utf-8')
        write_image(self.root / 'template.png', self.image + 1)
        changed = Catalog(self.images, self.config, self.catalog.results)
        self.assertEqual(changed.summary()['counts'], {'pending': 3})
        row = next(iter(self.catalog.rows.values()))
        write_image(row['_image'], self.image + 2)
        self.catalog.refresh()
        self.assertEqual(self.catalog.rows[row['id']]['status'], 'pending')

    def test_preview_jpeg_and_artifact_boundary(self):
        identifier = next(iter(self.catalog.rows))
        self.assertTrue(self.catalog.preview(identifier).startswith(b'\xff\xd8'))
        self.run_jobs(ids=[identifier])
        for mode in ('input', 'overlay', 'mask', 'split'):
            encoded = self.catalog.preview(identifier, mode=mode, width='96')
            decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(decoded.shape[0], 0)
        row = self.catalog.rows[identifier]
        self.assertEqual(self.catalog.detail(identifier)['metadata']['class_ids'], [1])
        with self.assertRaises(ValueError):
            artifact_path(row['_manifest'], {'artifacts': {'input': '../../config.json'}}, 'input')
        with self.assertRaises(ValueError):
            self.catalog.preview(identifier, alpha='nan')

    def test_failures_isolated_stop_and_single_queue(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(config, output):
            entered.set()
            release.wait(5)
            return self.runner(config, output)
        jobs = Jobs(self.catalog, runner=blocked)
        jobs.start(filters={})
        self.assertTrue(entered.wait(5))
        with self.assertRaises(RuntimeError):
            jobs.start(filters={})
        jobs.stop()
        release.set()
        jobs.thread.join(10)
        self.assertFalse(jobs.thread.is_alive())
        self.assertEqual(jobs.snapshot()['done'], 1)
        self.assertTrue(jobs.snapshot()['cancelled'])
        def failing(config, output):
            raise ValueError('测试坏图')
        jobs = Jobs(self.catalog, runner=failing)
        jobs.start(filters={}, skip_completed=False)
        jobs.thread.join(10)
        self.assertEqual(jobs.snapshot()['errors'], 3)
        self.assertEqual(self.catalog.summary()['counts'], {'error': 3})
        self.assertTrue(all(not r['result'] for r in self.catalog.rows.values()))
        resumed = Catalog(self.images, self.config, self.catalog.results)
        self.assertEqual(resumed.summary()['counts'], {'error': 3})

    def test_rejected_and_unannotated_results_never_imply_masks(self):
        def runner(config, output):
            metadata = self.runner(config, output)
            metadata.update(accepted=False, prediction_status='match_rejected', artifacts={'input': 'input.png'})
            (output / 'result.json').write_text(json.dumps(metadata), encoding='utf-8')
        jobs = Jobs(self.catalog, runner=runner)
        jobs.start(filters={})
        jobs.thread.join(10)
        self.assertEqual(self.catalog.summary()['counts'], {'rejected': 3})
        identifier = next(iter(self.catalog.rows))
        self.assertEqual(self.catalog.detail(identifier)['artifacts'], {'input': 'input.png'})
        self.assertTrue(self.catalog.preview(identifier, mode='overlay').startswith(b'\xff\xd8'))

    def test_external_results_require_matching_source_pixels(self):
        folder = self.root / 'legacy'
        folder.mkdir()
        config = folder / 'run.json'
        first = next(iter(self.catalog.rows.values()))
        config.write_text(json.dumps(self.catalog.batch_spec(first['_image'])), encoding='utf-8')
        self.runner(config, folder)
        imported = Catalog(self.images, self.config, self.catalog.results, [folder])
        self.assertEqual(imported.rows[first['id']]['provenance'], 'imported')
        write_image(first['_image'], self.image + 1)
        imported = Catalog(self.images, self.config, self.catalog.results, [folder])
        self.assertEqual(imported.summary()['counts'], {'pending': 3})
        self.assertTrue(imported.warnings)

    def test_safe_output_and_target_mask(self):
        with self.assertRaises(ValueError):
            Catalog(self.images, self.config, self.images / 'results')
        self.spec['target_valid_mask'] = 'anything.png'
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')
        with self.assertRaises(ValueError):
            Catalog(self.images, self.config, self.root / 'results')

    def test_http_endpoints_and_csrf(self):
        server = make_server(self.catalog, Jobs(self.catalog, runner=self.runner), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/api/summary', timeout=5) as response:
                summary = json.load(response)
            identifier = next(iter(self.catalog.rows))
            with urlopen(base + '/api/items?group=G1&limit=1', timeout=5) as response:
                self.assertEqual(json.load(response)['total'], 2)
            with urlopen(base + '/preview/' + identifier, timeout=5) as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
            with urlopen(base + '/', timeout=5) as response:
                self.assertIn('AOI'.encode(), response.read())
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(base + '/api/jobs', data=b'{}', method='POST'), timeout=5)
            self.assertEqual(denied.exception.code, 403)
            headers = {'X-Viewer-Token': summary['token'], 'Content-Type': 'application/json'}
            with self.assertRaises(HTTPError) as denied:
                urlopen(Request(base + '/api/jobs', data=b'{}', headers=dict(headers, Origin='http://evil.example'), method='POST'), timeout=5)
            self.assertEqual(denied.exception.code, 403)
            body = json.dumps({'ids': [identifier]}).encode()
            with urlopen(Request(base + '/api/jobs', data=body, headers=headers, method='POST'), timeout=5) as response:
                self.assertEqual(json.load(response)['total'], 1)
            server.jobs.thread.join(10)
            with urlopen(base + f'/artifact/{identifier}/prediction_mask', timeout=5) as response:
                self.assertTrue(response.read().startswith(b'\x89PNG'))
            for path in ('/api/item/not-an-id', f'/artifact/{identifier}/unknown', '/../../config.json'):
                with self.assertRaises(HTTPError):
                    urlopen(base + path, timeout=5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


    def test_default_runner_excludes_image_writes_from_elapsed(self):
        row = next(iter(self.catalog.rows.values()))
        clock = 0.

        def slow_write(path, image):
            nonlocal clock
            clock += 5.
            return write_image(path, image)

        from periodic_template_matcher._internal.io.export import _write_json

        def timed_write_json(out, document):
            nonlocal clock
            clock += 2.
            return _write_json(out, document)

        jobs = Jobs(self.catalog)
        with patch('time.perf_counter', side_effect=lambda: clock), \
             patch('periodic_template_matcher._internal.io.export._write_json', side_effect=timed_write_json), \
             patch('periodic_template_matcher._internal.io.export.write_image', side_effect=slow_write):
            jobs.start(ids=[row['id']])
            jobs.thread.join(10)
        self.assertFalse(jobs.thread.is_alive())
        self.assertEqual(jobs.snapshot()['errors'], 0)
        self.assertEqual(clock, 12.)
        self.assertEqual(row['elapsed'], 2.)
        receipt = json.loads((row['_manifest'].parent/'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['elapsed'], 2.)
        resumed = Catalog(self.images, self.config, self.catalog.results)
        self.assertEqual(resumed.rows[row['id']]['timing_scope'], 'excluding_image_save')


if __name__ == '__main__':
    unittest.main()

