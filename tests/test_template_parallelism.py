"""Parallel registration must preserve serial matching and template ordering."""
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.registration import core
from tests.helpers import texture


class ParallelRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.image = texture()
        self.cfg = MatcherConfig(max_side=160, angles=(0.,), scales=(1.,),
                                 refine_candidates=2, ecc_iterations=15, template_workers=1)

    def test_worker_options_and_validation(self):
        self.assertEqual(MatcherConfig().template_workers, 0)
        for value in (0, 1, 2, 4):
            cfg = MatcherConfig.from_options({'step2': {'template_workers': value}})
            self.assertEqual(cfg.template_workers, value)
        for value in (-1, True, 1.5, '4', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'template_workers'):
                MatcherConfig(template_workers=value).validate()

    def test_worker_bounds(self):
        with patch.object(core.os, 'cpu_count', return_value=8):
            self.assertEqual(core._template_worker_count(4, 0), 4)
            self.assertEqual(core._template_worker_count(12, 0), 4)
            self.assertEqual(core._template_worker_count(2, 4), 2)
            self.assertEqual(core._template_worker_count(12, 16), 8)
        with patch.object(core.os, 'cpu_count', return_value=None):
            self.assertEqual(core._template_worker_count(4, 0), 1)

    def test_real_serial_and_parallel_results_are_identical(self):
        other = texture(19)
        templates = [self.image, other, self.image.copy(), other.copy()]
        # Use a translated target: exact self-correlation has symmetric tied
        # secondary peaks whose rounding can vary even across serial runs.
        target = cv2.warpAffine(self.image, np.float32([[1, 0, 5], [0, 1, -4]]), (200, 160))
        before = [image.copy() for image in templates]
        for backend in ('fft', 'opencv'):
            cfg = replace(self.cfg, backend=backend)
            serial = core.select_templates(templates, target, cfg)
            parallel = core.select_templates(templates, target, replace(cfg, template_workers=4))
            serial.pop('config'); parallel.pop('config')
            self.assertEqual(serial, parallel)
            self.assertLess(parallel['ranking'].index(0), parallel['ranking'].index(2))
            for image, original in zip(templates, before):
                np.testing.assert_array_equal(image, original)

    def test_target_prepared_once_and_tasks_overlap(self):
        barrier = threading.Barrier(4, timeout=5)
        original = core._register_prepared
        threads = set()
        lock = threading.Lock()

        def register(*args):
            with lock:
                threads.add(threading.get_ident())
            barrier.wait()
            return original(*args)

        with patch.object(core.os, 'cpu_count', return_value=8), \
             patch.object(core, '_register_prepared', side_effect=register), \
             patch.object(core, '_prepare_image', wraps=core._prepare_image) as prepare:
            core.select_templates([self.image] * 4, self.image, replace(self.cfg, template_workers=4))
        self.assertEqual(len(threads), 4)
        self.assertEqual(prepare.call_count, 5)

    def test_single_template_and_explicit_serial_do_not_start_executor(self):
        with patch.object(core, 'ThreadPoolExecutor', side_effect=AssertionError('unexpected pool')):
            core.select_templates([self.image], self.image, replace(self.cfg, template_workers=4))
            core.select_templates([self.image] * 2, self.image, self.cfg)

    def test_worker_exception_is_not_swallowed(self):
        with patch.object(core, '_register_prepared', side_effect=RuntimeError('worker failed')):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                core.select_templates([self.image] * 2, self.image, replace(self.cfg, template_workers=2))


if __name__ == '__main__':
    unittest.main()
