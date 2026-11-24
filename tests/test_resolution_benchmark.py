"""Resolution comparisons must isolate the knob and preserve input files."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from periodic_template_matcher.utils import write_image
from tests.helpers import texture
from tools.benchmark_resolution import benchmark


class ResolutionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        image = texture()
        write_image(self.root/'image.png', image)
        labels = np.zeros(image.shape[:2], np.uint16)
        labels[30:100, 40:120] = 1
        write_image(self.root/'labels.png', labels)
        self.spec = dict(target='image.png', templates=[dict(image='image.png', labels='labels.png')],
                         options=dict(step1=dict(max_side=120, scale=None),
                                      step2=dict(angles=[0], scales=[1.], peaks_per_hypothesis=1),
                                      step3=dict(ecc_iterations=10)))
        self.config = self.root/'config.json'
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')

    def test_isolates_resolution_uses_abba_and_keeps_source_config(self):
        original = self.config.read_bytes()
        output = self.root/'comparison'
        with redirect_stdout(StringIO()):
            report = benchmark(self.config, output, [], reference=120, candidate=96, repeats=2)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual([row['max_side'] for row in report['rows']], [120, 96, 96, 120])
        a = json.loads((output/'config.120.json').read_text(encoding='utf-8'))
        b = json.loads((output/'config.96.json').read_text(encoding='utf-8'))
        a['options']['step1']['max_side'] = b['options']['step1']['max_side']
        self.assertEqual(a, b)
        self.assertEqual(a['options']['step2']['template_workers'], 1)
        self.assertFalse(report['ground_truth_used'])
        drift = report['rows'][1]['difference_from_reference']
        self.assertIn('lost_known_pixels', drift)
        self.assertIn('class_iou_on_shared_known', drift)
        self.assertTrue((output/'target1.96/result.json').is_file())
        with self.assertRaises(FileExistsError):
            benchmark(self.config, output, [], reference=120, candidate=96)

    def test_explicit_scale_is_rejected_instead_of_ignoring_max_side(self):
        self.spec['options']['step1']['scale'] = .5
        self.config.write_text(json.dumps(self.spec), encoding='utf-8')
        output = self.root/'invalid'
        with self.assertRaisesRegex(ValueError, 'overrides max_side'):
            benchmark(self.config, output, [])
        self.assertFalse(output.exists())

    def test_invalid_parameters_do_not_create_output(self):
        output = self.root/'invalid'
        for kwargs in (dict(repeats=0), dict(candidate=20), dict(candidate=800)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                benchmark(self.config, output, [], **kwargs)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()