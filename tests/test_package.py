"""Public imports, dependency isolation and supported CLI entry points."""
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import periodic_template_matcher as package
from periodic_template_matcher import __main__ as cli
from periodic_template_matcher import config, matcher
from periodic_template_matcher._internal import results


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / 'periodic_template_matcher'


class PackageTests(unittest.TestCase):
    def run_python(self, *args, cwd=ROOT):
        return subprocess.run([sys.executable, *args], cwd=cwd, check=True,
                              capture_output=True, text=True, timeout=30)

    def test_public_types_are_single_definitions(self):
        self.assertIs(package.MatcherConfig, config.MatcherConfig)
        self.assertIs(package.PeriodicTemplateMatcher, matcher.PeriodicTemplateMatcher)
        self.assertIs(package.MatchResult, results.MatchResult)
        self.assertIs(package.EvaluationResult, results.EvaluationResult)
        self.assertEqual(set(package.__all__), {'MatcherConfig', 'PeriodicTemplateMatcher',
                            'MatchResult', 'EvaluationResult', 'run_config'})

    def test_relative_imports_ignore_unrelated_top_level_modules(self):
        code = '''
import sys
from types import ModuleType
names = ('annotations', 'workflow', 'visual_report', 'segmentation', 'config',
         'utils', 'results', 'matcher', 'registration', 'periodic_matcher')
for name in names:
    sys.modules[name] = ModuleType(name)
from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher import run_config
assert PeriodicTemplateMatcher(MatcherConfig()).config.label_mode == 'semantic'
assert callable(run_config)
'''
        self.run_python('-c', code)

    def test_package_cli_help(self):
        for module in ('periodic_template_matcher',):
            with self.subTest(module=module):
                completed = self.run_python('-m', module, '--help')
                self.assertIn('--config', completed.stdout)
                self.assertIn('--out', completed.stdout)
                self.assertNotIn('RuntimeWarning', completed.stderr)

    def test_package_root_is_minimal_and_legacy_modules_are_removed(self):
        self.assertEqual({p.name for p in PACKAGE_DIR.glob('*.py')},
                         {'__init__.py', '__main__.py', 'config.py', 'matcher.py'})
        for name in ('periodic_matcher', 'segmentation', 'cli', 'test_matcher', 'test_matcher_api'):
            self.assertIsNone(importlib.util.find_spec(f'periodic_template_matcher.{name}'))

    def test_cli_argument_routing(self):
        result = dict(best_template_index=0, accepted=True, status='accepted',
                      template_score_gap=None, matches=[dict(template_name='A',
                      match_score=90., geometry_accepted=True, periodic_detected=False)])
        output = StringIO()
        with patch('periodic_template_matcher.__main__.run_config', return_value=result) as run:
            with redirect_stdout(output):
                cli.main(['--config', 'sample.json', '--out', 'output'])
        run.assert_called_once_with('sample.json', 'output')
        self.assertIn('score=90.00', output.getvalue())


if __name__ == '__main__':
    unittest.main()