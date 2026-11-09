"""Native boundary tests: missing library, wrong indexing and ownership are failures."""
import importlib.util
import unittest
import math


class NativeAvailabilityTests(unittest.TestCase):
    def test_compiled_core_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec('periodic_template_matcher_cpp._native'),
                             'C++ extension must be built; Python fallback is not acceptable')

    def test_rounding_and_percentile_match_python_semantics(self):
        from periodic_template_matcher_cpp import _native
        for value in (-2.5,-1.5,-.5,.5,1.5,2.5):
            self.assertEqual(_native.pyround(value),round(value))
        self.assertEqual(_native.percentile([0.,10.,30.,50.],25),7.5)

    def test_nonfinite_rounding_raises_instead_of_native_trap(self):
        # A subprocess protects the test runner from a native conversion trap.
        import subprocess
        import sys
        result=subprocess.run([sys.executable,'-c',
            'from periodic_template_matcher_cpp import _native\n'
            'try: _native.pyround(float("nan"))\n'
            'except ValueError: pass\n'
            'else: raise AssertionError("Nonfinite round accepted")'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
