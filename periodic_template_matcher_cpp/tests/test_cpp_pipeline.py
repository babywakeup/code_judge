import unittest
import numpy as np
import cv2
from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher as PythonMatcher
from periodic_template_matcher_cpp import PeriodicTemplateMatcher as CppMatcher
from tests.helpers import texture


class PipelineParity(unittest.TestCase):
    def test_missing_ground_truth_is_rejected_before_matching(self):
        image=texture()
        with self.assertRaises(ValueError):
            CppMatcher().evaluate([image],[None],image,None)

    def test_translation_labels_and_evaluation(self):
        source=texture()
        H=np.array([[1.,0,6],[0,1,-4],[0,0,1]])
        target=cv2.warpPerspective(source,H,(200,160))
        labels=np.zeros(source.shape[:2],np.uint16);labels[35:70,40:90]=1001
        truth=cv2.warpPerspective(labels,H,(200,160),flags=cv2.INTER_NEAREST)
        for backend in ('fft','opencv'):
            cfg=MatcherConfig(max_side=200,angles=(0.,),scales=(1.,),refine_candidates=3,ecc_iterations=35,backend=backend)
            a=CppMatcher(cfg).evaluate([source],[labels],target,truth)
            b=PythonMatcher(cfg).evaluate([source],[labels],target,truth)
            self.assertEqual(a.match.accepted,b.match.accepted)
            self.assertEqual(a.match.registration['status'],b.match.registration['status'])
            np.testing.assert_allclose(a.match.matrix_template_to_target,b.match.matrix_template_to_target,atol=.1)
            np.testing.assert_array_equal(a.match.mapped_labels,b.match.mapped_labels)
            np.testing.assert_array_equal(a.match.label_valid,b.match.label_valid)
            self.assertAlmostEqual(a.score,b.score,places=6)

    def test_noncontinuous_and_result_lifetime(self):
        original=texture();source=original[:,::-1]
        labels=np.zeros(source.shape[:2],np.uint16);labels[30:60,40:65]=2
        cfg=MatcherConfig(max_side=200,angles=(0.,),scales=(1.,))
        result=CppMatcher(cfg).match([source],[labels],source.copy())
        del source
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.mapped_labels,labels)
        np.testing.assert_array_equal(result.input_image,original[:,::-1])


if __name__=='__main__':unittest.main()
