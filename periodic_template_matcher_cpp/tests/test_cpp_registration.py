import unittest
import numpy as np
from dataclasses import asdict
from periodic_template_matcher.config import MatcherConfig
from periodic_template_matcher._internal.registration.core import _features, select_templates
from periodic_template_matcher._internal.registration.correlation import _corr, _translation_peaks, _periods
from periodic_template_matcher._internal.registration.scoring import _score, _refine
import periodic_template_matcher_cpp._native as native

class RegistrationParity(unittest.TestCase):
    def test_same_matrix_image_warps_are_exact(self):
        import cv2
        image=np.random.default_rng(4).integers(0,256,(96,112,3),dtype=np.uint8)
        for tx,ty in ((0.,0.),(4.,-3.),(.0713,2.503)):
            H=np.array([[1.,.003,tx],[-.002,1.,ty],[0.,0.,1.]])
            actual=native.warp(image,H,(96,112),cv2.INTER_LINEAR)
            np.testing.assert_array_equal(actual,cv2.warpPerspective(image,H,(112,96)))

    def setUp(self):
        self.cfg=MatcherConfig(angles=(0,),scales=(1,),refine_candidates=2,ecc_iterations=10)
        self.image=np.random.default_rng(5).integers(0,256,(96,112),dtype=np.uint8)
        self.mask=np.ones(self.image.shape,np.uint8)
    def test_features_and_score(self):
        self.mask[15:25,20:35]=0
        oracle=_features(self.image,self.mask)
        actual=native.features(self.image,self.mask)
        for a,b in zip(actual,oracle): np.testing.assert_allclose(a,b,atol=2e-5)
        expected=_score(oracle,oracle,np.eye(3),self.cfg)
        result=native.score(actual,actual,np.eye(3),asdict(self.cfg))
        for k,v in expected.items():
            if v is None: self.assertIsNone(result[k])
            else: self.assertAlmostEqual(result[k],v,places=4)
    def test_gradient_normalization_uses_float32_division(self):
        # A horizontal ramp has no vertical gradient, isolating division from
        # cross-build magnitude/sqrt rounding. NumPy casts the percentile cap
        # back to float32 before dividing every float32 gradient sample.
        image=np.tile(np.arange(128,dtype=np.uint8),(64,1))
        mask=np.ones(image.shape,np.uint8)
        actual=native.features(image,mask)[1]
        expected=_features(image,mask)[1]
        np.testing.assert_array_equal(actual,expected)
    def test_shifted_score_same_features(self):
        # Nonzero distances/NCC expose precision differences hidden by identity.
        # Use the same Python features on both sides to isolate scoring from
        # platform-dependent OpenCV filtering and distance-transform kernels.
        self.mask[15:25,20:35]=0
        f=_features(self.image,self.mask)
        h=np.eye(3); h[0,2]=2.25; h[1,2]=-1.5
        expected=_score(f,f,h,self.cfg)
        actual=native.score(f,f,h,asdict(self.cfg))
        for key,value in expected.items():
            if value is None: self.assertIsNone(actual[key])
            else: self.assertAlmostEqual(actual[key],value,places=4)
    def test_signed_translation(self):
        feature=_features(self.image,self.mask)[1]
        target=np.roll(feature,(-13,17),(0,1))
        for backend in ('fft','opencv'):
            self.cfg.backend=backend
            if backend=='fft': oracle=_translation_peaks
            else:
                from periodic_template_matcher._internal.registration.opencv import translation_peaks as oracle
            np.testing.assert_allclose(native.translation_peaks(feature,target,self.mask,self.mask,asdict(self.cfg)),oracle(feature,target,self.mask,self.mask,self.cfg),atol=2e-5)
    def test_stable_selection(self):
        expected=select_templates([self.image,self.image],self.image,self.cfg)
        actual=native.select_templates([self.image,self.image],self.image,asdict(self.cfg),[None,None],None,'semantic')
        self.assertEqual(actual['ranking'],expected['ranking'])
        self.assertEqual(actual['status'],expected['status'])
        for a,b in zip(actual['matches'],expected['matches']): self.assertAlmostEqual(a['match_score'],b['match_score'],places=3)



class RegistrationDiagnosticsParity(unittest.TestCase):
    def test_correlation_float32_exact_forward_axis_rounding(self):
        # NumPy 2.5's integer forward normalization selects double FFT loops,
        # with complex64 rounding after each axis (not just after rfft2).
        # Autocorrelation alone cannot exercise general complex products;
        # independent signed inputs also protect the verified FMA direction.
        rng=np.random.default_rng(170)
        feature=rng.normal(size=(37,43)).astype(np.float32)
        mask=rng.integers(0,2,size=feature.shape,dtype=np.uint8)
        centered=(feature-feature[mask>0].mean())*mask
        other=rng.normal(size=feature.shape).astype(np.float32)
        for shape in ((75,90),(81,91),(32,40),(256,320)):
            for name,a,b in (('auto',centered,centered),
                             ('cross',centered,other),
                             ('reverse',other,centered)):
                with self.subTest(shape=shape,pair=name):
                    expected=_corr(a,b,shape)
                    actual=native.correlate(a,b,shape)
                    self.assertEqual(actual.dtype,np.dtype(np.float32))
                    np.testing.assert_array_equal(actual,expected)

    def test_correlation_dtype_and_normalization(self):
        # The coordinator exposes correlate(a, b, fft_shape=(height, width)).
        # Compare all four _periods correlations independently: a mixed FFT
        # must NOT convert the float32 input to double before transforming it.
        rng=np.random.default_rng(170)
        feature=rng.normal(size=(37,43)).astype(np.float32)
        mask=rng.integers(0,2,size=feature.shape,dtype=np.uint8)
        centered=(feature-feature[mask>0].mean())*mask
        squared=centered*centered
        pairs=((centered,centered),(squared,mask),(mask,squared),(mask,mask),
               (centered.astype(np.float64),centered.astype(np.float64)),
               (mask.astype(np.int32),mask.astype(np.int32)))
        for shape in ((75,90),(81,91),(32,40)):
            for a,b in pairs:
                with self.subTest(shape=shape,a_dtype=a.dtype,b_dtype=b.dtype):
                    expected=_corr(a,b,shape)
                    actual=native.correlate(a,b,shape)
                    self.assertEqual(actual.shape,shape)
                    self.assertEqual(actual.dtype,expected.dtype)
                    # Float32 FFT roundoff is magnitude-dependent. Integer and
                    # double-only pairs must retain double precision.
                    tolerance=2e-5 if a.dtype==np.float32 or b.dtype==np.float32 else 2e-12
                    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=2e-7)

    def test_translation_fft_dtype_combinations(self):
        cfg=MatcherConfig(angles=(0,),scales=(1,),backend='fft')
        rng=np.random.default_rng(72)
        base=rng.uniform(.05,1,size=(73,91))
        for dtype in (np.float32,np.float64):
            for mask_dtype in (np.uint8,np.float32,np.float64):
                with self.subTest(dtype=dtype,mask_dtype=mask_dtype):
                    source=base.astype(dtype)
                    target=np.roll(source,(11,-17),(0,1))
                    sm=np.ones(source.shape,dtype=mask_dtype)
                    tm=sm.copy();tm[:6,:9]=0
                    expected=_translation_peaks(source,target,sm,tm,cfg)
                    actual=native.translation_peaks(source,target,sm,tm,asdict(cfg))
                    # Never relax displacement parity when scores are close.
                    self.assertEqual([p[1:] for p in actual],[p[1:] for p in expected])
                    np.testing.assert_allclose([p[0] for p in actual],
                                               [p[0] for p in expected],atol=2e-6,rtol=0)

    def test_periodic_lattice(self):
        yy,xx=np.indices((128,160))
        feature=(np.sin(xx*2*np.pi/24)+np.cos(yy*2*np.pi/32)).astype(np.float32)
        mask=np.ones(feature.shape,np.uint8)
        expected=_periods(feature,mask)
        actual=native.periods(feature,mask)
        self.assertEqual([x['vector_work_px'] for x in actual],[x['vector_work_px'] for x in expected])
        np.testing.assert_allclose([x['autocorrelation'] for x in actual],[x['autocorrelation'] for x in expected],atol=2e-6)
    def test_periodic_lattice_mask_fft_dtypes(self):
        yy,xx=np.indices((127,157))
        feature=(np.sin(xx*2*np.pi/23)+np.cos(yy*2*np.pi/31)
                 +.03*np.random.default_rng(9).normal(size=xx.shape)).astype(np.float32)
        for feature_dtype in (np.float32,np.float64):
            for mask_dtype in (np.uint8,np.float32,np.float64):
                with self.subTest(feature_dtype=feature_dtype,mask_dtype=mask_dtype):
                    f=feature.astype(feature_dtype)
                    mask=np.ones(f.shape,dtype=mask_dtype);mask[:7,:13]=0
                    expected=_periods(f,mask)
                    actual=native.periods(f,mask)
                    self.assertEqual([x['vector_work_px'] for x in actual],
                                     [x['vector_work_px'] for x in expected])
                    np.testing.assert_allclose([x['autocorrelation'] for x in actual],
                                               [x['autocorrelation'] for x in expected],
                                               atol=2e-6,rtol=0)
    def test_periodic_lattice_float32_masked_dc_mean(self):
        # A large DC offset distinguishes NumPy's float32 pairwise reduction
        # from cv::mean's double accumulation (even if cast back to float32).
        yy,xx=np.indices((128,160))
        feature=(1e6+np.sin(xx*2*np.pi/24)+np.cos(yy*2*np.pi/32)
                 +.01*np.random.default_rng(41).normal(size=xx.shape)).astype(np.float32)
        mask=np.ones(feature.shape,np.uint8); mask[:7,:13]=0
        self.assertNotEqual(feature[mask>0].mean(),
                            np.float32(feature[mask>0].mean(dtype=np.float64)))
        expected=_periods(feature,mask)
        actual=native.periods(feature,mask)
        self.assertEqual([x['vector_work_px'] for x in actual],
                         [x['vector_work_px'] for x in expected])
        # Same tolerance as the original lattice test, without NumPy's default
        # relative tolerance masking a several-micro-unit centering error.
        np.testing.assert_allclose([x['autocorrelation'] for x in actual],
                                   [x['autocorrelation'] for x in expected],
                                   atol=2e-6,rtol=0)
    def test_refinement_and_insufficient_overlap(self):
        image=np.random.default_rng(22).integers(0,256,(112,128),dtype=np.uint8)
        mask=np.ones(image.shape,np.uint8)
        f=_features(image,mask)
        cfg=MatcherConfig(ecc_iterations=10)
        expected,ok=_refine(f,f,np.eye(3),cfg)
        actual,actual_ok=native.refine(f,f,np.eye(3),asdict(cfg))
        self.assertEqual(actual_ok,ok)
        np.testing.assert_allclose(actual,expected,atol=1e-3)
        h=np.eye(3); h[0,2]=200
        self.assertEqual(native.score(f,f,h,asdict(cfg)),_score(f,f,h,cfg))


if __name__=='__main__': unittest.main()


