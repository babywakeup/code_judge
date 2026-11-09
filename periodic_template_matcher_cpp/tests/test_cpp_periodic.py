import unittest
from dataclasses import asdict
import numpy as np
import periodic_template_matcher_cpp._native as native
from periodic_template_matcher.config import MatcherConfig
from periodic_template_matcher._internal.segmentation.periodic import anchor_vectors, complete_periodic
from periodic_template_matcher._internal.segmentation.evaluation import evaluate_segmentation
from periodic_template_matcher._internal.models import MatchResult

class PeriodicParity(unittest.TestCase):
    def test_anchor_vectors(self):
        a=np.zeros((100,120),np.uint16)
        for y in (5,35,65):
            for x in (5,35,65): a[y:y+5,x:x+5]=3
        v=np.ones(a.shape,np.uint8)
        np.testing.assert_allclose(native.anchor_vectors(a,v,3),anchor_vectors(a,v,3))
    def test_evaluation_abstentions(self):
        p=np.array([[0,1,2],[1,2,65000]],np.uint16); t=np.array([[1,1,0],[2,2,65000]],np.uint16)
        known=np.array([[0,1,1],[1,1,0]],np.uint8); roi=np.ones(p.shape,np.uint8)
        for bg in (False,True):
            self.assertEqual(native.evaluate_segmentation(p,t,known,roi,bg),evaluate_segmentation(p,t,known,roi,include_background=bg))
    def test_empty_roi_and_empty_foreground(self):
        a=np.zeros((4,5),np.uint16)
        for roi in (np.zeros(a.shape,np.uint8),np.ones(a.shape,np.uint8)):
            for bg in (False,True):
                known=np.ones(a.shape,np.uint8)
                self.assertEqual(native.evaluate_segmentation(a,a,known,roi,bg),evaluate_segmentation(a,a,known,roi,include_background=bg))
    def test_no_period(self):
        image=np.zeros((40,50),np.uint8); valid=np.ones(image.shape,bool); known=valid.copy(); known[:,25:]=False
        registration={'matches':[{'matrix_template_to_target':np.eye(3).tolist()}],'ranking':[0]}
        result=MatchResult(registration,image,image,image.copy(),valid.copy(),np.zeros(image.shape,np.uint16),known)
        payload={k:v for k,v in vars(result).items()}
        cfg=MatcherConfig()
        actual=native.complete_periodic(payload,result.mapped_labels,valid,valid,valid,asdict(cfg))
        complete_periodic(result,result.mapped_labels,valid,valid,valid,cfg)
        self.assertEqual(actual['registration'],result.registration)
        np.testing.assert_array_equal(actual['label_valid'],result.label_valid)

    def test_configured_period(self):
        import copy
        tile=np.random.default_rng(17).integers(0,256,(48,32),dtype=np.uint8)
        source=np.tile(tile,(1,2)); target=np.tile(tile,(1,3))
        labels=np.zeros(source.shape,np.uint16); labels[12:30,24:40]=2
        annotation=np.ones(source.shape,bool); target_valid=np.ones(target.shape,bool)
        known=np.zeros(target.shape,bool); known[:,:64]=True
        mapped=np.zeros(target.shape,np.uint16); mapped[:,:64]=labels
        warped=np.zeros(target.shape,np.uint8); warped[:,:64]=source
        registration={'matches':[{'matrix_template_to_target':np.eye(3).tolist()}],'ranking':[0]}
        result=MatchResult(registration,target,source,warped,known.copy(),mapped,known.copy())
        payload=copy.deepcopy(vars(result))
        cfg=MatcherConfig(period_vectors=((32,0),),periodic_max_steps=2,ecc_iterations=8)
        actual=native.complete_periodic(payload,labels,annotation,annotation,target_valid,asdict(cfg))
        complete_periodic(result,labels,annotation,annotation,target_valid,cfg)
        for key in ('mapped_labels','label_valid','completed_valid','periodic_valid','label_periodic_valid'):
            np.testing.assert_array_equal(actual[key],getattr(result,key),err_msg=key)
        info=actual['registration']['periodic_completion']; expected=result.registration['periodic_completion']
        for key in ('status','period_vectors','added_label_pixels','remaining_unknown_pixels','boundary_foreground_pixels'):
            self.assertEqual(info[key],expected[key],key)
    def test_anchor_fallback_and_local_validation(self):
        import copy
        import cv2
        from tests.test_periodic_completion import fixture
        from periodic_template_matcher.utils import warp_labels, warp_valid_mask
        source, labels, target, _, H = fixture()
        valid=np.ones(source.shape,bool)
        for mode in ('anchor','fallback','local_rejection','blocked_class'):
            with self.subTest(mode=mode):
                labs=labels.copy(); image=target.copy()
                opts=dict(max_side=240,ecc_iterations=25,periodic_max_steps=1)
                best={'matrix_template_to_target':H.tolist()}
                if mode=='fallback':
                    tile=np.zeros((80,80),np.uint16); tile[20:42,10:20]=1; tile[20:42,50:60]=1
                    labs=np.tile(tile,(3,3))
                    best.update(working_target_size=[240,240],lattice_candidates=[{'vector_work_px':[80,0]},{'vector_work_px':[0,80]}])
                if mode in ('local_rejection','blocked_class'):
                    opts['period_vectors']=((80.,0.),(0.,80.))
                if mode=='local_rejection':
                    image[:,-12:]=np.random.default_rng(59).integers(0,255,(240,12),dtype=np.uint8)
                if mode=='blocked_class':
                    opts['class_periodicity']={'1':('y',),'1001':()}
                mapped,known=warp_labels(labs,H,image.shape,valid)
                geom=warp_valid_mask(valid,H,image.shape)
                warped=cv2.warpPerspective(source,H,(240,240))
                result=MatchResult({'matches':[best],'ranking':[0]},image,source,warped,geom,mapped,known)
                payload=copy.deepcopy(vars(result)); cfg=MatcherConfig(**opts)
                actual=native.complete_periodic(payload,labs,valid,valid,valid,asdict(cfg))
                complete_periodic(result,labs,valid,valid,valid,cfg)
                for key in ('mapped_labels','label_valid','completed_valid','periodic_valid','label_periodic_valid'):
                    np.testing.assert_array_equal(actual[key],getattr(result,key),err_msg=mode+':'+key)
                ai=actual['registration']['periodic_completion']; ei=result.registration['periodic_completion']
                for key in ('status','period_source','period_vectors','added_label_pixels','remaining_unknown_pixels','class_periodicity'):
                    self.assertEqual(ai[key],ei[key],mode+':'+key)
if __name__=='__main__': unittest.main()




