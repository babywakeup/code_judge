"""Python IO/API adapter. All matching, completion and metric computation is native."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from periodic_template_matcher.config import MatcherConfig
from periodic_template_matcher._internal.models import MatchResult, EvaluationResult
from periodic_template_matcher._internal.io.export import save_result
from periodic_template_matcher._internal.io.workflow import InferenceFiles
from . import _native


class PeriodicTemplateMatcher:
    def __init__(self,config=None):
        if config is not None and not isinstance(config,MatcherConfig):
            raise TypeError('config must be MatcherConfig')
        self.config=deepcopy(config) if config is not None else MatcherConfig()
        self.config.validate()

    def match(self,templates,masks,input,*,template_valids=None,annotation_valids=None,
              target_valid=None,template_valid_heights=None,target_valid_height=None,template_objects=None):
        return MatchResult(**_native.match(templates,masks,input,asdict(self.config),
            template_valids,annotation_valids,target_valid,template_valid_heights,target_valid_height,template_objects))

    def evaluate(self,templates,masks,input,input_masks,*,input_valid=None,template_valids=None,
                 annotation_valids=None,target_valid=None,template_valid_heights=None,target_valid_height=None,
                 template_objects=None):
        if input_masks is None:
            raise ValueError('input_masks must be a class-ID array or binary mask list')
        # Reuse the native input boundary to validate/normalize GT without exposing
        # ground truth to matching or completion. This call performs no matching.
        prepared=_native.prepare_inputs([input],[input_masks],input,
            dict(asdict(self.config),ordered_object_class=None),[target_valid],None,target_valid,
            [target_valid_height],target_valid_height,None)
        truth=prepared['labels'][0]
        roi=prepared['target_valid']>0
        if input_valid is not None:
            if not isinstance(input_valid,np.ndarray) or input_valid.shape!=roi.shape:
                raise ValueError('input_valid must match image height and width')
            roi &= input_valid>0
        matched=self.match(templates,masks,input,template_valids=template_valids,
            annotation_valids=annotation_valids,target_valid=target_valid,
            template_valid_heights=template_valid_heights,target_valid_height=target_valid_height,
            template_objects=template_objects)
        if matched.mapped_labels is None:
            metrics=dict(status='match_rejected' if not matched.accepted else 'no_template_annotations',
                score=None,mean_iou=None,mean_dice=None,pixel_accuracy=None,
                prediction_coverage=0.0 if roi.any() else None,evaluated_pixels=0,per_class=[],overlap_only=None)
        else:
            metrics=_native.evaluate_segmentation(matched.mapped_labels,truth,matched.label_valid,roi,
                                                  self.config.include_background)
        return EvaluationResult(matched,metrics,truth,roi)

    def save_result(self,result,output_dir):
        return save_result(result,output_dir)


def run_config(config_path,output_dir):
    files = InferenceFiles(config_path)
    target_path = files.target_path(None)
    templates = files.read_templates()
    target, valid = files.read_target(target_path)
    matcher = PeriodicTemplateMatcher(files.config)
    result = matcher.match(templates.images, templates.labels, target,
                           template_valids=templates.valids, annotation_valids=templates.annotations,
                           target_valid=valid, template_valid_heights=templates.heights,
                           target_valid_height=files.target_height, template_objects=templates.objects)
    files.annotate(result, target_path, templates.labels)
    out = Path(output_dir).resolve()
    files.validate_output(out, target_path)
    manifest = matcher.save_result(result, out)
    return json.loads(manifest.read_text(encoding='utf-8'))
