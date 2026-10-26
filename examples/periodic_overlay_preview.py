"""Experimental full-image periodic overlay preview for the SD-1 trial.

Run: python -m examples.periodic_overlay_preview
Produces direct and periodic image layers; does not change production matchers.
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.registration import _features, _refine, _score
from periodic_template_matcher.utils import read_image, write_image


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / 'periodic_template_matcher/results'
    out = results / 'periodic_overlay_preview'
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((results / 'sd1_labelme_defect_opencv/result.json').read_text(encoding='utf-8'))
    best = result['matches'][result['ranking'][0]]
    source = read_image(best['template_image'])
    target = read_image(result['target'])
    H = np.asarray(best['matrix_template_to_target'])
    h, w = target.shape[:2]
    cfg = MatcherConfig(**result['config'])
    original_min_overlap = cfg.min_overlap
    cfg.min_overlap = .2  # Period-shifted copies can have a smaller visible area.
    scale = best['working_scale']
    size = (round(w * scale), round(h * scale))
    source_size = (round(source.shape[1] * scale), round(source.shape[0] * scale))
    Da = np.diag([source_size[0]/source.shape[1], source_size[1]/source.shape[0], 1.])
    Db = np.diag([size[0]/w, size[1]/h, 1.])
    sm = np.ones(source.shape[:2], np.uint8)
    sm[int(sm.shape[0]*(1-cfg.bottom_ignore_fraction)):] = 0
    tm = np.ones((h, w), np.uint8)
    tm[int(h*(1-cfg.bottom_ignore_fraction)):] = 0
    src = _features(cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), source_size),
                    cv2.resize(sm, source_size, interpolation=cv2.INTER_NEAREST))
    dst = _features(cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), size),
                    cv2.resize(tm, size, interpolation=cv2.INTER_NEAREST))
    # Use class-1 centers from the previous independent period experiment.
    probe = json.loads((results / 'periodic_labelme_probe/labelme_refined/metrics.json').read_text(encoding='utf-8'))
    a = np.array(probe['horizontal_period_original_px'])
    b = np.array(probe['vertical_period_original_px'])

    def warp(matrix):
        image = cv2.warpPerspective(source, matrix, (w, h))
        valid = (cv2.warpPerspective(sm, matrix, (w, h), flags=cv2.INTER_NEAREST) > 0) & (tm > 0)
        return image, valid

    direct, direct_valid = warp(H)
    completed = direct.copy()
    valid = direct_valid.copy()
    provenance = np.zeros((h, w), np.uint8)
    provenance[direct_valid] = 1
    copies = []
    for m, n in ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
        P = np.eye(3)
        P[:2, 2] = m*a + n*b
        candidate = H @ P
        work = Db @ candidate @ np.linalg.inv(Da)
        refined, converged = _refine(src, dst, work, cfg)
        before, after = _score(src, dst, work, cfg), _score(src, dst, refined, cfg)
        if after['match_score'] >= before['match_score']:
            candidate = np.linalg.inv(Db) @ refined @ Da
            metrics = after
        else:
            metrics = before
        # Require image evidence rather than blindly repeating image content.
        accepted = metrics['structure_ncc'] >= .75 and metrics['edge_alignment'] >= .6
        count = 0
        if accepted:
            image, candidate_valid = warp(candidate)
            fill = candidate_valid & ~valid
            completed[fill] = image[fill]
            provenance[fill] = len(copies)+2
            valid |= fill
            count = int(fill.sum())
        copies.append(dict(period_offset=[m, n], accepted=bool(accepted), added_pixels=count,
                           ecc_converged=bool(converged), metrics=metrics,
                           matrix_template_to_target=candidate.tolist()))
    added = valid & ~direct_valid
    blend = target.copy()
    both = cv2.addWeighted(target, .5, completed, .5, 0)
    blend[valid] = both[valid]
    original_blend = target.copy()
    original_blend[direct_valid] = cv2.addWeighted(target, .5, direct, .5, 0)[direct_valid]
    direct_rgba = cv2.cvtColor(direct, cv2.COLOR_BGR2BGRA)
    direct_rgba[:, :, 3] = direct_valid.astype(np.uint8)*255
    periodic_rgba = cv2.cvtColor(completed, cv2.COLOR_BGR2BGRA)
    periodic_rgba[:, :, 3] = added.astype(np.uint8)*255
    marked = blend.copy()
    for mask, color in ((direct_valid, (70, 220, 60)), (added, (0, 170, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, color, 3)
    for name, image in [('input.jpg', target), ('direct_layer.png', direct_rgba),
                        ('periodic_layer.png', periodic_rgba), ('overlay_50.jpg', blend),
                        ('direct_only_50.jpg', original_blend), ('regions.jpg', marked),
                        ('provenance.png', provenance)]:
        write_image(out / name, image)

    def data_url(name, mime):
        return 'data:' + mime + ';base64,' + base64.b64encode((out / name).read_bytes()).decode('ascii')

    html = '''<!doctype html><meta charset="utf-8"><title>周期模板透明叠加预览</title>
<style>body{background:#15191d;color:#eee;font:16px system-ui;margin:24px}label{margin-right:24px}
.image{position:relative;max-width:1400px}.image img{display:block;width:100%}.image .layer{position:absolute;inset:0;opacity:.5}
input{vertical-align:middle}a{color:#8cd4ff}</style>
<h2>模板与 input 透明叠加（实验预览）</h2>
<p>底图为 input；直接配准区和周期补齐区分别控制透明度。底部设备文字保持 input 原图。</p>
<p><label>直接配准 <input id="a" type="range" min="0" max="100" value="50"> <span id="av">50%</span></label>
<label>周期补齐 <input id="b" type="range" min="0" max="100" value="50"> <span id="bv">50%</span></label></p>
<p><a href="regions.jpg">区域边界：绿色为直接配准，橙色为周期补齐</a> · <a href="overlay_50.jpg">下载 50% 叠加图</a></p>
<div class="image"><img src="INPUT"><img class="layer" id="direct" src="DIRECT"><img class="layer" id="periodic" src="PERIODIC"></div>
<p>周期补齐显示重复单元的参考外观，不是对缺陷的修复或真实观测。</p>
<script>for(const [slider,layer,value] of [['a','direct','av'],['b','periodic','bv']]){
document.getElementById(slider).oninput=function(){document.getElementById(layer).style.opacity=this.value/100;document.getElementById(value).textContent=this.value+'%'}}
</script>'''
    html = html.replace('INPUT', data_url('input.jpg', 'image/jpeg')).replace('DIRECT', data_url('direct_layer.png', 'image/png')).replace('PERIODIC', data_url('periodic_layer.png', 'image/png'))
    (out / 'preview.html').write_text(html, encoding='utf-8')
    summary = dict(note='Experimental full-template periodic overlay; not a segmentation export',
                   original_min_overlap=original_min_overlap, alpha=.5,
                   direct_roi_coverage=float(direct_valid.sum()/tm.sum()),
                   completed_roi_coverage=float(valid.sum()/tm.sum()),
                   added_pixels=int(added.sum()), uncovered_roi_pixels=int(((tm > 0) & ~valid).sum()),
                   copies=copies)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'copies'}, indent=2))
    print(out / 'preview.html')


if __name__ == '__main__':
    main()
