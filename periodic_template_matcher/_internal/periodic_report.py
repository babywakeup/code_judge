"""Offline two-opacity preview for direct and periodic template image layers."""
import base64

import cv2
import numpy as np

from ..utils import write_image


ARTIFACTS = ('periodic_overlay.html', 'completed_alignment.jpg', 'completed_geometric_valid.png',
             'periodic_valid.png', 'label_periodic_valid.png', 'completion_regions.jpg')


def bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def export_completion(out, match):
    target, complete = bgr(match.input_image), bgr(match.completed_template)
    blend = target.copy()
    blend[match.completed_valid] = cv2.addWeighted(target, .5, complete, .5, 0)[match.completed_valid]
    write_image(out/'completed_alignment.jpg', blend)
    for name, mask in (('completed_geometric_valid.png', match.completed_valid),
                       ('periodic_valid.png', match.periodic_valid),
                       ('label_periodic_valid.png', match.label_periodic_valid)):
        write_image(out/name, mask.astype(np.uint8)*255)
    marked = blend.copy()
    for mask, color in ((match.geometric_valid, (70, 220, 60)), (match.periodic_valid, (0, 170, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, contours, -1, color, 3)
    write_image(out/'completion_regions.jpg', marked)

    def uri(image, mask=None):
        if mask is not None:
            image = cv2.cvtColor(bgr(image), cv2.COLOR_BGR2BGRA)
            image[:, :, 3] = mask.astype(np.uint8)*255
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Could not encode completion preview')
        return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')

    info = match.registration['periodic_completion']
    html = '''<!doctype html><meta charset="utf-8"><title>周期补齐叠加</title>
<style>body{background:#15191d;color:#eee;font:16px system-ui;margin:24px}label{margin-right:24px}
.image{position:relative;max-width:1400px}.image img{display:block;width:100%}.layer{position:absolute;inset:0;opacity:.5}a{color:#8cd4ff}</style>
<h2>直接配准与周期补齐</h2><p>SUMMARY</p>
<p><label>直接配准 <input id="a" type="range" min="0" max="100" value="50"><span id="av">50%</span></label>
<label>周期补齐 <input id="b" type="range" min="0" max="100" value="50"><span id="bv">50%</span></label></p>
<p><a href="completion_regions.jpg">绿色：直接配准；橙色：周期补齐</a> · <a href="report.html">匹配报告</a></p>
<div class="image"><img src="TARGET"><img class="layer" id="direct" src="DIRECT"><img class="layer" id="periodic" src="PERIODIC"></div>
<p>周期复制的结构及类别属于预测；缺陷区域仍须单独检查。被排除的 ROI 保留 input 原图。</p>
<script>for(const [s,l,v] of [['a','direct','av'],['b','periodic','bv']]){document.getElementById(s).oninput=function(){document.getElementById(l).style.opacity=this.value/100;document.getElementById(v).textContent=this.value+'%'}};</script>'''
    html = html.replace('SUMMARY', f"有效 ROI 标签覆盖率 {info['label_coverage_roi']:.2%}；剩余未知像素 {info['remaining_unknown_pixels']}")
    html = html.replace('TARGET', uri(target)).replace('DIRECT', uri(match.warped_template, match.geometric_valid)).replace('PERIODIC', uri(complete, match.periodic_valid))
    (out/'periodic_overlay.html').write_text(html, encoding='utf-8')
    return blend
