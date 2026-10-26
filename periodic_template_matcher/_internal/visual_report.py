"""Self-contained, offline HTML report and a readable six-panel contact sheet."""
import base64
from html import escape

import cv2
import numpy as np

from .results import EvaluationResult, MatchResult
from ..utils import overlay_labels, to_gray, write_image


def visualize_match(result, output_path=None):
    """Render MatchResult/EvaluationResult as six BGR panels without GUI deps.

    Prediction colors are deterministic class-ID colors. Uncovered/unknown
    regions are dark, never background. In the error view: green=correct,
    red=wrong/missing class, orange=false-positive background, magenta=uncovered
    GT foreground. A rejected match shows alignment for diagnosis, not labels.
    """
    evaluation = result if isinstance(result, EvaluationResult) else None
    match = result.match if evaluation is not None else result
    if not isinstance(match, MatchResult):
        raise TypeError('result must be MatchResult or EvaluationResult')

    def bgr(image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def shaded(image, valid):
        image = bgr(image)
        image[~valid] = (image[~valid].astype(np.float32) * .18).astype(np.uint8)
        return image

    target = bgr(match.input_image)
    valid = match.completed_valid if match.completed_valid is not None else match.geometric_valid
    blend = shaded(target, valid)
    if match.warped_template is not None:
        warped = bgr(match.completed_template if match.completed_template is not None else match.warped_template)
        blend[valid] = cv2.addWeighted(target, .5, warped, .5, 0)[valid]
    prediction = shaded(target, np.zeros(valid.shape, bool))
    if match.mapped_labels is not None:
        prediction = overlay_labels(shaded(target, match.label_valid),
                                     match.mapped_labels, match.label_valid)
    panels = [('TEMPLATE (top ranked)', bgr(match.template_image)),
              ('INPUT', target), ('ALIGNMENT 50/50 (dark=unknown)', blend),
              ('PREDICTED LABELS' if match.mapped_labels is not None else 'NO ACCEPTED LABEL PREDICTION', prediction)]
    legend = 'Class colors: deterministic by ID; dark areas: unknown, not background.'
    if evaluation is not None:
        truth, roi = evaluation.ground_truth, evaluation.ground_truth_valid
        ground_truth = overlay_labels(shaded(target, roi), truth, roi)
        errors = shaded(target, np.zeros(valid.shape, bool))
        if match.mapped_labels is not None:
            known = match.label_valid & roi
            correct = known & (match.mapped_labels == truth)
            wrong = known & (match.mapped_labels != truth)
            errors[correct] = (65, 165, 65)
            errors[wrong] = (55, 55, 235)
            errors[wrong & (truth == 0)] = (0, 165, 255)
            errors[roi & ~match.label_valid & (truth > 0)] = (210, 40, 210)
        panels += [('GROUND TRUTH', ground_truth), ('LABEL ERRORS (not image differences)', errors)]
        legend = 'Errors: green=correct | red=wrong/FN | orange=FP | magenta=uncovered GT foreground'
    else:
        coverage = shaded(target, valid)
        coverage[valid] = (.65 * target[valid] + .35 * np.array([70, 220, 70])).astype(np.uint8)
        known = match.label_valid if match.label_valid is not None else np.zeros(valid.shape, bool)
        panels += [('GEOMETRIC COVERAGE (green)', coverage), ('KNOWN LABEL REGION (dark=unknown)', shaded(target, known))]

    width, height, header, caption = 640, 420, 100, 36
    canvas = np.full((header + 3 * (height + caption), 2 * width, 3), 24, np.uint8)
    state = 'ACCEPTED' if match.accepted else 'REJECTED / ALIGNMENT REVIEW ONLY'
    title = f'{state} | template={match.best_template_index} | image match={match.match_score:.2f}/100'
    cv2.putText(canvas, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, .62, (220, 230, 240), 1, cv2.LINE_AA)
    score_text = 'Segmentation GT unavailable; visual alignment is not segmentation accuracy.'
    if evaluation is not None:
        score = evaluation.score
        score_text = f"Segmentation score={'N/A' if score is None else f'{score:.2f}/100'} | {evaluation.metrics['status']}"
    cv2.putText(canvas, score_text, (16, 53), cv2.FONT_HERSHEY_SIMPLEX, .52, (100, 210, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, legend, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, .46, (210, 210, 210), 1, cv2.LINE_AA)
    for i, (title, image) in enumerate(panels):
        x, y = (i % 2) * width, header + (i // 2) * (height + caption)
        cv2.putText(canvas, title, (x + 14, y + 25), cv2.FONT_HERSHEY_SIMPLEX, .52, (230, 230, 230), 1, cv2.LINE_AA)
        scale = min(width / image.shape[1], height / image.shape[0])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ox, oy = x + (width - small.shape[1]) // 2, y + caption + (height - small.shape[0]) // 2
        canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    if output_path is not None:
        write_image(output_path, canvas)
    return canvas


def _uri(image, max_side=1280):
    scale = min(1., max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if not ok:
        raise ValueError('Cannot encode report image')
    return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')


def _diagnostics(target, warped, valid):
    """Target edges cyan, warped-template edges magenta, coincidences white."""
    gray = to_gray(target)
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(12, 12))
    # Smooth sensor/JPEG grain before drawing native-resolution edges.
    a = clahe.apply(cv2.GaussianBlur(gray, (0, 0), 1.5))
    b = clahe.apply(cv2.GaussianBlur(to_gray(warped), (0, 0), 1.5))
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ea = (cv2.Canny(a, 50, 130) > 0) & safe
    eb = (cv2.Canny(b, 50, 130) > 0) & safe
    edges = (target.astype(np.float32) * .30).astype(np.uint8)
    edges[ea] = (255, 230, 30)
    edges[eb] = (230, 45, 255)
    edges[ea & eb] = (255, 255, 255)
    # Diagnostic only: local contrast-normalized image difference, not defects.
    def norm(x):
        x = cv2.GaussianBlur(x.astype(np.float32) / 255, (0, 0), 1.)
        mean = cv2.GaussianBlur(x, (0, 0), 15)
        var = np.maximum(cv2.GaussianBlur(x*x, (0, 0), 15) - mean*mean, 0)
        return (x - mean) / np.sqrt(var + .0025)
    diff = np.clip(np.abs(norm(gray) - norm(to_gray(warped))) * 100, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    heat[~safe] = (30, 34, 45)
    yy, xx = np.indices(gray.shape)
    cells = ((xx // 140 + yy // 140) % 2 == 0) & valid
    checker = target.copy()
    checker[cells] = warped[cells]
    coverage = target.copy()
    coverage[valid] = (.72 * target[valid] + .28 * np.array([100, 220, 40])).astype(np.uint8)
    coverage[~valid] = (.35 * target[~valid] + .65 * np.array([45, 45, 70])).astype(np.uint8)
    contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(coverage, contours, -1, (100, 255, 100), 3)
    return edges, checker, heat, coverage


def _overview(out, panels, result):
    width = 680
    height = 565
    canvas = np.full((3 * (height + 54) + 90, 2 * width, 3), (24, 19, 15), np.uint8)
    cv2.putText(canvas, 'TEMPLATE REGISTRATION / ANNOTATION TRANSFER', (22, 34),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 240, 240), 2, cv2.LINE_AA)
    state = 'ACCEPTED' if result['accepted'] else 'REJECTED - REVIEW ONLY'
    cv2.putText(canvas, f"{state} | class labels: {'REAL' if result['labels_exported'] else 'UNAVAILABLE / DEMO ONLY'}",
                (22, 68), cv2.FONT_HERSHEY_SIMPLEX, .58, (80, 210, 240), 1, cv2.LINE_AA)
    for index, (title, image) in enumerate(panels[:6]):
        x, y = (index % 2) * width, 90 + (index // 2) * (height + 54)
        scale = min(width / image.shape[1], height / image.shape[0])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ox, oy = x + (width-small.shape[1])//2, y + 44 + (height-small.shape[0])//2
        canvas[oy:oy+small.shape[0], ox:ox+small.shape[1]] = small
        cv2.putText(canvas, title, (x + 16, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (235, 235, 235), 1, cv2.LINE_AA)
    write_image(out / 'overview.jpg', canvas)


def build_report(out, result, images, target, views):
    best = result['matches'][result['ranking'][0]]
    selected = images[best['template_index']]
    available = 'warped' in views
    label_title = '真实分割标注映射' if result['labels_exported'] else '几何示意映射 · 非真实标注'
    source_title = '模板及真实标注' if result['labels_exported'] else '模板及几何示意 · 非真实标注'
    panels = [('01 TEMPLATE (original)', selected), ('02 TARGET (original)', target)]
    cards = [('模板原图', selected, '原始坐标系；底部设备文字不参与本例匹配。'),
             ('目标原图', target, '所有候选模板使用同一个目标有效区域和评分标准。')]
    compare = ''
    if available:
        warped, valid = views['warped'], views['valid']
        edges, checker, heat, coverage = _diagnostics(target, warped, valid)
        aligned = target.copy()
        aligned[valid] = warped[valid]
        panels += [('03 ALIGNED BLEND (50/50)', views['blend']),
                   ('04 EDGE CHECK: cyan=target magenta=template', edges),
                   ('05 COVERAGE: green=valid dark=unknown', coverage),
                   ('06 ' + ('REAL LABEL TRANSFER' if result['labels_exported'] else
                             'DEMO SHAPES ONLY / NOT SEGMENTATION') if 'label_target' in views else
                    '06 CHECKERBOARD', views.get('label_target', checker))]
        cards += [('配准后 50% 叠加', views['blend'], '检查结构轮廓是否重合；重影提示残余配准差。'),
                  ('边缘对齐', edges, '青色＝目标边缘；品红＝模板边缘；白色＝像素级重合。未覆盖处不计算。'),
                  ('棋盘格切换', checker, '交替显示模板和目标；格子边界处结构应连续，亮度变化不等同错位。'),
                  ('有效几何覆盖', coverage, '绿色＝模板可覆盖且匹配有效的区域；暗色＝未知／已排除。'),
                  ('局部归一化差异', heat, '冷色差异小，暖色差异大。受模糊、光照、配准影响，不是缺陷检测或分割结果。')]
        compare = f'''<section class="card"><div class="section-head"><div><h2>交互式配准检查</h2>
<p>左侧为目标，右侧为变换后的模板；未覆盖区域保持目标原图。</p></div><span class="tag">拖动分割线</span></div>
<div class="compare" id="compare"><img src="{_uri(target)}" alt="目标图"><img class="top" id="aligned" src="{_uri(aligned)}" alt="配准模板"><div id="divider"></div>
<span class="left-label">目标</span><span class="right-label">已配准模板</span></div>
<div class="controls"><label>分割位置 <input id="split" type="range" min="0" max="100" value="50"></label>
<label>模板透明度 <input id="alpha" type="range" min="0" max="100" value="100"></label><button id="blink">按住查看目标</button></div></section>'''
    if 'label_target' in views:
        cards += [(source_title, views['label_source'], '相同颜色代表相同类别 ID；示意形状仅用于检查坐标迁移。' if not result['labels_exported'] else '原始模板上的真实类别标注。'),
                  (label_title, views['label_target'], '圆形、矩形、多边形均为任意绘制的流程示意，不能用于训练或生产。' if not result['labels_exported'] else '迁移模板有效标注；启用周期补齐时，包含经图像验证的周期预测。')]
    if not available:
        panels += [('03 NO REGISTRATION - NO LABEL TRANSFER', target)] * 4
    _overview(out, panels, result)
    rows = []
    for rank, index in enumerate(result['ranking'], 1):
        item = result['matches'][index]
        aliases = '、'.join(a['name'] for a in result['template_aliases'][index])
        rows.append(f'''<tr><td>#{rank}</td><td><strong>{escape(item['template_name'])}</strong><small>{escape(aliases)}</small></td>
<td><div class="bar"><i style="width:{item['match_score']:.1f}%"></i></div>{item['match_score']:.2f}</td>
<td>{item.get('target_coverage', 0):.1%}</td><td>{'有' if item['annotation_available'] else '待标注'}</td>
<td>{'通过' if item['geometry_accepted'] else '拒绝'}</td></tr>''')
    gallery = ''.join(f'''<article class="card image-card"><h3>{escape(title)}</h3>
<a href="{_uri(image)}" target="_blank"><img loading="lazy" src="{_uri(image)}" alt="{escape(title)}"></a>
<p>{escape(caption)}</p></article>''' for title, image, caption in cards)
    warnings = ''.join(f'<li>{escape(w)}</li>' for w in result['warnings'])
    matrix = best.get('matrix_template_to_target')
    matrix_text = '\n'.join('  '.join(f'{v:12.6f}' for v in row) for row in matrix) if matrix else '无可用变换'
    accepted = result['accepted']
    status = '匹配已接受' if accepted else '匹配未接受 · 仅供诊断'
    annotation = '已输出真实标签' if result['labels_exported'] else '真实标注尚未输出'
    gap = result['template_score_gap']
    gap_text = f'{gap:.2f}' if gap is not None else '不适用（单模板）'
    link_names = ['overview.jpg', 'result.json']
    if available:
        link_names += ['matrix_template_to_target.npy', 'geometric_valid.png', 'best_alignment.jpg']
    if result['labels_exported']:
        link_names += ['mapped_labels.png', 'label_valid.png', 'label_overlay.jpg']
    if result['demo_labels_exported']:
        link_names += ['demo_source_labels.png', 'demo_mapped_labels.png', 'demo_label_valid.png', 'demo_label_overlay.jpg']
    periodic_section = ''
    if 'periodic_completion' in result:
        completion = result['periodic_completion']
        link_names += ['periodic_overlay.html', 'completed_alignment.jpg', 'completed_geometric_valid.png',
                       'periodic_valid.png', 'label_periodic_valid.png', 'completion_regions.jpg']
        periodic_section = f'''<section class="card"><h2>周期补齐</h2>
<p>有效 ROI 标签覆盖率：{completion['label_coverage_roi']:.2%}；剩余未知像素：{completion['remaining_unknown_pixels']}；状态：{escape(completion['status'])}</p>
<p><a href="periodic_overlay.html">打开直接配准 / 周期补齐双透明度预览</a></p></section>'''
    links = ''.join(f'<a download href="{name}">{name}</a>' for name in link_names)
    region_section = ''
    if 'valid_regions' in result:
        regions = result['valid_regions']
        region_rows = ''.join(f"<li>{escape(name)}：有效高度 {region['valid_height']} / {region['image_height']}，有效像素 {region['valid_pixels']}</li>"
                              for name, region in [('input', regions['target'])] +
                              list(zip(result['template_names'], regions['templates'])))
        region_section = f'<section class="card"><h2>匹配前有效区域</h2><p>高度按原图像素计数，仅使用顶部 [0, H)；同时与图像 mask 取交集。</p><ul>{region_rows}</ul></section>'
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>模板匹配与标注映射 · SD-1</title><style>
:root{{--bg:#0a1020;--card:#121d32;--line:#293851;--text:#e7edf8;--muted:#9aaecb;--accent:#52ddc2}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at top right,#16354b,var(--bg) 55%);color:var(--text);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1450px;margin:auto;padding:40px 28px}}
h1{{font-size:34px;letter-spacing:-1px;margin:8px 0}}h2{{font-size:22px;margin:0 0 10px}}h3{{margin:0 0 16px;font-size:17px}}p{{color:var(--muted);margin:6px 0}}.eyebrow{{color:var(--accent);letter-spacing:3px;font-size:12px}}.tag{{display:inline-block;padding:5px 12px;border:1px solid #397362;border-radius:30px;color:var(--accent);white-space:nowrap}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:17px;padding:23px;margin:22px 0;overflow:hidden}}.section-head{{display:flex;justify-content:space-between;gap:20px;margin-bottom:20px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-top:26px}}.metric{{background:#142339;border:1px solid var(--line);border-radius:14px;padding:20px}}.metric b{{display:block;font-size:29px;margin:6px 0;color:var(--accent)}}.metric span,.metric small{{color:var(--muted)}}.warning{{border-left:4px solid #ffcb72;background:#302a22}}.warning li{{color:#ffe0aa;margin:5px 0}}
table{{width:100%;border-collapse:collapse;text-align:left}}th,td{{padding:14px;border-bottom:1px solid var(--line)}}th,small{{color:var(--muted)}}td small{{display:block}}.bar{{display:inline-block;width:100px;height:6px;background:#304158;margin-right:12px;border-radius:5px;vertical-align:middle}}.bar i{{display:block;height:100%;background:var(--accent);border-radius:5px}}.table-wrap{{overflow:auto}}
.compare{{position:relative;overflow:hidden;background:#090d16;border-radius:10px}}.compare img{{display:block;width:100%}}.compare .top{{position:absolute;inset:0;clip-path:inset(0 0 0 50%)}}#divider{{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#fff;pointer-events:none}}.left-label,.right-label{{position:absolute;top:12px;background:#0a1020cc;border-radius:5px;padding:5px 12px}}.left-label{{left:12px}}.right-label{{right:12px}}.controls{{display:flex;align-items:center;gap:24px;flex-wrap:wrap;margin-top:18px}}input{{vertical-align:middle;accent-color:var(--accent)}}button{{background:#1d3e4c;border:1px solid #397362;color:var(--text);padding:8px 16px;border-radius:8px;cursor:pointer;touch-action:none}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}.image-card{{margin:0;padding:18px}}.image-card img{{width:100%;border-radius:8px}}.image-card p{{margin-top:12px;font-size:13px}}pre{{overflow:auto;background:#0a1322;padding:18px;border-radius:9px;color:#9ee6dc}}.downloads{{display:flex;gap:12px;flex-wrap:wrap}}a{{color:#74dcd2}}.downloads a{{border:1px solid var(--line);padding:8px 12px;border-radius:7px;text-decoration:none}}footer{{color:var(--muted);text-align:center;padding:25px;font-size:13px}}
@media(max-width:800px){{main{{padding:20px 14px}}h1{{font-size:26px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.gallery{{grid-template-columns:1fr}}.section-head{{display:block}}.bar{{width:55px}}}}
</style></head><body><main><header><div class="eyebrow">VISION WORKBENCH / REGISTRATION</div><h1>周期结构 · 模板匹配与分割映射</h1>
<p>目标：{escape(result['target'].split('/')[-1])} ｜ 原图 {target.shape[1]} × {target.shape[0]} ｜ 全局仿射配准</p>
<div style="margin-top:16px"><span class="tag">{status}</span> <span class="tag">{annotation}</span></div></header>
<div class="metrics"><div class="metric"><span>最佳模板匹配分</span><b>{best['match_score']:.2f}<small> /100</small></b><small>工程相似度，不是概率</small></div>
<div class="metric"><span>目标有效 ROI 覆盖</span><b>{best.get('target_coverage', 0):.1%}</b><small>不含已排除的底部文字区</small></div>
<div class="metric"><span>唯一模板 / 输入条目</span><b>{result['unique_template_count']} / {result['input_template_count']}</b><small>重复图像不参与模板竞争</small></div>
<div class="metric"><span>结构相关性 NCC</span><b>{best.get('structure_ncc', 0):.3f}</b><small>最佳模板：{escape(best['template_name'])}</small></div></div>
<section class="card warning"><h3>使用边界与标注状态</h3><ul>{warnings}</ul><p>状态：{escape(result['status'])} ｜ 标签模式：{escape(result['label_mode'])} ｜ 模板分差：{gap_text}</p></section>
<section class="card"><h2>模板候选排行榜</h2><div class="table-wrap"><table><thead><tr><th>排序</th><th>模板 / 原始别名</th><th>匹配分</th><th>ROI 覆盖</th><th>真实标注</th><th>几何检查</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>同一目标 ROI、同一工作分辨率评分。分数或模板分差不足时，不输出正式迁移标签。</p></section>
{region_section}{periodic_section}{compare}<h2 style="margin:30px 0 18px">可视化检查面板</h2><div class="gallery">{gallery}</div>
<section class="card"><h2>坐标变换与诊断</h2><p>下列 3×3 矩阵由模板原始像素 [x, y, 1] 映射到目标原始像素；类别掩码始终使用最近邻采样。</p><pre>{matrix_text}</pre>
<p>score = 100 × (0.55 × NCC + 0.30 × 边缘对齐 + 0.15 × 局部一致性) × √覆盖率</p>
<p>边缘对齐：{best.get('edge_alignment', 0):.3f} ｜ 局部一致性：{best.get('local_consistency', 0):.3f} ｜ ECC 收敛：{best.get('ecc_converged', False)} ｜ 周期结构：{best.get('periodic_detected', False)}</p>
<p>只有 label_valid 非零的位置才有有效类别预测；0 可以是背景，也可以是无标签填充值。周期补齐像素由 label_periodic_valid 单独记录，验证失败的位置保持未知。</p></section>
<section class="card"><h2>下载结果</h2><p>本页图像已内嵌，可离线浏览；下载链接需与结果文件保存在同一目录。</p><div class="downloads">{links}</div></section>
<footer>无真实标注时仅验证配准与坐标迁移流程，不宣称真实分割精度或绝对周期身份。</footer></main>
<script>const split=document.getElementById('split'),alpha=document.getElementById('alpha'),aligned=document.getElementById('aligned'),divider=document.getElementById('divider');
if(split){{const update=()=>{{aligned.style.clipPath='inset(0 0 0 '+split.value+'%)';divider.style.left=split.value+'%';aligned.style.opacity=alpha.value/100;}};
split.oninput=update;alpha.oninput=update;const blink=document.getElementById('blink');blink.onpointerdown=e=>{{blink.setPointerCapture(e.pointerId);aligned.style.visibility='hidden';}};blink.onpointerup=blink.onpointercancel=()=>{{aligned.style.visibility='visible';}};
const compare=document.getElementById('compare');const move=e=>{{const r=compare.getBoundingClientRect();split.value=Math.max(0,Math.min(100,100*(e.clientX-r.left)/r.width));update();}};
compare.onpointerdown=e=>{{compare.setPointerCapture(e.pointerId);move(e);}};compare.onpointermove=e=>{{if(compare.hasPointerCapture(e.pointerId))move(e);}};update();}}</script></body></html>'''
    (out / 'report.html').write_text(html, encoding='utf-8')
