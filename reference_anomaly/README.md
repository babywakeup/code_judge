# 输入图与正常参考图的异常显化

比较 `periodic_template_matcher` 保存的输入图和校正后正常参考图，生成通用异常线索，
便于人工检查或后续分割。缺陷不限定为白色；不读取 XML 标注、原缺陷框或目标真值。
沿用项目已有 NumPy/OpenCV 环境，无需安装分割模型。

在项目根目录运行：

```powershell
.venv\Scripts\python.exe -m reference_anomaly --result periodic_template_matcher/results/sd1_opencv
```

默认输出到 `periodic_template_matcher/results/sd1_opencv/anomaly/`，不改写 matcher 结果。
也可以传入 `result.json` 文件，或用 `--out` 指定独立输出目录。

```powershell
.venv\Scripts\python.exe -m reference_anomaly --result periodic_template_matcher/results/sd1_opencv --tolerance-radius 2 --high-threshold 1.8 --low-threshold 1.0
```

## 主要输出

| 文件 | 用途 |
|---|---|
| `anomaly_overview.jpg` | 输入、正常参考、严格热图、容差热图、叠加图和候选/优先候选对比 |
| `anomaly_heatmap.png` | 严格差异强度热图 |
| `anomaly_tolerant_heatmap.png` | 允许小幅位移后的差异强度热图 |
| `anomaly_overlay.png` | 在原图上突出较强的严格响应 |
| `anomaly_candidates.png` | 严格分支的连通候选区域，255 为候选 |
| `anomaly_priority.png` | 容差后仍然明显、且处于高可信范围的优先候选，255 为候选 |
| `anomaly_valid.png` | 可比较区域；未知区域为 0，不能当作正常 |
| `anomaly_confidence.png` | 255：参考可靠且远离接缝；128：靠近接缝/无效区；0：不可比较 |
| `anomaly_color.png` / `anomaly_structure.png` / `anomaly_multiscale.png` | 各独立信号的可视化 |
| `anomaly_scores.npz` | 原尺寸 float32 严格、容差分数与三类原始响应，可用于后续处理 |
| `anomaly.json` | 参数、标定尺度、来源光度校正记录、候选峰值点及面积、输出路径 |

热图显示范围固定为分数 0..3，采用 gamma 1.4；灰色表示不可比较区域。
分数是相对于背景变化尺度的响应强度，不是缺陷概率。PNG 会截断显示范围，
后续模型处理应读取 NPZ 原始浮点数据，并同时读取有效 mask。
独立信号 PNG 显示各自响应 / 标定尺度的 0..2；NPZ 保留未截断的原始响应。

`anomaly_scores.npz` 的键：`strict_score`、`tolerant_score`、`color_response`、
`structure_response`、`multiscale_response`。颜色响应为 Lab 距离，其中 L 乘 2.55；
结构响应为灰度 Sobel 向量差，平滑响应为带有效 mask 归一化高斯平滑后的 Lab 距离。
不同信号独立标定后取最大值，不要求所有信号同时触发。
结构响应默认乘以 **1.5** 后参与严格与容差分支合成，颜色和平滑信号权重仍为 1。
可使用 `--structure-weight 2.0` 进一步加强，或 `--structure-weight 1.0` 恢复原权重。
这是响应倍数，不是固定百分比；`anomaly_structure.png` 显示加权后的响应，
NPZ 中的 `structure_response` 保留未加权原值，实际权重记录在 `anomaly.json.config`。

## 如何解释两条分支

### 实验：异常显著性热图

新增 `anomaly_salience_overview.jpg`，并排显示输入、严格图、容差图、显著性图、
显著性叠加图与显著性候选。可先看这张图比较背景抑制效果。
单独文件为 `anomaly_salience_heatmap.png`、`anomaly_salience_overlay.png`、
`anomaly_salience_candidates.png`，重新运行原命令即可生成。

该分支根据正常参考图的平滑梯度复杂度分组，计算每组常见严格残差的中位数和 MAD 尺度，
显著性为 `max(0, (strict - median) / scale - 0.5)`，其中尺度下限为 0.5。
再乘以 `0.25 + 0.75 * tolerant / strict`（比值限制在 0..1）；接缝附近再乘 0.5。
因此局部超出背景的变化更突出，可被小位移解释的响应会变弱，仍保留部分响应。
纹理分组样本不足 64 时使用全局统计；优先用远离接缝的可靠像素拟合。
显著性候选使用高/低阈值 3.0/1.5，并限定在原始严格候选内。
NPZ 新增 `salience_score` 和 `salience_weight`，JSON 的 `salience` 保存分组与参数。

这是局部异常显著性试验，未使用缺陷真值。常见或大面积一致变化可能被压低，
需同时检查保留的严格/容差图；显著性候选减少不代表缺陷准确率得到验证。

### 原始严格与容差分支

严格分支保留明暗、异色、结构和细小异常。容差分支允许同一来源片段内最多 2 原图像素
的位移（`--tolerance-radius` 可设 0..5），对所有颜色通道和结构/平滑信号使用同一个候选位移。
容差只影响优先级，严格分数与候选保持独立。

候选以高阈值像素为种子，在低阈值连通区域内扩展，不按面积删除细小区域。
默认高/低阈值 1.8/1.0 是初始设置，尚未用缺陷真值评估。
颜色、结构、平滑响应的标定下限分别为 10、6、8，使用 4.5 倍 MAD 稳健尺度，
上限为各自下限的 3 倍；不减去整体残差中位数，因此一致的整体变化仍可产生响应。
可用较低阈值提高召回，也可提高阈值减小人工检查范围；原始分数不受候选阈值影响。

候选 JSON 最多记录按峰值强度、面积排序的前 200 个连通区域；`region_count` 记录总数，
`regions_truncated` 提示是否截断，mask 保留所有候选。`peak_xy` 是原图坐标的候选内部峰值点，
可作为后续分割提示的起点，并不代表最终精确轮廓。候选/优先候选对比图中粉红色为候选，
黄色为优先候选。

## 多窗口 SSIM 和正常模板基准试验

原命令自动增加多窗口 SSIM 比较；若同时试验正常模板基准，运行：

```powershell
.venv\Scripts\python.exe -m reference_anomaly --result periodic_template_matcher/results/sd1_opencv --normal-config periodic_template_matcher/config.sd1.labelme.opencv.json
```

`--normal-config` 使用配置中的正常模板图、语义标注、有效高度和 mask；不读取配置的缺陷
`target`，不修改配置和 matcher 结果。每次将一张正常模板作为目标，其余正常模板作为参考，
只采用通过配准验证、光度拟合可靠、语义类别一致的有效区域。
当前基准排除启用独立身份匹配的类别（SD-1 为类别 2），因为相同类别不能证明实例身份一致。
对缺陷输入应用基准时，还要求已保存的 `prediction_valid` 为有效，未知语义标签不会被当成背景。
重复图像不会作为独立正常样本，所有配置的原始文件（包括重复别名）均受覆盖保护。

| 新增文件 | 含义 |
|---|---|
| `anomaly_methods_overview.jpg` | 输入、旧显著性、增强显著性、独立 SSIM、正常基准及增强叠加的对比 |
| `anomaly_ssim_heatmap.png` / `anomaly_ssim_overlay.png` | 独立多窗口 SSIM 热图与叠加图 |
| `anomaly_ssim_valid.png` | SSIM 具有完整有效窗口且未跨来源接缝的位置 |
| `anomaly_normal_heatmap.png` / `anomaly_normal_overlay.png` | 仅由正常模板统计标定的差异图 |
| `anomaly_normal_valid.png` | 正常模型覆盖的类别及纹理组；0 为缺少可靠基准 |
| `anomaly_supported_salience_heatmap.png` / `anomaly_supported_salience_overlay.png` | 用新证据有上限地增强已有显著性 |

SSIM 在灰度图上用 5、11、21 像素高斯窗口计算完整亮度/对比度/结构相似度；
每窗口响应为 `clip(1 - SSIM, 0, 2)`，取最大响应，除以 0.15 后乘以已有错位/接缝降权。
这属于全分辨率多窗口 SSIM，并非标准下采样乘积型 MS-SSIM。窗口穿过无效区或模板片段
接缝时不可比较，灰色为未知；颜色变化仍由原 Lab 分支保留。

正常模型在固定参考纹理组（梯度切分 1.5、3、6）和语义类别内，估计正常图之间
颜色、Sobel 结构、平滑颜色、SSIM 残差的中位数和 MAD 尺度。各信号尺度下限为
4、2、4、0.03；至少 128 样本才建立对应组。
输入图按 `max(0, (response - normal_median) / normal_scale - 0.5)` 比较，结构信号继续
使用配置的权重，取各信号最大值并乘错位/接缝降权。缺少类别、纹理组或有效窗口时保持未知。
`anomaly.json.normal_calibration` 保存正常图配准记录、统计量和覆盖范围，未指定配置时
状态为 `not_requested`，无法建立正常统计时为 `insufficient_normal_samples`。

直接 SSIM 和少量正常模板的统计仍可能突出正常纹理及几何残差。融合图以已有显著性为主：
`supported = salience * (1 + max(clip(ssim_score / 3), clip(normal_score / 3)))`，
其中 clip 限制在 0..1，且没有可靠正常基准的位置不使用该项。
增强倍数在 1..2，原显著性为 0 的位置保持 0，不直接把新响应加入候选。
方法对比图中的旧/增强显著性采用相同 **0..6** 显示范围，独立 SSIM 和正常基准为 0..3。
这避免仅靠改色阶比较增强效果；既有严格、容差、显著性分数及候选均保持原算法。

NPZ 额外保存 `ssim_response`、`ssim_response_5/11/21`、`ssim_score`、`normal_score`、
`supported_salience_score`。应同时读取对应有效 mask。
只有两张正常模板时，正常变化覆盖有限；独立新图未必优于旧显著性图，不能据热度或
候选范围判断缺陷准确率。当前仍将这些输出作为可比较的试验分支。

## 限制

参考残余错位、正常纹理差异和接缝会产生响应，候选不能直接视为缺陷真值。
可信图表达参考来源与接缝信息，也不是模型置信概率。
前级光度校正可能吸收整片均匀变化；仅凭校正后的参考图和输入图无法恢复已被吸收的变化。
`anomaly.json.source.normal_reference` 保留前级校正参数，供后续检查。
当前模块不调用 SAM 类模型，也不执行 contour 连接数量统计。

数组 API：

```python
from reference_anomaly import AnomalyConfig, analyze_pair

result = analyze_pair(input_image, normal_reference, reference_valid,
                      photometric_valid=reference_photometric_valid,
                      source_map=reference_source_map,
                      config=AnomalyConfig(tolerance_radius=2))
```

图像须为已对齐、同尺寸的 uint8 灰度/BGR/BGRA；不参与 alpha 通道的差异比较。
所有输入保持不变。没有可比较像素时输出空候选与明确的 `no_comparable_pixels` 状态。

验证：`.venv\Scripts\python.exe -m unittest tests.test_reference_anomaly`

## 检测框 + 异常种子的图割基线

独立模块 [box_graphcut.py](box_graphcut.py) 提供数组 API `predict_boxes`；
[box_prediction.py](box_prediction.py) 读取已保存 matcher/anomaly 产物，不重新匹配、
不改动原有异常分支。依赖现有 SciPy 稀疏最大流，无需新增模型或安装依赖。

预测入口为 `python -m reference_anomaly.box_prediction`，必需 `--result` 指向 matcher
结果目录，再提供 `--config` 或同时提供 `--target` / `--boxes`。
配置**只提取 target 和 box**，不调用可能读取标注的 matcher 配置加载器；
`GT_labelme` 字段不解析路径、不打开文件、不参与种子、阈值或结果选择。
`--anomaly` 可指定现有异常目录，默认读取结果目录下 anomaly 的产物。

算法固定默认值如下，均在首次实际预测前确定：

- 检测框每侧增加 10%、至少 16 像素上下文。默认 XML 为 VOC **1-based inclusive**，
    转为内部 0-based half-open；若数据为后者，用 `--xml-coordinates zero_half_open` 显式指定。
    XML 尺寸必须匹配；越界框会裁剪上下文，完全在图外的框记录 `outside_image`。
- 默认使用未截断的 `salience_score`，不是热图 PNG；也可事先固定使用
    `--score supported_salience_score`。不能根据当前 GT 在两者中选优。
- 框内可靠参考（confidence=255）且分数 ≥3 的连通区域提供前景种子：按峰值排序最多
    16 个组件，每个取最高分的 10%、至少 1、最多 32 像素。截断会明确提示。
- 只有**所有检测框外**、可靠且分数 ≤1 的像素成为硬背景。框内低热度像素并非硬背景。
- 分数 sigmoid 与种子学习的有符号 Lab 参考残差多原型构成 unary，外框惩罚为软约束；
    原输入 Lab 四邻域对比构成平滑项，SciPy 求 s-t 最小割，容量量化尺度 1024。
    不以白色为先验，多个检测框独立分割再 union，结果不强行裁回检测框。
- 无可靠参考、无前景/背景种子时明确 abstain；单 ROI 超过 150 万像素也 abstain，
    不静默缩放。未知参考不补造、不输出为缺陷；评价时该区域仍纳入可能的 FN。
- `touches_box`、`touches_roi`、`touches_unknown_reference` 提示可能不完整或边界外溢，
    不能解释为已获得完整目标。

默认新建 anomaly/box_segmentation 输出目录，`--out` 可指定其他**不存在**的目录。
拒绝覆盖已有目录或源文件，预测 manifest 最后写入，作为完成标志。
保存内容包括 mask、带孔洞层级的 contours、原图、种子图、叠加图及 reference validity；
**mask 是精确结果**，contours 保留 OpenCV `RETR_TREE` 层级。
来源文件、配置参数、算法文件和全部预测产物都记录 SHA-256，便于追踪和冻结核验。
种子图绿色为前景、蓝色为背景、灰色为未知参考。

### 独立事后评价

预测完成后，另行运行 `python -m reference_anomaly.box_evaluation`，
提供 `--prediction` 指向冻结预测目录，`--gt` 指向 LabelMe 轮廓文件。
评价器先核验**全部预测产物哈希**，再打开 GT；结果写入同级 box_segmentation_evaluation
新目录，可用 `--out` 改为其他新目录，绝不写回预测目录。

GT 支持 polygon / rectangle，非 ignore 形状取 union，像素坐标四舍六入五成双后用
OpenCV inclusive 填充；错误尺寸、越界点、不支持的形状会报错，不静默忽略。
检查 `imagePath` 文件名和尺寸；若对应外部原图存在，必须与冻结输入逐像素一致。
LabelMe 内嵌图像也会解码并核验：若外部原图已验证，内嵌副本不同会记录警告；
没有外部原图时，内嵌图必须逐像素一致。两者都不存在时明确记录仅名称/尺寸匹配。

评价覆盖**全图**，不限定在框内或有效参考内；输出 TP/FP/FN/TN、Precision/Recall/IoU/Dice、
GT 参考覆盖率与未覆盖像素数。空预测且空 GT 时四指标为 1，其他零分母为 0。
对比图绿色 TP、红色 FP、蓝色 FN，另存 GT mask 与带预测/GT 哈希的指标记录。

### 首次实际运行（2026-11-18）

使用现有 SD-1 结果和配置，默认 `salience_score`，单框固定参数预测后再读取 GT，
没有根据 GT 重选种子、阈值或 mask：

| 指标 | 数值 |
|---|---:|
| 预测前景像素 | 36,540 |
| TP / FP / FN | 21,783 / 14,757 / 23,276 |
| Precision | 59.61% |
| Recall | 48.34% |
| IoU | 36.42% |
| Dice | 53.39% |
| GT 参考覆盖率 | 100%（未覆盖 0 像素） |

预测存在触框、触 ROI、触未知参考、种子组件截断提示。该图 GT 内嵌副本与输入不同，
但其指向的外部原图与冻结输入逐像素一致，已记录警告。
当前基线误检/漏检明显，不能据此称为可靠精确分割，也不进行本图 GT 驱动调参。

合成测试覆盖容量方向与穷举能量、明暗/多色多组件、框外生长、缺参考/种子拒绝、
XML 坐标、GT 路径隔离、冻结输出保护、像素评价算术、来源匹配和哈希防篡改。

## SAM 3：本地权重与异常提示接入

新增可选入口 `python -m reference_anomaly.sam3_prediction`。它不修改 matcher、
原异常分支或图割基线，不读取 GT，不提供视频推理入口；Transformers 框模式支持可选文本。
SAM 看的是**原图 RGB 裁剪**，异常图只生成点提示，不作为输入图像替代品。

### 本地文件与环境

- 默认 `--backend auto`：本地目录中存在 model.safetensors 时选择 **Transformers**，
    否则选择官方 `.pt` 后端；选定后加载失败不会偷偷换后端。
    推荐 `--checkpoint weights --backend transformers`。旧参数 weights/sam3.pt
    在 auto/transformers 下也会解析为同目录 HF bundle，不会把 `.pt` 交给 Transformers。
- HF bundle 至少包含非空 config.json、processor_config.json、model.safetensors，
    并需有处理器所需的本地 tokenizer 文件。所有加载均为 `local_files_only=True`，不联网下载。
    当前本地视频 bundle 的 detector/tracker 子配置由相应 Transformers 模型读取；
    若实际加载发现缺失或不匹配参数，明确失败，**不允许随机初始化参数继续预测**。
- 默认 `--device auto`，使用 `"cuda" if torch.cuda.is_available() else "cpu"`；
    `--device cpu` 将模型、图像及提示张量放在 CPU，使用 FP32、eval 和 inference_mode，
    不依赖 CUDA、官方 sam3 包或 GPU autocast。当前已安装 Torch 2.14.1+cpu、Transformers 5.21.0。
- `--backend official` 才需要[官方 SAM 3 仓库及依赖](https://github.com/facebookresearch/sam3#installation)
    和 sam3.pt。官方仓库的 CUDA 推荐不能套用于 Transformers CPU 路径。
- `python -m reference_anomaly.sam3_prediction --checkpoint weights --backend transformers --check`
    只检查本地文件/依赖和自动设备选择，不代表模型已加载、内存足够或推理成功。

本地 CPU 框推理：

```powershell
.venv\Scripts\python.exe -m reference_anomaly.sam3_prediction --result periodic_template_matcher/results/sd1_opencv --config periodic_template_matcher/config.sd1.labelme.opencv.json --checkpoint weights --backend transformers --device cpu --mode box
```

也可在 VS Code 中运行任务 **SAM3: local Transformers CPU inference**。
显式设为 `--device auto` 即按设备可用性选择；无须更换现有 CPU Torch。

### 提示与对照模式

各模式都需要 `--result` 指向已保存结果，并通过 `--config` 只提取 target / box；
也支持直接传 `--target` / `--boxes`。沿用显式 XML 坐标约定
`--xml-coordinates voc_1based_inclusive` 或 `zero_half_open`。

| 参数 | 意义 |
|---|---|
| `--mode box` | 每个检测框一个框提示，不读取异常分数和 confidence；参考 valid 仅用于覆盖记录 |
| `--mode anomaly_points` | 默认：框＋异常正负点，每个高响应组件独立提示，再 union 结果 |
| `--mode tracker_box` | 使用 Sam3TrackerModel 的纯框对照，不读取异常分数和 confidence |
| `--mode anomaly_guided` | 同一 Tracker 模型，联合异常核心正点、框内外近邻正常负点，再按参考证据筛选候选；详见下节 |
| `--context-fraction` | 裁剪框每侧相对原检测框宽/高的扩展比例，默认 0.1；例如 0.5 为每侧 50%，提示框保持原检测框 |
| `--text` | 可选文本，只能与 Transformers 的 box 模式结合；不由 GT 生成 |
| `--threshold` / `--mask-threshold` | Transformers 图像模式实例置信度/像素阈值，均默认 0.5；不能用当前 GT 调参 |
| `--prepare-only` | 只保存可检查的提示与输入，不构造模型，不输出 prediction mask |
| `--out` | 可选新输出目录，已有目录/源文件禁止覆盖 |

默认裁剪上下文每侧 10%、至少 16px，并截断到原图边界。`--context-fraction` 只扩大
模型看到的原图区域，不扩大检测框提示；提示坐标仅减去裁剪左上角。扩大上下文的重跑必须
以 `--out` 指定新目录，不能覆盖默认目录中的冻结结果。比例须为有限非负数，0 仍保留 16px
最小上下文。旧 `anomaly_points` 规则保持不变：框内 valid 且 confidence=255、
salience_score ≥3 的高响应组件按峰值排序最多取 16 个，每个最多 3 个分散内部正点。
负点来自**所有检测框外**、可靠且 salience_score ≤1 的区域，最多 8 个分散内部点。
参考未知位置不生成正负点，框内低热度位置不直接当负点。多个组件分开提示，
不会把其他异常组件作为负点。没有可靠参考/前景/背景时明确 abstain。

Transformers 入口沿用 `from_pretrained` → `processor(...).to(device)` → `model(**inputs)`：

- box 模式使用 **Sam3Model / Sam3Processor**，传入检测框和可选文本，
    用 `post_process_instance_segmentation` 恢复 crop 原尺寸。
    输出是**不同实例**，保留所有通过固定阈值的非空实例并取并集，不能当作备选 mask 只取最高分。
- anomaly_points 模式使用 **Sam3TrackerModel / Sam3TrackerProcessor**，缓存每个 ROI 的
    图像 embedding，然后传框及正负点。Sam3Model 本身不接受点，不能静默丢掉点提示。
    `post_process_masks` 恢复原尺寸；多个输出是同一提示的**备选 mask**，
    在满足全部正负点且非空的候选里按模型自身质量选最高者，否则 abstain。
    图像模式的两个阈值不用于 tracker；tracker 沿用处理器默认二值化阈值 0。
- 官方后端仍通过 `build_sam3_image_model(..., load_from_HF=False,
    enable_inst_interactivity=True)`、`Sam3Processor.set_image(PIL RGB)` 和 `predict_inst`，
    使用备选 mask 规则。该后端 `normalize_coords=True` 表示内部归一化像素坐标，不能反转。

所有提示都是 crop-local 像素坐标，由各处理器转换。每次保存全部候选及分数、筛选记录；
分数不是已测量的 GT IoU，禁止根据 GT 选择候选。
预测保留模型给出的原始边界：不与异常阈值图相交，不裁回检测框，不按参考 valid
截断，不默认取最大组件，不额外平滑或填洞；SAM 自身上游后处理除外。
无参考区域仍可能由 SAM 根据原图推断，额外记录 `predicted_without_reference_pixels`，
这表示**没有参考证据支撑**，并非该处已被正常参考验证。

### 输出与当前验证状态

Transformers 提示准备默认新建 anomaly/sam3_box_transformers_prepared 或
anomaly/sam3_anomaly_points_transformers_prepared（官方后端不带 _transformers 后缀），
包含 preparation manifest、原图、valid、点图、prompt preview 和每组件提示列表。
预览中黄色为检测框、浅蓝色为上下文框、绿色为正点、蓝色为负点；
叠加显示所有任务的点，但真正推理按每个组件独立调用。
manifest 的 `model_executed=false`，不产生 mask，独立评价器拒绝将预览当成预测。
本次实际样本已生成两种提示预览，高响应组件保留 16 组并明确提示截断。

Transformers 真实预测默认新建 anomaly/sam3_box_transformers 或
anomaly/sam3_anomaly_points_transformers（官方后端不带 _transformers 后缀）。
冻结导出包含 mask、轮廓层级、叠加图、全部候选 mask、提示和选择记录、
checkpoint、HF 配置/tokenizer、输入/响应的来源哈希及适配器代码哈希。
输出 manifest 的 kind 为 `sam3_prediction`；全部提示提前 abstain 时 model_executed=false。
先保存预测，之后才可用已有独立评价入口 `reference_anomaly.box_evaluation` 读取 GT；
`GT_labelme` 不作为 SAM 模块参数，预测阶段不解析其路径。

**2026-11-19 首次 CPU 尝试（历史）：**已执行上述本地 Transformers CPU 命令，模型加载失败于
Windows **页面文件/虚拟内存提交限制（1455）**，当时提交余量仅约 2–3 GB。
这不是“不支持 CPU”或“必须安装 CUDA”。需要释放足够内存提交量或调整系统虚拟内存后重试；
程序没有擅自终止其他进程、修改页面文件、驱动或 Torch 安装。

**2026-11-19 分支切换后续跑：两条真实 CPU 推理均成功。**在
`feat/supposed-to-be-good` 上，以原有固定提示/阈值加载本地权重，Sam3Model 和
Sam3TrackerModel 均完成加载、CPU FP32 推理和冻结导出，missing/mismatched keys 均为空。
本机另有 Windows WMI 查询阻塞；回归运行器及此次真实推理启动包装仅在各自进程内让
`platform._wmi_query` 抛出 OSError，触发 Python 标准 Windows 回退。不模拟模型、张量、
候选或分数，不修改生产模块或系统服务。普通 VS Code 任务不包含此临时包装。
首次无输出的推理任务已停止，未终止无关进程。

两份 prediction manifest 在读取 GT **之前**均已生成并记录 SHA256；独立评价器随后
校验全部冻结 artifact 哈希、验证外部原图像素身份，并新建独立评价目录。结果如下：

| 固定路径 | 预测像素 | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|---:|
| GraphCut（已有基线） | 36,540 | 59.61% | 48.34% | 36.42% | 53.39% |
| SAM3 框提示 | 22,742 | 17.69% | 8.93% | 6.31% | 11.86% |
| SAM3 框＋异常点（16 组） | 182,715 | 24.45% | 99.15% | 24.40% | 39.23% |

SAM3 框提示结果和评价分别位于 anomaly/sam3_box_transformers、
anomaly/sam3_box_transformers_evaluation；异常点结果分别位于
anomaly/sam3_anomaly_points_transformers、anomaly/sam3_anomaly_points_transformers_evaluation。
框提示仍漏掉大部分 GT；异常点方案扩大到正常纹理，误分严重。**当前 SAM3 方案并未超过
GraphCut**。框＋异常点还同时更换了 image/tracker 模型接口，因此这不是隔离异常提示作用的
严格消融，不能据此断言正常参考可删除或必然改善分割。当前图是已见开发样本，不能代表
未见测试集；没有据此重选提示、阈值或候选，也没有裁剪/后处理冻结结果。

合成测试覆盖提示规划、离线 CPU 参数、两类 HF 模型接口、embedding 缓存、图像多实例并集、
交互候选选择、GT 隔离及冻结契约；这些合成测试不加载权重，不能替代上述真实推理。
修复 HF 模拟配置覆盖业务配置的测试夹具后，相关 61 项测试通过；完整 Python 回归
**271 项通过，86.020 秒**（仅运行器使用上述 WMI 回退）。

**2026-11-19 按用户要求扩大裁剪：**固定使用 `--mode box --context-fraction 0.5`，
其余模型和阈值不变，没有异常点或自定义文本。裁剪从 618×561 扩为 **1028×935**，
原图 ROI 为 `[835,102,1863,1037]`。检测框仍是 `[1092,336,1606,803]`，传入处理器的
crop-local 提示仅平移为 `[257,234,771,701]`，不是把 ROI 当成提示框。
真实 CPU 推理另存 anomaly/sam3_box_transformers_context50，4 个通过原阈值的实例
合并为 119,937 像素，132 像素无参考覆盖，有 touches_roi 提示。从叠加图可见也选中了
框外重复结构；框提示不是输出边界限制，不能直接断言扩大裁剪后更准确。本轮没有再次读取
GT 或计算指标，两份历史预测 manifest 哈希保持不变。
相关测试 **66 项通过 / 0.591 秒**，完整回归 **276 项通过 / 83.399 秒**。

### 新增异常辅助 Tracker 策略（2026-11-19）

`--mode anomaly_guided` 与 `--mode tracker_box` 都使用 Sam3TrackerModel /
Sam3TrackerProcessor。原图 RGB、原始检测框、裁剪比例、模型权重和模型后处理均相同；
差异是异常点提示和候选证据筛选。旧 box/image 与 anomaly_points 路径和冻结结果不变。

辅助规则在真实推理前固定，完整记录见
[实验计划](../docs/superpowers/plans/2026-11-19-sam3-anomaly-guidance.md)：

- 仅 confidence=255 的参考可提供证据。分数 ≥3 的组件内部核心产生正点，最多
    16 个组件、每个最多 3 点，**合成一次联合提示**，不再独立分割后并集。
- 分数 ≤1 的可靠区域可以产生正常负点；距任意可靠高响应区域至少 3px，并限制在
    max(8px, ceil(原检测框短边×0.1)) 邻域。框内最多 8 点，所有检测框外最多 8 点。
    confidence=0/128、分数中间区域不作负点。两组负点池都缺失时 abstain。
- 候选必须满足全部点，覆盖至少 50% 的所选核心；在核心与可靠正常证据域内，正常占比
    不超过 50%。按“核心覆盖率－正常占比”排序，再按模型质量与稳定索引打破平局。
    未知区域中立；证据分不是 GT IoU。全部不合格就拒绝输出，不回退到质量最高的错误候选。
- **不裁切、不平滑模型掩膜**。冻结导出额外包含 guidance_evidence.npz、完整策略常量、
    证据代码哈希及每个候选的覆盖率、正常占比和拒绝原因。

本次真实同模型对照使用 `--context-fraction 0.5`、CPU FP32、无文本。原保存结果的
input.png 和 result.json 已变化，首次执行被原图一致性检查拒绝，未绕过或关闭检查。
从历史冻结输入副本及哈希仍匹配的异常分数/置信度/valid 建立
results/sd1_sam3_guidance_replay 独立重放快照，明确标记为历史证据重放而非新 matcher
运行；原始目标图与冻结输入的像素相等，来源与哈希链已记录。现有 matcher 文件未覆盖。
两组真实预测均使用这一份已校验快照，输出另存于原 anomaly 目录：

| 路径 | 真实推理 | 输出 |
|---|---|---|
| sam3_tracker_box_transformers_context50 | 成功，3 个备选 | 按模型质量选第 3 个，104,838 像素 |
| sam3_anomaly_guided_transformers_context50 | 成功，48 正点＋16 负点，3 个备选 | 全部拒绝，status=abstained，输出空掩膜 |

辅助候选分别漏掉 42、11、19 个正点；后两个还覆盖了 1 个负点。证据域内正常占比约
96.45%、98.86%、99.01%，全部超过预设门槛。**这是未取得可信分割，不是判定图中无缺陷**。
原始候选全部保留；另有 sam3_anomaly_guided_context50_diagnostics/rejected_candidates.png
诊断图，不属于最终预测。没有据此放宽门槛、重选点、修改掩膜或读取 GT 计算新指标。
同模型/权重/输入/裁剪/原框及全部冻结 artifact 哈希已校验，历史三份 manifest 未变。

相关 **82 项测试通过 / 0.758 秒**；完整 Python 回归 **292 项通过 / 89.403 秒**。
真实模型和回归继续仅使用进程级 WMI 回退，不修改模型计算、Torch 或系统服务。
