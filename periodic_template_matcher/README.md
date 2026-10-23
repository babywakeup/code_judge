# 周期结构多模板配准与分割标注映射

本代码比较两张或多张模板与同一目标图的匹配程度，给出0—100的match score、排序和变换矩阵，并将选中模板的类别掩码映射到目标图。

这是可运行的传统视觉基线，采用结构归一化、线性相关多候选、仿射ECC精配准和统一质量评分。支持重复模板去重、PNG / LabelMe 分割标注、离线中文交互报告及六宫格总览。它尚未实现局部非刚性变形、设备坐标标定或周期标签自动外推。现场精度需要用真实标注验证。

## 本次 SD-1 图片：直接看结果

- [交互报告](results/sd1_1792571365518/report.html)：模板排序、拖动分割线、透明度切换、按住查看目标、边缘对齐、棋盘格、差异热图、有效覆盖、标注映射示意。图片内嵌，可离线查看。
- [六宫格总览](results/sd1_1792571365518/overview.jpg)。
- [结构化结果](results/sd1_1792571365518/result.json)。
- [本次配置](config.sd1.json)。

上述历史报告生成时，两条模板输入完全相同并被合并。当前 [配置](config.sd1.json) 已改为两张不同模板（尾号 1258237、1343428），目标为尾号 1365518，均位于 SD-1/10-21 数据目录。重构前调试时两张模板得分分别约 73.74、19.81；历史报告不会自动更新，应重新运行生成新报告。文件内容完全一致时，配置流程仍自动合并重复模板。

实测匹配分 **73.74 / 100**，目标有效 ROI 覆盖 **83.80%**，结构 NCC **0.8496**，全局仿射配准通过。覆盖率针对排除底部 8% 设备文字后的工作 ROI，**不是全图覆盖，也不是分割准确率**。本例检测到周期结构，绝对周期编号未验证。

当前没有真实分割标注，故 **不生成正式 `mapped_labels.png` / `label_valid.png`**。配置中的 `demo_annotations: true` 仅绘制任意圆形、矩形、多边形来演示坐标映射；`demo_*` 文件和报告均明确标记“非真实标注”，不得用于训练或生产。

## 安装及运行

建议Python 3.10或以上。OpenCV使用headless版，无GUI依赖。

推荐在**工作区根目录（本包的父目录）**运行：

```bash
python -m pip install -r periodic_template_matcher/requirements.txt
python -m periodic_template_matcher --config periodic_template_matcher/config.sd1.json --out periodic_template_matcher/results/new_run
python -m unittest discover -s tests -t . -p "test_*.py" -v
python -m examples.matcher_demo --out periodic_template_matcher/results/api_demo
```

旧版单文件启动、旧配置类和兼容导出已经移除，不再维护多套调用方式。只使用上面的模块启动命令或下方的 Matcher API。请从工作区根目录运行，而不是进入包目录。输出路径相对于运行目录；配置内的输入路径相对于配置文件，不受运行目录影响。

所有输入路径相对于JSON文件所在目录，也可以使用绝对路径。Windows中文路径可读取。

当前 [SD-1 配置](config.sd1.json) 包含两张模板，可按实际输入替换路径。若只检查配准，无需设置 `labels` 和 `annotation_valid`。底部文字排除比例默认为 0，本例设为 0.08，须按实际图像调整。

每次运行建议使用新的输出目录，便于保存不同参数结果。重用输出目录时，仅清理程序拥有的已知结果文件，避免上次的真实标签被误认成本次结果；其他文件不会删除。禁止输出覆盖配置中的输入文件。

文件流程可通过 `from periodic_template_matcher import run_config` 导入，再调用 `run_config(config_path, output_dir)`，返回与 `result.json` 相同的字典。

## 模块结构与阅读顺序

包顶层只有 **4 个 Python 文件**，只呈现主要接口。算法、评估和文件流程集中放入私有实现目录；测试、示例放在工作区根目录，不随业务接口混排。所有代码使用正常的包导入，不再修改 `sys.path`。

| 模块 | 单一职责 |
|---|---|
| [包入口](./__init__.py) | 明确列出公开 API；不执行 CLI 或读写配置 |
| [配置](config.py) | 唯一的 `MatcherConfig` 与参数校验 |
| [Matcher](matcher.py) | `match` / `evaluate` / `visualize` 的流程编排 |
| [模块启动](./__main__.py) | 唯一的命令行启动入口及参数解析 |
| [内部实现](_internal/__init__.py) | 配准、评估、结果类型、标注、报表及 JSON 工作流；不是公开 API |
| [图像工具](utils/images.py) | 图像 IO、灰度转换与照片校验 |
| [掩码工具](utils/masks.py) | 标签格式转换、有效区、标签坐标映射与叠加 |

公开入口只导出 `MatcherConfig`、`PeriodicTemplateMatcher`、`MatchResult`、`EvaluationResult`、`run_config`；图像读写从 `periodic_template_matcher.utils` 导入。日常调用只需看配置和 Matcher，不必了解内部文件。内部模块保留职责边界，不为减少顶层文件数而拼成一个大文件。

建议阅读顺序：**配置 → Matcher → 返回结果**。如需修改算法，再进入内部实现。基础工具不引用 Matcher，可视化只依赖结果与工具，评估不依赖配准，不存在循环导入。

VS Code 调试已改为模块方式启动，工作目录固定为包的父目录。流程断点设在 `match()` / `evaluate()`，算法断点设在配准内核的候选评分与 ECC 部分，不再在旧兼容入口查找实现。

### 测试与示例放在哪里

推荐将自动测试保留为多个 `test_*.py`，统一放在包外的 **tests/**，而不是塞进生产模块或单个超大测试文件。当前使用标准库 `unittest`，不额外引入测试依赖：

- [配准及文件流程测试](../tests/test_matcher.py)：配准、LabelMe、去重、输出与周期保护。
- [Matcher 及指标测试](../tests/test_matcher_api.py)：类接口、IoU/Dice、未知区、真值和可视化。
- [包结构测试](../tests/test_package.py)：统一入口、CLI 及顶层文件精简约束。
- [共享测试数据](../tests/helpers.py)：独立造图工具；测试文件不互相导入。

可运行演示放在包外的 [examples/matcher_demo.py](../examples/matcher_demo.py)。演示不是自动化测试。VS Code 左侧“测试”视图已配置为发现 tests/，可以直接运行或调试单个用例。

## 输入标注格式

- `labels`：与模板原图同宽高的单通道8位或16位PNG。每个像素为整数类别ID，范围0—65535。不要使用RGB彩色可视化图作为标签。
- `annotation_valid`：与模板同尺寸的单通道掩码，非零为已完成标注的像素。如果只有局部标注，应提供此文件，避免把未标注区域误作背景。
- `valid_mask`：可选的模板匹配有效区，用于排除文字、反光等。`target_valid_mask`是目标图的对应配置，置于JSON顶层。
- `labels`中的0可以是真实背景类别，只有同时查看`label_valid.png`才能区分未标注区域。

### 后续接入真实 segmentation 标注

方式一：在对应模板条目增加 `labels`，指向与模板原图同尺寸的类别 PNG；局部标注同时设置 `annotation_valid`。路径相对配置文件。

方式二：增加 `labelme` 和显式 `class_map`：

```json
{
    "name": "SD1_template_A",
    "image": "../source/data/SD-1/10-21/Tlink-Image-1792571258237.jpeg",
    "labelme": "annotations/template_A.json",
    "class_map": {"wire": 1, "via": 2, "electrode": 3}
}
```

示例中的标注文件路径是未来需准备的输入，当前不存在。支持 LabelMe `polygon`、`rectangle`、`circle`；重叠时后绘制的形状覆盖先绘制的形状。不支持 line / point / COCO，遇到不支持的类型或未知类别会报错，不会静默丢弃。

`labels` 与 `labelme` 二选一。LabelMe 的未绘制像素默认视为已知背景 0；若标注未覆盖全图，必须提供 `annotation_valid`，不要把未画区域都当成背景。不同模板应使用统一的 `class_map` 语义。

接入真实标注后原样重新运行：仅在匹配通过时输出正式掩码、有效范围及叠加图。使用最近邻采样，保留 16 位类别 ID。所选模板有真实标注时不再生成示意标注。

内容完全一致的模板图片会合并（包括不同路径下的相同文件）；如果重复图片配置了互相冲突的标注或有效区，则报错并要求先消除冲突。底层 `select_templates` API 仍按给定候选逐项评分，不做文件级去重；去重属于配置执行流程。

## 多模板match score

全部模板在同一个目标ROI、同一个工作分辨率、同一组参数下评价：

```text
score = 100 × (0.55 × structure_ncc
             + 0.30 × edge_alignment
             + 0.15 × local_consistency)
             × sqrt(target_coverage)
```

- `structure_ncc`：局部亮度归一化后，在有效重叠区域的结构相关性。
- `edge_alignment`：双向边缘距离的指数相似度，两边都计入，降低只对齐少量边缘导致的误判。
- `local_consistency`：4×4网格中有效区域相关性的25分位数，检查不同部位是否同时对齐。
- `target_coverage`：成功映射到有效模板区域的目标ROI比例，降低小面积巧合对齐的分数。

工程默认阈值：score至少60、目标及源粗搜索覆盖至少45%、两个最佳模板分差至少3分。这些数值没有经过现场校准，match score不是概率或准确率。图像尺度、边缘宽度和光照会影响分数，需要用多产品已知正确/错误样本调参。

模板分差不足时仍报告最高分模板`best_template_index`，但`accepted=false`，不自动导出映射标签。所有模板均不达标时，`best_template_index=null`。

应选取可比的视野和标注完整度。覆盖率惩罚会偏好能覆盖更多目标区域的模板；如果只关心缺陷附近，应通过相同`target_valid_mask`定义评估ROI，而不是比较不相关区域的全图分数。

## 周期歧义与模式

`label_mode`有两种：

- `semantic`：每个重复单元共享同一类别含义，允许视觉配准通过后映射已有标注。此模式并未证明绝对周期序号。
- `instance`：需要唯一器件、讯号或周期编号。周期结构缺少外部锚点时不自动通过，也不会导出标签。

`periodic_detected`结合轴向自相关和近分候选做诊断，不是对任意倾斜晶格的完整证明。`lattice_candidates`可能包含谐波，不能直接当作唯一基础周期。有限候选搜索可能遗漏其他周期匹配，因此高候选分差也不是唯一位置的证明。

本版本没有接入设备坐标或定位标记来消除绝对周期歧义。`absolute_cell_verified`在非周期且候选分离时仅代表图像证据；周期图不声明绝对身份。用于唯一编号的生产流程还需要外部锚点及标定。

## 输出文件

- `report.html`：自包含中文交互报告；下载链接要求结果文件仍保存在同目录。
- `overview.jpg`：六宫格总览，可直接在 VS Code 中打开。
- `result.json`：各模板score、排序、第二模板分差、各候选、诊断及变换矩阵。
- `matrix_template_to_target.npy`：3×3矩阵，明确从模板原始像素坐标映射到目标原始像素坐标。
- `best_alignment.jpg`：最高分模板与目标的半透明叠加预览，仅供检查，`accepted=false`时也可能存在。
- `geometric_valid.png`：图像的有效几何重叠范围。
- `mapped_labels.png`：接受匹配且该模板有标注时输出的16位类别ID图。
- `label_valid.png`：标注在目标图上的有效范围，未覆盖处不能解释为背景。
- `label_overlay.jpg`：标签叠加预览。
- `demo_source_labels.png` / `demo_mapped_labels.png` / `demo_label_valid.png` / `demo_label_overlay.jpg`：仅在所选模板无标注、存在配准矩阵且显式开启 `demo_annotations` 时生成；**包括匹配拒绝时也可作为诊断示意，绝非正式标签**。

`result.json` 中 `labels_exported`、`annotation_status`、`demo_labels_exported` 区分真实标签和示意。`template_aliases` 保留原始模板条目与去重候选的映射；`best_template_index` 是去重候选索引。

报告在匹配被拒绝时仍显示最高分候选及拒绝状态，用于排查，不能将可视化预览误当成已接受结果。差异热图只是局部归一化后的图像差异，不是缺陷分割。底部排除区与没有源模板支持的区域均保持未知，不自动补齐。

本版本只迁移模板已有标注，不会把标注自动铺满未覆盖的周期。图一、图二裁切不同会留下未覆盖区域，这是正确的未知区域处理。

`edge_distance_p95_work_px`是图像边缘间距离的诊断值，单位为工作分辨率像素，不是人工标注误差，也不能代替对Short所需精度的验收。

## 类式 Python API（推荐）

`PeriodicTemplateMatcher` 是统一匹配入口；`MatcherConfig` 初始化配准、标签模式和评估参数。类接口统一负责匹配、标签迁移、评估和检查图；JSON 文件流程也调用同一个 Matcher。

### 输入约定

- `templates`：非空图像数组列表，每张图为 uint8 灰度/BGR/BGRA，传路径时先用 `read_image` 读取。`input` 是**一张**目标图，而非 batch。
- `masks`：与模板一一对应的列表。每个条目为同尺寸 H×W 整数类别图（0—65535，0 为真实背景），或按类排列的二值 mask 列表，或 `None`（没有标注）。
- 二值列表第 k 个元素对应类别 k+1，支持 bool、0/1、0/255；不同类别不能重叠。**类别图中的 255 就是类别 255，不会自动变成类别 1。** 例如单类二值标注写作 `masks=[[binary_A], [binary_B]]`，真值写作 `input_masks=[binary_gt]`；高位类别 ID 用类别图传入。
- 空二值列表 `[]` 表示整张图已知背景；没有标注必须传 `None`，不能把空列表当作未标注。
- `input_masks`：目标真值，可以是 H×W 类别图，或对应相同类顺序的二值 mask 列表。**不是多个互相替代的标注版本，也不是 batch**。单张类别图直接传数组，勿包额外的列表。
- 局部标注通过 `annotation_valids`（模板侧）和 `input_valid`（真值侧）显式指定。图像 ROI 用独立的 `template_valids`、`target_valid`。非零为有效；真值有效区不会用于配准，避免真值泄漏。
- 数组 API 不做文件级去重；重复候选仍会因分差不足被拒绝。文件去重属于 JSON 工作流。

### 初始化、匹配、可视化与评估

```python
import cv2
from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher.utils import read_image

# 以下路径需替换为实际图像和真实标注；无需传入路径到 Matcher 内部。
templates = [read_image("template_A.jpg"), read_image("template_B.jpg")]
masks = [read_image("labels_A.png", cv2.IMREAD_UNCHANGED),
         read_image("labels_B.png", cv2.IMREAD_UNCHANGED)]
target = read_image("target.jpg")
config = MatcherConfig(
    bottom_ignore_fraction=0.08, refine_candidates=8,
    label_mode="semantic", include_background=False,
)
matcher = PeriodicTemplateMatcher(config)
result = matcher.match(templates, masks, target)
print(result.accepted, result.best_template_index, result.match_score)
print(result.matrix_template_to_target)
if result.mapped_labels is not None:
    predicted = result.mapped_labels  # H×W uint16 类别图
    known = result.label_valid       # H×W bool；False 是未知，不是背景
canvas = matcher.visualize(result, "match_check.jpg")  # BGR 数组，无需 GUI

truth = read_image("target_labels.png", cv2.IMREAD_UNCHANGED)
evaluated = matcher.evaluate(templates, masks, target, truth)
print(evaluated.score, evaluated.metrics)
matcher.visualize(evaluated, "evaluation_check.jpg")
```

`match()` 返回 `MatchResult`：`registration` 保存各候选、排序、状态和阈值；`mapped_labels` / `label_valid` 仅在匹配通过且最佳模板有标注时存在。`warped_template` / `geometric_valid` 可以在拒绝时存在，但仅供诊断。构造时复制配置；不修改调用方的图像或标签，不自动写文件。

`evaluate()` 每次内部调用一次 `match()`；如果已经调用过 `match()` 再调用 `evaluate()` 会重新配准。结果为 `EvaluationResult`，包含 `match`、`metrics`、`ground_truth`、`ground_truth_valid`。需要同时评分与可视化时，直接调用一次 `evaluate()`，再使用其中的 `match` 即可。

可视化统一调用 `matcher.visualize(result, output_path=None)`，不再增加独立的公开可视化入口。六宫格含模板、输入、配准叠加、预测；有真值时增加真值与错误图，无真值时展示几何及标签覆盖。错误图绿色=正确、红色=错类/漏分、橙色=背景误报、品红=未覆盖真实前景，暗色=未知。图片可直接在 VS Code 中打开，函数不会启动 GUI。

### 评估指标与边界情况

- `score = 100 × mean_iou`：0—100，**真实分割评估分**，不是 `match_score`。后者只衡量图像配准相似度。
- 每类 `IoU = TP/(TP+FP+FN)`，`Dice = 2TP/(2TP+FP+FN)`；返回每类 TP/FP/FN、IoU/Dice、`mean_iou`、`mean_dice`、`pixel_accuracy`。
- 默认 `include_background=False`，宏平均不计背景 0，像素准确率仍包含背景。类别集合为真值与有效预测中出现的类的并集，预测中多出来的类会被处罚；双方都没有的类不计分。
- 主指标在整个**真值有效 ROI** 上计算。预测未覆盖的真实前景计 FN；未覆盖位置是拒答，不当作类别 0，像素准确率也不把它计为正确。
- 同时返回 `prediction_coverage`、`covered_pixels`、`uncovered_pixels` 和 `overlap_only`（仅交集区域的指标）。主 score 不等于 overlap-only score，不可忽略覆盖率。
- `input_valid=False` 的真值像素不参与指标。目标匹配 ROI 和底部排除区也不参与指标。
- 匹配被拒绝时 `status=match_rejected`，无模板标注时 `status=no_template_annotations`，没有有效真值时 `status=no_valid_ground_truth`；这些情况 `score=None`，不伪造为 0 或满分。
- 默认不计背景且预测/真值全为背景时，前景 mIoU/Dice/score 为 `None`，像素准确率仍可为 1；如需背景分，显式开启 `include_background=True`。
- 周期图在 `instance` 模式下仍会被身份保护拒绝，不能绕过匹配门限计算正式预测分。所有模式均不自动外推未覆盖周期。

[可运行示例](../examples/matcher_demo.py) 自动生成有已知变换和两类标注的**合成**图，调用评估并保存检查图、预测标签、有效范围及指标 JSON；无需额外输入。它验证 API 流程，不代表产线精度。

## 已进行的验证

原有 8 项自动化测试包含平移和小旋转/缩放恢复、模板排序、相同模板分差拒绝、空白拒绝、16位类别 ID 及映射方向、有效背景与未知区区别、重复文件合并与冲突拒绝、LabelMe 读取与不支持形状拒绝、真实标注/示意隔离、重跑后旧输出清理、HTML 名称转义，以及周期身份拒绝时的标签导出保护。新增 [类接口与评估测试](../tests/test_matcher_api.py) 覆盖真实算法合成配准、二值列表、掩码校验、配置隔离、真值泄漏保护、手算 IoU/Dice、覆盖率惩罚、假阳性类、空真值、背景和拒绝状态，以及灰度/BGRA可视化。生产周期身份保护的 IO 测试使用受控诊断结果，不是现实设备绝对身份验证。

`example_real_pair`包含用户提供图一→图二的历史实测结果与叠加图：score约75.02，有效覆盖约82.5%，识别到周期结构，绝对周期编号未验证。当前已有第二张模板，但尚无真实分割标注，因此真实标注边界精度仍未验收。合成例子的标签仅用于演示软件流程，不是产线标注。

模块职责见上方结构表。[包结构测试](../tests/test_package.py) 额外验证公开类型身份一致、无关顶层同名模块不干扰包导入、模块命令行启动、旧模块已删除及参数转发。无需额外 GUI 或网页框架依赖。
