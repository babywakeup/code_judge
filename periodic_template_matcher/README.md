# 周期结构多模板配准与分割标注映射

## 匹配后端选择

在 JSON 的 `options` 中设置 `"backend": "fft"` 或 `"backend": "opencv"`。
省略时默认 `fft`，保留原有行为；Python API 对应 `MatcherConfig(backend='opencv')`。

- `fft`：原有 FFT 线性相关位移搜索。
- `opencv`：使用 `cv2.matchTemplate(..., cv2.TM_CCORR)` 计算相关、有效重叠和能量，
  再按真实有效交集归一化。补边支持同尺寸图像的正负位移和部分重叠，补边不计入有效面积。

两个后端共用角度/缩放枚举、多个候选、ECC 仿射精配准、最终评分、接受条件及标签迁移。
OpenCV 后端替换粗搜索；两个后端都可启用下述周期补齐，计算耗时依图像和搜索范围而异。
报告 `result.json` 的 `config.backend` 记录实际配置。

```powershell
# OpenCV 后端：同一批 SD-1 图像，输出到独立目录
.\.venv\Scripts\python.exe -m periodic_template_matcher --config periodic_template_matcher/config.sd1.labelme.opencv.json --out periodic_template_matcher/results/sd1_labelme_defect_opencv
```

本代码比较两张或多张模板与同一目标图的匹配程度，给出0—100的match score、排序和变换矩阵，并将选中模板的类别掩码映射到目标图。

这是可运行的传统视觉基线，采用结构归一化、线性相关多候选、仿射 ECC 精配准和统一质量评分。支持重复模板去重、PNG / LabelMe 分割标注、周期标签补齐和独立的预测查看 app。推理和可视化分为两个阶段：先保存原始预测数据，再打开 app 观察。尚未实现局部非刚性变形或设备坐标标定。

## 匹配前设置有效高度

模板和 input 的底部文字可以通过原图像素高度排除：

```json
{
  "target": "input.jpeg",
  "target_valid_height": 1891,
  "templates": [
    {"image": "template_A.jpeg", "valid_height": 1891},
    {"image": "template_B.jpeg", "valid_height": 1850}
  ],
  "options": {"valid_height": 1891}
}
```

H=1891 表示有效行是 `[0, 1891)`，第 1891 行及之后不参与匹配。
优先级为：每张图的显式高度 > `options.valid_height` 公共默认高度 >
`bottom_ignore_fraction`。显式高度覆盖比例设置，不叠加裁剪；所有情况仍与已有图像 mask 取交集。
高度必须为正整数且不超过对应图像高度，有效像素不足时会明确报错。

实现集中在 `utils/regions.py` 的 `ValidRegion` 类中：

- `resolved_height(image)`：解析有效 H。
- `build_mask(image, mask=None)`：构造原图大小的 0/1 mask，与额外 mask 取交集。
- `resize_for_matching(gray, mask, size)`：仅用有效像素加权缩小，避免文字在缩放边界混入。
- `crop(image)`：可选的外部裁切便利方法；正式 matcher 保留原图画布和坐标。
- `describe(image, mask)`：生成可检查的高度和有效像素统计。

Matcher 的 `_prepare_regions()` 在提取特征前统一解析图像 ROI，之后复用于配准、标签迁移、
周期补齐及评价。标注有效范围 `annotation_valid` 与图像有效范围分别处理。
输出图像和 mask 保持原始 input 尺寸；无效底部的 `label_valid` 始终为 0。
`result.json.valid_regions` 和 HTML 报告会记录每张图的最终有效 H。

```python
from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher.utils import ValidRegion

region = ValidRegion(valid_height=1891)
valid_mask = region.build_mask(image)
matcher = PeriodicTemplateMatcher(MatcherConfig(valid_height=1891))
result = matcher.match(
    templates, labels, input_image,
    template_valid_heights=[1891, 1850],
    target_valid_height=1891,
)
# evaluate() 同样支持 template_valid_heights 和 target_valid_height。
```

## 周期补齐与完整 ROI 预测

在 `options.periodic` 中设置 `"periodic_completion": true`，匹配接受且有语义标注后执行补齐。
SD-1 的两份 LabelMe 配置已经启用；通用 API 默认关闭，以保留原有行为。

周期来源优先级：`period_vectors` 显式配置、`periodic_anchor_class`（默认 1）的重复
连通区域、input 图像自相关。标注锚点自动估计适用于接近水平/垂直的重复排列；倾斜晶格可显式
提供一至两个模板原图坐标向量，如 `"period_vectors": [[875, 1], [-2, 874]]`。
显式向量需要独立且非零。算法不读取目标 LabelMe 真值来指导预测。

先保留直接配准，再生成整数周期副本，只处理有机会提供未知标签的候选。每个候选进行 ECC
校正和图像验证，校正幅度受周期约束；默认要求结构 NCC >= 0.75、边缘评分 >= 0.6。
还会对每个待新增连通区域及其有限邻域单独验证，避免内部大面积重合掩盖边缘不匹配。
通过后，以最近邻迁移类别和 annotation_valid，填充背景与前景，保留类别 ID 和已有非零前景标签。
若同一个周期前景连通区域横跨未知区与直接配准区，会将其在背景 0 上的部分一并补齐，避免物体被配准边界截断；完全位于直接区域内的其他物体不会因此扩张。
`periodic_max_steps` 默认 2（范围 1..5），限定每个周期方向的复制步数，避免无界搜索。

API 的 `mapped_labels` / `label_valid` 为最终预测；`geometric_valid` / `warped_template`
仍是直接配准。新增 `completed_template`、`completed_valid`、`periodic_valid`、
`label_periodic_valid` 分别表示补齐图像、补齐几何范围、图像补齐区和标签补齐区。
未执行周期补齐时这些新增字段为 None。仅在 semantic 模式补齐，不复制唯一实例编号。

最终类别预测保存在 `mapped_labels.png`，其中已包含通过验证的周期补齐。`label_periodic_valid.png` 单独记录由周期补齐的标签像素；app 可切换“全部预测 / 直接配准 / 周期补全”观察。

`result.json.periodic_completion` 记录来源、周期向量、每个副本的变换和评分、
`label_coverage_roi`、`remaining_unknown_pixels` 以及 complete/incomplete/no_period 状态。
100% 覆盖表示 ROI 内每个像素都有类别预测（包括背景），不表示前景覆盖整图或 100% 准确率。
无可靠周期、验证失败或源标注不完整时保持未知并报告原因。局部标注必须配置 annotation_valid，
否则 LabelMe 未画区域按既有约定视为已知背景。底部排除区域始终不参与补齐。

## 两阶段运行与查看

在项目根目录运行以下两个命令，或直接运行 `bash run.sh`：

```bash
# Stage 1：匹配、周期补齐并保存 input 和预测 mask
python -m periodic_template_matcher --config periodic_template_matcher/config.sd1.labelme.opencv.json --out periodic_template_matcher/results/sd1_opencv

# Stage 2：读取保存的数据，在独立桌面 app 中观察
python -m periodic_template_matcher.app --result periodic_template_matcher/results/sd1_opencv
```

app 支持原图、类别 mask、透明叠加、左右同步对照、类别筛选、周期来源筛选、滚轮缩放、拖动平移和像素类别查询。打开或重新读取结果不会再次匹配。完整说明见 [app/README.md](app/README.md)。

旧的 `visualize()` 和 HTML 报告已移除。无标注或匹配被拒绝时保存原图与状态，app 明确显示没有预测；`demo_annotations` 旧字段不再生成示意 mask。

## 安装及运行

建议 Python 3.10 或以上。第一阶段使用 headless OpenCV；第二阶段桌面 app 使用 Tkinter，Windows 的标准 Python 通常自带，无需安装其他 pip 包。

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
| [Matcher](matcher.py) | `match` / `evaluate` / `save_result` 的流程编排 |
| [结果查看 app](app/README.md) | 独立读取保存的 input、mask 和有效区域，交互观察 |
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

- `input.png`：无损保存的原始分辨率 input，不依赖原始输入路径。
- `result.json`：各模板分数、排名、诊断、预测状态，以及 `artifacts` 文件清单。
- `matrix_template_to_target.npy`：3×3矩阵，明确从模板原始像素坐标映射到目标原始像素坐标。
- `geometric_valid.png`：图像的有效几何重叠范围。
- `mapped_labels.png`：接受匹配且该模板有标注时输出的16位类别ID图。
- `label_valid.png`：标注在目标图上的有效范围，未覆盖处不能解释为背景。
- `label_periodic_valid.png`：可选，标记周期补齐产生的有效标签。
- `ground_truth.png` / `ground_truth_valid.png`：保存 `EvaluationResult` 时附带真实标签和评估区域；指标保存在 `result.json.evaluation`。

`result.json.prediction_status` 为 `available`、`match_rejected` 或 `no_template_annotations`。后两种情况不生成预测文件，也会清理同目录上次的旧预测。`template_aliases` 保留去重映射；`best_template_index` 是去重候选索引。

app 在匹配拒绝时显示原图和拒绝状态。底部排除区保持未知；其他未覆盖区域仅在启用周期补齐且验证通过时填充。紫色斜纹表示未知区域，不是背景类别 0。

默认只迁移模板已有标注；启用周期补齐后，可使用验证通过的周期副本补齐裁切差异造成的未知区域。

`edge_distance_p95_work_px`是图像边缘间距离的诊断值，单位为工作分辨率像素，不是人工标注误差，也不能代替对Short所需精度的验收。

## 类式 Python API（推荐）

`PeriodicTemplateMatcher` 是统一匹配入口；`MatcherConfig` 初始化配准、标签模式和评估参数。类接口负责匹配、标签迁移、评估和结果保存；JSON 文件流程也调用同一个 Matcher。观察效果使用独立的第二阶段 app。

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
manifest = matcher.save_result(result, "periodic_template_matcher/results/my_match")
print(manifest)  # result.json 的绝对路径

truth = read_image("target_labels.png", cv2.IMREAD_UNCHANGED)
evaluated = matcher.evaluate(templates, masks, target, truth)
print(evaluated.score, evaluated.metrics)
matcher.save_result(evaluated, "periodic_template_matcher/results/my_evaluation")
```

`match()` 返回 `MatchResult`：`registration` 保存各候选、排序、状态和阈值；`mapped_labels` / `label_valid` 仅在匹配通过且最佳模板有标注时存在。`warped_template` / `geometric_valid` 可以在拒绝时存在，但仅供诊断。构造时复制配置；不修改调用方的图像或标签，不自动写文件。

`evaluate()` 每次内部调用一次 `match()`；如果已经调用过 `match()` 再调用 `evaluate()` 会重新配准。结果为 `EvaluationResult`，包含 `match`、`metrics`、`ground_truth`、`ground_truth_valid`。需要同时评分与可视化时，直接调用一次 `evaluate()`，再使用其中的 `match` 即可。

第一阶段调用 `matcher.save_result(result, output_dir)` 保存数据并返回 manifest 路径，不创建窗口。第二阶段运行 `python -m periodic_template_matcher.app --result OUTPUT_DIR` 观察 input 与预测 mask；也可只启动 app，然后点击“打开结果目录”。

### 评估指标与边界情况

- `score = 100 × mean_iou`：0—100，**真实分割评估分**，不是 `match_score`。后者只衡量图像配准相似度。
- 每类 `IoU = TP/(TP+FP+FN)`，`Dice = 2TP/(2TP+FP+FN)`；返回每类 TP/FP/FN、IoU/Dice、`mean_iou`、`mean_dice`、`pixel_accuracy`。
- 默认 `include_background=False`，宏平均不计背景 0，像素准确率仍包含背景。类别集合为真值与有效预测中出现的类的并集，预测中多出来的类会被处罚；双方都没有的类不计分。
- 主指标在整个**真值有效 ROI** 上计算。预测未覆盖的真实前景计 FN；未覆盖位置是拒答，不当作类别 0，像素准确率也不把它计为正确。
- 同时返回 `prediction_coverage`、`covered_pixels`、`uncovered_pixels` 和 `overlap_only`（仅交集区域的指标）。主 score 不等于 overlap-only score，不可忽略覆盖率。
- `input_valid=False` 的真值像素不参与指标。目标匹配 ROI 和底部排除区也不参与指标。
- 匹配被拒绝时 `status=match_rejected`，无模板标注时 `status=no_template_annotations`，没有有效真值时 `status=no_valid_ground_truth`；这些情况 `score=None`，不伪造为 0 或满分。
- 默认不计背景且预测/真值全为背景时，前景 mIoU/Dice/score 为 `None`，像素准确率仍可为 1；如需背景分，显式开启 `include_background=True`。
- 周期图在 `instance` 模式下仍会被身份保护拒绝，不能绕过匹配门限计算正式预测分。周期补齐仅用于通过匹配的 semantic 模式。

[可运行示例](../examples/matcher_demo.py) 自动生成有已知变换和两类标注的**合成**图，调用 `evaluate()` 与 `save_result()` 保存 input、预测标签、有效范围及指标 JSON，并打印 app 打开命令；无需额外输入。它验证 API 流程，不代表产线精度。

## 已进行的验证

原有 8 项自动化测试包含平移和小旋转/缩放恢复、模板排序、相同模板分差拒绝、空白拒绝、16位类别 ID 及映射方向、有效背景与未知区区别、重复文件合并与冲突拒绝、LabelMe 读取与不支持形状拒绝、真实标注/示意隔离、重跑后旧输出清理、HTML 名称转义，以及周期身份拒绝时的标签导出保护。新增 [类接口与评估测试](../tests/test_matcher_api.py) 覆盖真实算法合成配准、二值列表、掩码校验、配置隔离、真值泄漏保护、手算 IoU/Dice、覆盖率惩罚、假阳性类、空真值、背景和拒绝状态，以及灰度/BGRA可视化。生产周期身份保护的 IO 测试使用受控诊断结果，不是现实设备绝对身份验证。

`example_real_pair`包含用户提供图一→图二的历史实测结果与叠加图：score约75.02，有效覆盖约82.5%，识别到周期结构，绝对周期编号未验证。当前已有第二张模板，但尚无真实分割标注，因此真实标注边界精度仍未验收。合成例子的标签仅用于演示软件流程，不是产线标注。

模块职责见上方结构表。[包结构测试](../tests/test_package.py) 额外验证公开类型身份一致、无关顶层同名模块不干扰包导入、模块命令行启动、旧模块已删除及参数转发。无需额外 GUI 或网页框架依赖。

## 匹配分辨率

在 JSON 的 `options` 中设置 `"scale": 0.5`，即可将所有模板和输入图像的宽高按同一倍率缩小后匹配，FFT、OpenCV 以及周期补全均使用此设置。

- `"scale": null`（默认或省略）：按 `max_side` 自动计算倍率，最长边不超过该值，不放大。
- `"scale": 1.0`：使用原始分辨率。
- `"scale": 0.5`：宽高各为原图的一半；有效范围为 `0 < scale <= 1`。

显式设置 `scale` 时覆盖 `max_side`。`scales` 仍用于搜索模板相对输入的尺寸变化，与此参数独立。缩小可能损失细节、影响匹配分数；最终变换矩阵、标签和输出坐标仍对应原始图像。若缩放后任一图像的宽或高小于 32 像素，匹配会报错。

## 配置分组

`options` 按处理阶段组织，三个 SD-1 配置均已采用此结构：

```json
"options": {
  "step1": {"scale": null, "max_side": 800, "bottom_ignore_fraction": 0.08},
  "step2": {"backend": "opencv", "angles": [-2, 0, 2], "scales": [0.97, 1.0, 1.03]},
  "step3": {"refine_candidates": 8, "ecc_iterations": 70, "min_overlap": 0.45, "min_score": 60, "min_template_gap": 3},
  "periodic": {"periodic_completion": true}
}
```

| 分组 | 参数与作用 |
| --- | --- |
| `step1` | 预处理：`scale`、`max_side`、`bottom_ignore_fraction`、默认 `valid_height` |
| `step2` | 粗匹配：`backend`、`angles`、`scales`、`peaks_per_hypothesis`、`seed` |
| `step3` | 精配准与判定：`refine_candidates`、`ecc_iterations`、`edge_tolerance_px`、`min_overlap`、`min_score`、`min_template_gap`、`min_candidate_gap` |
| `periodic` | 周期补全：`periodic_completion`、`period_vectors`、`periodic_anchor_class`、`periodic_max_steps`、`periodic_min_ncc`、`periodic_min_edge` |
| `evaluation` | 标签策略与评估：`label_mode`、`include_background` |

分组用于组织配置；`min_overlap` 等共享阈值也会影响粗匹配筛选。图像路径、每张图像的有效高度和标注仍位于原有位置；顶层 `label_mode` 若存在则优先于 `options.evaluation.label_mode`。
旧的平铺 `options` 仍可读取，但同一参数不能重复出现在平铺与分组中，分组内放错参数也会报错。Python 构造函数仍使用 `MatcherConfig(scale=0.5, ...)`；读取分组字典使用 `MatcherConfig.from_options(options)`。结果报告中的 `config` 仍保存实际生效的平铺参数。

各配置分组可添加字符串 `description` 说明用途；读取配置时会忽略该字段，不影响匹配参数。

## 推理代码与类型标注

函数参数和返回值均有类型标注，数组约定集中在 `utils/types.py`：

| 类型 | 数据约定 |
| --- | --- |
| `Image` | `uint8` 原图，形状为 `(H, W)` 或 `(H, W, 3/4)`，彩色通道为 BGR/BGRA |
| `LabelInput` | 二维整数类别图，或同一图像各类别的二值掩码序列 |
| `LabelMap` | 规范化后的 `(H, W)`、`uint16` 类别图 |
| `Mask` / `BoolMask` | 有效区域或二值类别掩码；布尔输出以 `True` 表示有效 |
| `Matrix` | `3×3 float64` 变换矩阵；公开结果将模板原图坐标映射到 input 原图坐标 |
| `Shape` / `Size` | 分别为 `(高度, 宽度)` 和 OpenCV 的 `(宽度, 高度)` |

图像读写函数也支持 `uint16` 标签图，因此使用通用数组类型 `Array` 标注。类型标注便于 IDE 跳转与提示，数据尺寸和值域仍由运行时校验。

配准流程按职责拆分：`_prepare_image()` 生成工作分辨率特征，`_coarse_candidates()` 搜索粗匹配候选，`_refine_candidates()` 执行 ECC 精配准，`_register_prepared()` 汇总判定结果。多模板选择时，input 的预处理和周期检测只做一次，并供所有模板复用。

`matcher.match()` 负责输入与 ROI 准备、调用模板选择、迁移标签以及可选的周期补全。`warp_valid_mask()` 直接投影有效区域，无需创建一张全零标签图。`workflow.run_config()` 负责配置、文件读写并调用 `save_result()` 导出结果。
