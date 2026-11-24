# 周期结构多模板配准与分割标注映射

## 代码阅读与 Step 职责

公共入口是 [matcher.py](matcher.py) 的 `PeriodicTemplateMatcher`；具体步骤由
[_internal/pipeline.py](_internal/pipeline.py) 的 `MatchPipeline` 显式组织：

输入准备 → 模板配准 → 独立实例定位 → 标签迁移 → 基础周期补全 → 实例支持补全 → 结果融合。

每一步对应独立的 Step 类；评价由 `SegmentationEvaluationStep` 单独负责。
内部按配准、实例、分割和文件流程四个功能目录组织；各 Step 与所属算法放在一起，
顶层只保留流水线、共享模型及包入口，不再单独维护平铺的算法文件和步骤目录。
完整流程图、类职责表、数据契约及“修改某项功能应该找哪个类”见
[ARCHITECTURE.md](ARCHITECTURE.md)。公共 API、结果格式及原有数值算法保持不变。

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

## 多模板并行与线程预算

模板配准默认使用线程池：`MatcherConfig(template_workers=0)` 自动取
`min(模板数, CPU逻辑核心数, 4)`；`1` 关闭外层并行，`2` / `4` 指定上限，
实际仍受模板数和 CPU 核心数限制。单模板不创建线程池。JSON 可直接放在
`options` 下，或与粗匹配参数一起放在 `options.step2`：

```json
{"options": {"step2": {"backend": "opencv", "template_workers": 4}}}
```

目标图特征和周期候选只计算一次，供各模板只读共享。每个 worker 独立预处理模板、
执行粗搜索和 ECC；结果仍按输入索引汇总后稳定排序，同分不会按完成顺序改变优先级。
原有 OpenCV 内部 2 线程策略保持不变，并在启动模板任务前设置，worker 内不修改它。
4 个模板任务可能叠加原生线程和临时数组，不等于仅占用 4 个 CPU 线程；
内存紧张或外层已批量并发处理目标时，应降低 `template_workers`。
OpenCV 线程数是进程级设置，不能理解为每个 matcher 各自隔离；调用期间不要由其他代码更改。

选择线程池是因为本流程主要使用 OpenCV/NumPy 数值运算，可共享目标特征和原图，
无需 Windows 进程启动、数组序列化和额外副本。纯 Python 密集计算、批量多目标和
常驻共享内存进程池需要另行实测，不能断言多线程对所有任务都更快。
本次并行仅覆盖模板配准，实例匹配和周期补全仍按原流程执行；完整推理加速比低于配准部分。

在当前机器和数据上复测（不保存或改写预测文件）：

```powershell
.\.venv\Scripts\python.exe -m tools.benchmark_template_parallelism --templates 4 --repeats 3 --processes
```

当现有模板不足 4 张时，基准循环使用现有模板组成 4 个任务，并明确输出 `repeated=true`。
这只能模拟计算量，不能替代未来 4 张不同模板的验收。统计区分公共目标预处理、
首轮/后续运行时间以及进程池创建/销毁时间，并核对分数和变换矩阵；不包含读图、后处理和保存。
基准的后续轮次复用线程/进程池；正式 API 每次调用创建并回收线程池，评估单次延迟时
应同时查看 `cold_s`，不要只取常驻进程池的后续时间比较。

## class 2 的局部周期与多模板参考

完整身份周期与输入可见数量分别配置，例如：

```json
"instance_matching": {
  "cycle": ["A", "B", "C", "D"],
  "visible_count": 3,
  "min_score": 0.7,
  "min_gap": 0.02,
  "min_visible": 0.15,
  "spacing_weight": 0.3
}
```

`visible_count=3` 时只在 ABC、BCD、CDA、DAB 四个连续窗口中联合判定。
每张模板的 `instance_layouts["2"].order` 可写它实际观察到的连续身份，例如 ABC 或 DAB，
不必单张覆盖四个身份；数组 API 对应 `object_visible_count=3`。
省略 `visible_count` 时保留完整周期模式，每个输入匹配全部身份。

所有已配置模板的参考参与局部候选生成，最终按位置、外观、顺序和间距联合选择。
全图配准失败的模板在局部周期模式下仍按配置的 angles/scales 尝试局部外观搜索，
全图低分本身不否定该模板的实例参考，也不直接认可任何局部候选。
目标左右边缘的实例允许被截断，相关性按实际可见交集计算；没有足够外观或身份分差时
保留未知/歧义状态。不会为了满足 ABCD 而生成图外缺失的第四身份，也不横向周期复制。

class 1 仍由基础语义分支处理。如果基础配准被拒绝，但局部实例通过验证，输出状态为
`accepted_objects_only`，`prediction_scope=ordered_objects`：只输出有证据的 class 2 前景，
其他区域保持未知，原始基础配准状态保留在 `base_registration_status`。
`ordered_objects` 中记录 `cycle_size`、`instance_count`、各模板候选数量和各实例的
`supporting_references`，便于核对多模板贡献。所有相同身份共享原有 1..N 身份编码。

当前 OpenCV 示例配置使用 templates2 中已明确身份的 ABC、DAB 两张模板。
新增模板需显式配置其实际顺序；仅把文件放进目录不会自动启用。

### 640 下的同身份轮廓补全

开启 `shared_periods` 后，普通副本未能补齐时，从已确认的同身份
`supporting_references` 重选画布内周期原点，不再只依赖靠近边缘的主变换。
只对晶格所属模板提出周期一致仿射约束，保留真实观测中心并限制在六个工作像素内；
候选仍需区域外观验证，不跨身份、跨源借用矩阵或扩展截断参考的未知横向范围。

外观异常截断连续源轮廓时，允许用同一轮廓上下独立通过像素 NCC、结构和边缘门槛的
同身份证据桥接中间前景；未验收的间隔受配置周期预算限制，不推断背景，不从新桥接
像素继续自举。没有两端证据的边界或整段遮挡仍保持未知。
所有匹配特征与 ECC 继续使用 `max_side=640`，无 800 回退、无新增图片并行。
新增空间诊断使用原图半开 `target_bbox`，同时记录真实源模板索引。
问题图的前后叠加、串行测量及非 GT 证据边界见
[性能与验证记录](../docs/single_image_performance.md)。

## 匹配前裁剪有效区域

在 `options.step1` 中给 target 和模板设置共同的原图像素边界：

```json
{
  "options": {
    "step1": {
      "valid_x_start": 0,
      "valid_x_end": null,
      "valid_y_start": 0,
      "valid_y_end": 1891
    }
  }
}
```

起点包含在有效区内，终点排除。起点默认是 `0`，终点为 `null` 时取图像相应边界。
Y 终点优先级为每张图的显式高度（`target_valid_height` 或模板的 `valid_height`）、
公共 `valid_y_end`、公共 `valid_height`、图像底部。
边界必须落在图像尺寸内。裁剪后仍与已有图像有效区掩膜取交集，标注有效范围单独处理。

设置矩形边界后，Python Matcher 使用裁剪后的图像、掩膜和标注做缩放、配准及补全。
仅使用旧版 `valid_height` 参数的调用继续使用原图大小的有效掩膜。
输出填回原图尺寸，
矩形之外的预测掩膜与有效性为零。变换矩阵和对象中心使用原图坐标。
`result.json.valid_regions` 记录每张图的有效边界。

```python
from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
matcher = PeriodicTemplateMatcher(MatcherConfig(valid_y_end=1891))
cropped = matcher.get_valid(input_image)
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
显式向量需要独立且非零。类别锚点的间距必须先通过模板图像的自平移验证；
若同一类别包含多种周期内结构、最近间距不是真正图像周期，则回退到 input 图像自相关。
`periodic_completion.anchor_period_validation` 记录验证结果，`period_fallback_reason`
记录回退原因。算法不读取目标 LabelMe 真值来指导预测。

先保留直接配准，再生成整数周期副本，只处理有机会提供未知标签的候选。每个候选进行 ECC
校正和图像验证，校正幅度受周期约束；默认要求结构 NCC >= 0.75、边缘评分 >= 0.6。
还会对每个待新增连通区域及其有限邻域单独验证，避免内部大面积重合掩盖边缘不匹配。
通过后，以最近邻迁移类别和 annotation_valid，填充背景与前景，保留类别 ID 和已有非零前景标签。
若同一个周期前景连通区域横跨未知区与直接配准区，会将其在背景 0 上的部分一并补齐，避免物体被配准边界截断；完全位于直接区域内的其他物体不会因此扩张。
`periodic_max_steps` 默认 2（范围 1..5），限定每个周期方向的复制步数，避免无界搜索。

可选 `options.periodic.shared_periods: true`（通用默认 `false`，SD-1 OpenCV 配置已开启）：
将模板自平移、目标自平移及同身份/同参考的出现间距联合校验为源/目标周期向量，可靠格点
共享几何，不再逐副本仿射 ECC；不可靠轴回退上述原路径。类权限仍按源轴检查，目标倾斜不
改变重复方向。对象支持按同身份/同参考复用残差平移，所有新增区域仍独立验收自身证据。
这是允许标签变化的算法选项，不保证旧输出逐位相同，也尚未在 C++ 内核实现；详见
[性能与差异说明](../docs/single_image_performance.md)。

共享模式还会在实例融合后检查**完全位于已知背景中的独立语义实例**，弥补上述未知种子
策略无法覆盖的漏项。只从融合后已经存在、周围标注有效且未被其他类别遮挡的完整轮廓出发，
按已验证晶格与类别源轴权限，在配置步数内提出同图周期对应。每个新轮廓须以原分辨率
离散像素 NCC、640 等既有工作分辨率的局部结构 NCC 和对称边缘评分独立通过原阈值。
只写已知空背景；不覆盖任何旧前景、身份或未知区域，不新增 ECC，不以新补结果继续播种。
相同目标的多个别名只选自身像素证据最强且通过结构/边缘验证的一份，不叠加扩大轮廓。
该路径不复制独立身份类别，不改变已验证实例纠正 base 相位的既有优先级。
`periodic_completion.interior_instances` 记录新增像素、源轮廓的 `source_target_bbox`、
目标 `target_bbox`、偏移、周期向量及评分；两个框均为目标原图坐标下的半开范围。
它只补标签，不把目标实测图像冒充健康模板，也不更改 `completed_template`。

API 的 `mapped_labels` / `label_valid` 为最终预测；`geometric_valid` / `warped_template`
仍是直接配准。新增 `completed_template`、`completed_valid`、`periodic_valid`、
`label_periodic_valid` 分别表示补齐图像、补齐几何范围、图像补齐区和标签补齐区。
未执行周期补齐时这些新增字段为 None。仅在 semantic 模式补齐，不复制唯一实例编号。

### 独立保留各阶段掩膜（允许重叠）

单通道 `mapped_labels` 每个像素只能容纳一个类别；已验证实例纠正 base 相位时，
最终 label=2 可能覆盖直接迁移的 label=1。现在另外保留以下互不覆盖的来源证据：

| API 字段 | 保存内容 |
| --- | --- |
| `direct_labels` / `direct_valid` | 直接语义匹配标签；启用独立对象分支时，先移除对应类别（当前配置为 label=2），再变换并保存 |
| `completion_labels` / `completion_valid` | 已验收周期补全，融合之前保存；也包含融合后通过验证的独立小实例补全 |
| `object_labels` / `object_layer_valid` | 独立对象前景，不因基础前景保护或最终优先级而删除 |
| `object_layer_identities` | 独立对象层身份；语义一致但身份不明的前景保持身份 0 |

这些字段均为目标原图尺寸，各自携带 validity，输出为同名 PNG。三个来源的同一位置
可以同时保留类别 1 和类别 2；它们是多份匹配假设，不表示冲突双方必然都正确。
启用对象分支时，label=2 只来自对象层，不从直接层或语义补全层引入未经独立定位的模板投影。
类别边界由 `ordered_object_class` 决定，并非硬编码为 2；未启用对象分支的纯语义模式不排除类别。
旧 `mapped_labels`、`label_valid`、身份字段和最终优先级不变，评价仍使用最终互斥标签。
不要再通过“最终标签减去对象/周期区域”反推直接匹配掩膜；那会不可逆地丢失重叠信息。
查看器默认“全部来源（重叠混色）”，可单独看三层或切换“最终互斥标签”；按类别查看时
保留该类别在各层的完整区域。旧导出没有独立层时沿用历史显示，必须重新推理才能补齐这些证据。

最终类别预测保存在 `mapped_labels.png`，其中已包含通过验证的周期补齐。`label_periodic_valid.png` 单独记录由周期补齐的标签像素；app 可切换“全部预测 / 直接配准 / 周期补全”观察。

`result.json.periodic_completion` 记录来源、周期向量、每个副本的变换和评分、
`label_coverage_roi`、`remaining_unknown_pixels` 以及 complete/incomplete/no_period 状态。
100% 覆盖表示 ROI 内每个像素都有类别预测（包括背景），不表示前景覆盖整图或 100% 准确率。
无可靠周期、验证失败或源标注不完整时保持未知并报告原因。局部标注必须配置 annotation_valid，
否则 LabelMe 未画区域按既有约定视为已知背景。底部排除区域始终不参与补齐。

### 按类别配置横向/纵向平移重复性

这里的“周期=1”表示该方向不存在让结构对应对齐的非零平移；“周期>1”表示存在这种平移。
不是出现次数，也不是周期距离为 1 像素。推荐在 `options.classes` 中按类别显式声明规则：

```json
"options": {
  "classes": {
    "1": {"repeat_axes": ["x", "y"]},
    "2": {
      "repeat_axes": ["y"],
      "instance_matching": {"cycle": ["A", "B", "C"]}
    }
  },
  "periodic": {
    "periodic_completion": true
  }
}
```

| 某类别的方向列表 | 横向 | 纵向 | 周期标签补全 |
|---|---|---|---|
| `[]` | =1 | =1 | 禁止；保留直接配准/独立定位的标签迁移 |
| `["x"]` | >1 | =1 | 仅横向 |
| `["y"]` | =1 | >1 | 仅纵向 |
| `["x", "y"]` | >1 | >1 | 两方向及其组合 |

例如以后类别 1 横纵均不可平移重复，只需把 `classes["1"].repeat_axes` 改为 `[]`，不必关闭其他类别的周期补全。
白色缺陷处理与此独立：获准方向的几何关系由可见纹理验证后，仍迁移缺陷覆盖处的源标签。

- 方向按模板原图坐标定义。周期距离仍由已有 `period_vectors` / 锚点 / 图像自相关提供，
  本字段不是按类别指定不同的像素周期；当前实现使用共享晶格。声明允许方向不代表跳过图像验证。
- 单方向策略允许不超过主方向 15% 的小倾斜；双方向策略支持斜晶格。每个实际使用的晶格向量
  都须符合该类别策略，不能用两个禁止的斜向平移相互抵消来绕过方向限制。
- `classes` 必须为非空对象，每个类别必须显式填写 `repeat_axes`；未列出的前景类别默认不可
  周期复制。规则可包含 `description`。没有 `instance_matching` 的类别只执行普通语义标签迁移。
- `instance_matching` 的策略作用于同一身份，不能把 A 当成 B 复制。
  循环实例定位器沿 `order_axis` 每个身份只定位一次，该轴不可重复；支持补全沿垂直于排列的方向。
  因此 x 排列可设 `[]` / `["y"]`，y 排列可设 `[]` / `["x"]`；沿排列轴重复会明确报错。
  普通语义类别支持表中全部四种规则。
- 禁止的类别不会被复制，也不会被简单改成背景 0。若某候选平移不符合任一基础前景类别的
  策略，该副本的背景不能证明这些类别不存在，因此只迁移获准的正标签，其余位置保留未知。
  独立实例的背景有效性仍由身份分支单独控制；基础标签和已有前景保持不变。
- `result.json.class_rules` 保留新的按类别配置，`config.class_periodicity` 是规范化后的运行时规则，
  `periodic_completion.class_periodicity` 记录基础类别生效规则，副本的 `blocked_class_ids` 记录禁止复制的类别。
  `geometry_accepted` 仅表示图像几何通过全局验证，`accepted` 表示确有通过局部验证的标签补全。
  独立实例不会仅凭前者填充，还必须通过自身的逐区域验证。

## 类别 2：N 个不同实例的循环排列

类别 2 包含 N 个不同实例，每个身份在横向只匹配一次。“横向周期=1”表示不存在保持该身份结构
对应对齐的非零横向平移，不是一像素周期，也不表示这些实例外观相同。实例独立按外观匹配，联合检查已知循环顺序
和模板中的相邻间距，支持 input 边缘残缺。例如 abcde 可变为 bcdea、cdeab 等。
使用时显式配置各模板原始标注的身份顺序，例如 MA 为 ABC、UA 为 CAB：

```json
{
  "templates": [
    {
      "image": "MA.jpg", "labelme": "MA.json", "class_map": {"1": 1, "2": 2},
      "instance_layouts": {"2": {"order_axis": "x", "order": ["A", "B", "C"]}}
    },
    {
      "image": "UA.jpg", "labelme": "UA.json", "class_map": {"1": 1, "2": 2},
      "instance_layouts": {"2": {"order_axis": "x", "order": ["C", "A", "B"]}}
    }
  ],
  "options": {
    "classes": {
      "1": {"repeat_axes": ["x", "y"]},
      "2": {
        "repeat_axes": ["y"],
        "instance_matching": {
          "cycle": ["A", "B", "C"],
          "min_score": 0.7,
          "min_gap": 0.02,
          "min_visible": 0.15,
          "spacing_weight": 0.3
        }
      }
    }
  }
}
```

`instance_layouts["2"]` 明确绑定类别 2；`order_axis: "x"` 表示按从左到右的位置给原始形状赋予
`order` 中的身份，UA 模板即 CAB。`classes["2"].repeat_axes: ["y"]` 则明确同身份横向周期=1、
纵向周期>1。**排列方向、模板中的具体顺序和同身份重复方向是三件不同的事。**

布局匹配器支持 `order_axis: "x"`（从左到右）和 `"y"`（从上到下）。所有模板必须声明一致的
排列方向。纵向布局通过可逆坐标适配复用同一候选、间距、顺序及支持补全算法；导出标签、中心、
变换矩阵、支持向量和校正矩阵仍处于实际输入坐标，不会泄漏内部转置坐标。
当前只支持一个类别启用 `instance_matching`；多类别独立实例匹配会报错，不会覆盖前一个类别。
`instance_layouts` 的类别必须与启用 `instance_matching` 的类别相同，每张模板须提供该类别布局。

布局保留每个原始 polygon，不按 JSON 绘制顺序赋值，也不从合并类别图恢复身份。
N 由 `instance_matching.cycle` 长度决定。完整周期模式下，每张模板包含 N 个参考形状，
`order` 是 `cycle` 的循环移位；设置 `visible_count` 后，模板也可提供连续的局部周期。
当前 SD-1 完整周期为 ABCD，每张输入可见三个连续身份，两个模板顺序分别为 ABC、DAB。
此功能只支持 semantic 模式。关闭时移除该类别的 `instance_matching` 及各模板对应的
`instance_layouts`，保留 `repeat_axes` 即可。设 `repeat_axes: []` 只关闭周期复制，不关闭身份定位。

参数 `cycle` 必填；`visible_count` 对应数组 API 的 `object_visible_count`，省略表示完整周期。
其余匹配参数可选：`min_score`、`min_gap`、`min_visible`、`candidates`、
`spacing_weight`。它们对应数组 API 的 `object_cycle`、`object_min_score`、`object_min_gap`、
`object_min_visible`、`object_candidates`、`object_spacing_weight`，匹配门限语义不变。

旧配置 `options.objects`、`options.periodic.class_periodicity` 和模板 `object_order` 仍可读取。
新 `classes` 不能与旧实例/类别方向参数混用，同一模板不能同时写 `instance_layouts` 和
`object_order`；有歧义时明确报错。旧方向字段省略或为 `{}` 时继续保持旧默认行为。
数组 API 仍使用 `MatcherConfig` 原有字段，新 JSON 只规范化到同一套算法，不引入第二套推理逻辑。

### 同时将模板和 input 旋转 90°

以原来从左到右 CAB、同身份纵向重复为例：

| 操作 | 新排列轴 | 新顺序（从上到下） | 类别 2 的 repeat_axes | cycle |
|---|---|---|---|---|
| 顺时针 90° | `"y"` | CAB | `["x"]` | ABC（不变） |
| 逆时针 90° | `"y"` | BAC | `["x"]` | CBA（原 cycle 反转） |

每张模板的 `order` 按相同规则处理；类别 1 的 `["x", "y"]` 仍保持双方向。若显式指定了
`period_vectors`，也须旋转向量（不带平移部分）。`angles` 是模板与 input 的**相对**旋转，
二者都转 90° 时不应把它机械改为 90。

标注及图像 ROI 也必须一起旋转。原来的 `valid_height: 1891` 是顶部有效行数，旋转后有效区
变成左侧/右侧区域，不能继续用同样的 height 截取；应旋转原始有效 mask，配置模板
`valid_mask` 和 `target_valid_mask`，移除原高度设置；数组 API 将配置中的 `valid_height` 设为 `None`。
数组 API 同理传入旋转后的 `template_valids` / `target_valid`。

自动测试覆盖顺/逆时针、全部循环相位、两端裁切、原坐标变换投影、旋转后的白色缺陷补全及
JSON 文件流程。另有真实 SD-1 两模板和当前 input 的完整配准测试；这些真实旋转结果是与原
预测旋转后的**一致性**对比，不是 GT 准确率，ECC 和离散峰值/参考选择不保证严格逐像素等变。

模块 [_internal/objects/matching.py](_internal/objects/matching.py) 对每个物体的原图外观构造候选，使用实际可见交集上的零均值
归一化相关性，分别计算交集均值和能量；两侧可见区域都必须有实际纹理，避免平坦背景因
数值误差得到伪高分。只在可见面积达到要求后比较 N 种循环排列的联合评分。
动态规划在每个循环排列中选一条从左到右的最佳实例路径，避免穷举 K^N 个组合。
联合分数为平均外观质量减去相邻间距误差：每个相邻对取
`object_spacing_weight * min(abs(observed_gap/expected_gap - 1), 1)`，再对相邻对平均。
预期间距从模板中相邻完整参考的原始 mask 中心测量，经过几何变换换算到 input 像素；
多个模板取中位数，不假设等间距，不用图像宽度虚构缺失的环绕间距。没有参考的相邻对不加间距约束。
不同循环排列接近时保持未知，同一排列的不同模板参考不会单独造成身份歧义；N=1 检查位置歧义。
完全不可见或证据不足的身份不由顺序强行生成。
周期定位仅提供几何初值，物体 mask 不按单周期横移复制。评分对短纵向片段作惩罚，避免小片段
偶然高相关压过完整物体。边缘物体可裁切匹配，但身份仍需外观和相邻物体共同支持。
源参考本身横向被截断时，不将未观察到的物体部分标成背景；完整的同身份参考可提供补充。

通用 API 的 `object_min_gap` 默认 0.03；当前 SD-1 场景设置为 0.02。这是联合相似度分差，
不是概率或准确率，需要更多真实样本校准。若无法区分身份或可见证据不足，类别 2 保持未知。
即使类别 1 已有周期预测，也不会把其副本中的 0 当成“这里没有类别 2”的证据。

输出语义类别仍为 2，另存：

- `object_identities.png`：uint16 身份图，1..N 对应 `object_cycle` 中的身份；这些数字不是语义类别 ID。
- `object_valid.png`：物体分支的有效范围。
- `base_labels.png`、`base_valid.png`：类别 2 合并前的基础标签与有效范围，保留原配准和周期补全输出。
- `result.json.ordered_objects`：类别 ID、排列方向、状态、候选/最终排列、分差、逐物体来源、可见比例和原图变换矩阵。
- `result.json.class_rules` / `template_instance_layouts`：按类别的原始规则和按模板顺序规范化的身份布局。

类别 2 不参与基础标签的周期复制。完整周期模式保护基础分支已有前景；局部周期模式下，
经过身份验证的类别 2 前景优先于基础标签，避免全局周期错相位的类别 1 剪掉实例分叉。
重叠像素数记录为 `foreground_conflict_pixels`，实际覆盖数记录为 `instance_priority_pixels`。
类别 2 不确定时不会清除类别 1 的前景。
最终 `label_valid` 区分未知背景；基础分支本身的完整有效范围仍由 `base_valid` 独立保存。

基础周期补全仍有空洞时，其他已通过几何验证且有标注的模板继续提供直接及经验证的周期标签，
只补未知区域，记录在 `periodic_completion.supplemental_templates`。
局部周期实例在选定位置后进行有位移上限的局部 ECC 精配准，相关性和边缘指标改善后才采用，
诊断字段为 `local_refined`。配置的有效 ROI 外区域不参与补全。
局部周期的候选搜索限制同一排列轴位置的重复峰数量，为其他位置（包括低分的边缘截断实例）
保留候选机会，避免纵向周期峰占满名额后错误地接受另一种身份排列。

若已确认身份的参考支持不完整，[_internal/objects/completion.py](_internal/objects/completion.py) 可补充垂直于排列轴方向的支持：
x 排列补纵向，y 排列补横向；仅使用基础分支已经验证的周期，并将其转换到 input 坐标。
同身份原始参考在新位置的每个填充区域都必须通过局部结构相关与边缘对齐检查，才能补充
mask 和背景有效性；不能沿排列轴周期复制实例，也不能补出源参考在排列轴上未观察到的范围。
诊断保存在 `ordered_objects.support_completion`，包括新增像素、变换和逐区域验证指标。

白色缺陷不是“缺失标签”：已确认身份的支持补全时，原图灰度至少 230、连通面积至少
64 像素的近白区域只从配准验证证据中排除，不从输出 ROI 或模板 segmentation 中扣除。
每个补全区域仍须有至少 64 像素且不少于一半面积的非遮挡纹理，通过原始灰度相关性、结构
相关性和边缘对齐门限，再用该变换迁移完整源标签（包含白色缺陷下面的前景与背景）。
不通过降低分数门限实现补全；随机不匹配、完全没有可见证据、未标注源区域和显式排除的
目标 ROI 仍不能被填充。诊断中的 `evidence_pixels` / `occluded_pixels` 可用于区分验证证据
与遮挡下的标签迁移；覆盖率表示标签有效范围，不是缺陷检测准确率或分割 GT 准确率。

`ambiguous_identity` 表示身份顺序仍有歧义。全局配准已接受时，局部周期模式可在各接近候选
共同确认的前景像素写入语义类别，身份仍为未知，已有的其他前景类别受到保护。
两项像素数分别记录在 `semantic_consensus_pixels` 和 `semantic_consensus_added_pixels`。
若矩形裁剪后仍有身份歧义，且已有可用于对象补全的周期几何，Matcher 会在原尺寸画布上
用同一有效区复核身份。原画布与裁剪结果给出同一顺序且复核通过身份门槛时，才按复核结果
继续逐区域验证补全。复核状态和两次分差记录在 `identity_retry`。
`partial_reference` 表示使用了不完整源参考，
未看到的范围保持未知。`accepted` 仅表示通过当前匹配条件。模板分差不足时，仅当几何匹配已通过且
物体身份有充分证据，才以 `accepted_ordered_objects` 解决语义模板歧义；不会绕过几何门限。

数组 API 的 match/evaluate 新增 `template_objects`，每张模板提供
`{"identity": "A", "mask": original_size_binary_mask}` 参考。完整周期模式要求全部 N 个身份；
局部周期模式允许连续的身份子集，身份名称必须属于 `object_cycle`。
无需传入 input 的标注。MatchResult 提供 `object_identities`、`object_valid` 数组。
独立查看 app 自动列出全部 N 个身份，可选择“独立物体”来源和具体身份，鼠标悬停可查看。

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
```

旧版单文件启动、旧配置类和兼容导出已经移除，不再维护多套调用方式。只使用上面的模块启动命令或下方的 Matcher API。请从工作区根目录运行，而不是进入包目录。输出路径相对于运行目录；配置内的输入路径相对于配置文件，不受运行目录影响。

所有输入路径相对于JSON文件所在目录，也可以使用绝对路径。Windows中文路径可读取。

当前 [SD-1 配置](config.sd1.json) 包含两张模板，可按实际输入替换路径。若只检查配准，无需设置 `labels` 和 `annotation_valid`。底部文字排除比例默认为 0，本例设为 0.08，须按实际图像调整。

每次运行建议使用新的输出目录，便于保存不同参数结果。重用输出目录时，仅清理程序拥有的已知结果文件，避免上次的真实标签被误认成本次结果；其他文件不会删除。禁止输出覆盖配置中的输入文件。

文件流程可通过 `from periodic_template_matcher import run_config` 导入，再调用 `run_config(config_path, output_dir)`，返回与 `result.json` 相同的字典。

重复使用同一套模板时，可以在创建 Matcher 时完成配置读取和预处理。

```python
from periodic_template_matcher import PeriodicTemplateMatcher

matcher = PeriodicTemplateMatcher(config_path)
result = matcher.infer()
images = matcher.save_images(result, output_dir)
manifest = matcher.save_json(result, output_dir, artifacts=images)
```

`infer()` 返回内存中的 `MatchResult`。`save_images()` 只写 PNG，`save_json()` 只写
`result.json`，可单独调用。若在同一个目录使用两者，传入 `artifacts=images` 让 JSON 引用已保存图片。
`save_result()` 一次性保存查看器所需的完整结果，包括独立的变换矩阵文件。
`infer_to()` 组合推理与完整保存，返回结果字典。传入 `timings={}` 可收集本次调用的
`seconds` 和单独的 `image_save_seconds`。`seconds` 保留推理和 JSON 导出耗时，扣除 PNG 编码与写盘。

目标路径省略时使用配置中的
`target`，显式传入的相对路径以配置文件所在目录为准。配置含有 `target_valid_mask` 时，该掩膜只适用于默认目标。
模板文件、默认目标及配置以构造时的内容为准，文件变更后请创建新实例。

## 模块结构与阅读顺序

包顶层只有 **4 个 Python 文件**，只呈现主要接口。私有实现顶层只有 **3 个 Python 文件**，
其余按配准、实例、分割、文件流程四个领域归组，Step 与对应算法共处同一领域。
测试放在工作区根目录，不随业务接口混排。所有代码使用正常的包导入，不再修改 `sys.path`。

| 模块 | 单一职责 |
|---|---|
| [包入口](./__init__.py) | 明确列出公开 API；不执行 CLI 或读写配置 |
| [配置](config.py) | 唯一的 `MatcherConfig` 与参数校验 |
| [Matcher](matcher.py) | `match` / `evaluate` / `save_result` 的流程编排 |
| [结果查看 app](app/README.md) | 独立读取保存的 input、mask 和有效区域，交互观察 |
| [模块启动](./__main__.py) | 唯一的命令行启动入口及参数解析 |
| [流水线](_internal/pipeline.py) | 唯一的推理顺序、分支与提前返回 |
| [共享模型](_internal/models.py) | 结果类和跨步骤数据契约 |
| [配准领域](_internal/registration/step.py) | 配准 Step、FFT/OpenCV 粗搜索、ECC、评分 |
| [实例领域](_internal/objects/steps.py) | 身份匹配、轴适配、支持补全、结果融合 |
| [分割领域](_internal/segmentation/transfer.py) | 输入准备、标签迁移、周期补全、独立评价 |
| [文件流程](_internal/io/workflow.py) | 标注读取、结果导出及 JSON 工作流 |
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

API 调用示例可参照 [tests/test_matcher_api.py](../tests/test_matcher_api.py)。
VS Code 左侧“测试”视图已配置为发现 tests/，可以直接运行或调试单个用例。

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
    valid_height=1891, refine_candidates=8,
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

[API 回归测试](../tests/test_matcher_api.py) 使用已知变换和合成标注验证 `match()`、
`evaluate()` 与 `save_result()` 的调用流程、预测有效性和指标；合成测试不代表产线精度。

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
  "step1": {"scale": null, "max_side": 800, "valid_height": 1891},
  "step2": {"backend": "opencv", "angles": [-2, 0, 2], "scales": [0.97, 1.0, 1.03]},
  "step3": {"refine_candidates": 8, "ecc_iterations": 70, "min_overlap": 0.45, "min_score": 60, "min_template_gap": 3},
  "periodic": {"periodic_completion": true}
}
```

| 分组 | 参数与作用 |
| --- | --- |
| `step1` | 预处理：`scale`、`max_side`、`valid_height`、`valid_x_start`、`valid_x_end`、`valid_y_start`、`valid_y_end` |
| `step2` | 粗匹配：`backend`、`angles`、`scales`、`peaks_per_hypothesis`、`seed` |
| `step3` | 精配准与判定：`refine_candidates`、`ecc_iterations`、`edge_tolerance_px`、`min_overlap`、`min_score`、`min_template_gap`、`min_candidate_gap` |
| `periodic` | 周期补全：`periodic_completion`、`shared_periods`、`period_vectors`、`periodic_anchor_class`、`periodic_max_steps`、`periodic_min_ncc`、`periodic_min_edge` |
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
