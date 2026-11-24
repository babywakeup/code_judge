# 单图推理性能优化与 Codex 审查说明（2026-11-23）

> 本文是实现说明和已有验证记录，不是独立审查结论。请 Codex 检查代码、证据和边界条件，不要仅根据“哈希一致”或本文的解释认定所有输入都等价。

## 最新：独立保留来源掩膜，避免融合不可逆丢失

按用户要求新增直接、周期补全、对象三套独立标签/valid，以及独立对象身份；原最终
单通道标签、融合优先级和验收阈值保持不变。直接层在移除独立匹配的 ordered 类之后保存，补全层
在融合前保存且追加后续已验收的小实例，独立对象层不受最终前景保护裁剪。查看器默认
对不同来源的重叠类别混色，可单独查看来源/类别，也可回到原“最终互斥标签”。
这避免了原查看器从最终预测反推来源时把 label=1 与对象 label=2 的重叠删掉；
保留多份假设不等于判定冲突双方都正确，也不能恢复各阶段从未产生过的掩膜。
初版曾在移除 ordered 类之前保存直接层，导致“全部来源”额外显示未经对象验证的 label=2。
后续修正为先排除该类别、再执行一次语义变换并保存独立副本；对象层及原最终标签规则不变。
下述 verified640 目录是修正前的历史证据，其直接层可能包含该类投影，不能作为修正后输出复用。

修正后再次按 640、逐图串行运行同两张图，保存于
[ordered_class_excluded640](../periodic_template_matcher/results/mask_layers/ordered_class_excluded640/validation.json)：
两图直接层/补全层的 label=2 均为 0，对象层和最终互斥标签与修正前逐像素完全一致。
直接层不再额外带入对象层之外的 49,161 / 15,062 个 label=2 像素；这些是已删除的
未经对象验证的额外投影数量，未读取 GT，不能将数量直接解释为精度提升或已确认的全部误检。
第一张 label=1 与对象 label=2 的 22,827 像素重叠仍完整保留。新增回归先复现 3 项失败，
修正后完整回归 **296 项，295 通过，仅同一原有目录检查失败，0 errors/0 skipped**，见
[本次回归](../periodic_template_matcher/results/mask_layers/ordered_class_excluded640/regression.json)。

640、逐图串行、`template_workers=1` 核对用户两张 TNXSS2：

- CS01NA / 53：直接层有 **22,827 个 class 1 像素**与对象 class 2 重叠，分为四个组件；
	原最终互斥标签会显示 class 2，现在直接层与类别 1 视图均保留全部像素。
- CS028A / 29：直接层和补全层与对象 class 2 的 class 1 重叠均为 **0**，没有复现同因；
	不能宣称该图所有缺口已由分层解决，也不能据此将其问题归咎于 NCC 阈值。

见 [实图记录](../periodic_template_matcher/results/mask_layers/verified640/validation.json)、
[第一张四视图](../periodic_template_matcher/results/mask_layers/verified640/target1/comparison.png)
和 [第二张四视图](../periodic_template_matcher/results/mask_layers/verified640/target2/comparison.png)。
单次计时不是本次性能对照，不据此宣称加速；额外数组和 PNG 会增加内存、存储及导出成本。

完整 Python 回归 **293 项，292 通过，仅原有内部 reference 目录检查失败**，见
[回归记录](../periodic_template_matcher/results/mask_layers/verified640/regression.json)。
新回归覆盖三层重叠、独立身份、有效区、来源渲染、文件校验/清理、裁剪还原和两张实图。
旧顺序测试 mock 更新为返回真实 `TransferredLabels` 契约，没有放宽融合或匹配断言。
实图测试逐张释放结果及加载副本，避免累积全分辨率数组。本轮没有增加图片并行。
图库需重启、刷新页面并强制重算才能使用新层，旧缓存不能事后恢复丢失信息。

## 最新：640 下恢复 TNXDP0 的四个 class 1 小实例

问题图为 TNXDP0 / AFE55CN01KA / 文件索引 59。旧的基础周期补全要求新增前景接触
未知像素；四个独立小实例完全位于已知背景中，因此即使 ROI 的 `label_valid` 已达
100%，仍不会提出补全。**覆盖率不是前景完整性或精度。**

新增 [独立语义轮廓验证](../periodic_template_matcher/_internal/segmentation/instances.py)，
在身份融合后执行。冻结已存在、标注有效且未贴源边界/其他类别的完整非身份轮廓，
按已验证共享晶格与类别源轴权限生成配置步数内的目标对应；来源和目标的实际离散
原图像素 NCC、工作分辨率局部结构 NCC 与对称边缘均须独立通过既有阈值。
不使用整图高分替小物体借分，也不对原像素做单侧平滑后放宽验收。

所有旧前景、class 2 掩膜及身份、标注 valid、base 快照、模板选择、全局矩阵和
CDA 顺序保持不变；只在已知空背景新增标签。不认证未知区、不修改健康模板，
不以新增轮廓继续播种；多个等价周期别名择优验收，不并集膨胀。未共享或轴未验证时
保留原策略，不变更已验证 class 2 纠正错误 base 相位的既有优先级。

本图恢复 **4 个独立实例、26,572 个 class 1 像素**，总 class 1 从
839,155→865,727。半开原图范围为：

| 目标范围 | 新增像素 | 自身离散原像素 NCC |
| --- | ---: | ---: |
| `[2020,427,2132,514)` | 5,401 | 0.98558 |
| `[1491,527,1573,665)` | 7,880 | 0.98253 |
| `[2014,1306,2127,1392)` | 5,409 | 0.98373 |
| `[1492,1402,1574,1541)` | 7,882 | 0.98711 |

四处结构 NCC 为 0.9885–0.9949，边缘评分 0.8303–0.8928；并非仅以面积增加判断恢复。
见 [原图/旧结果/新结果局部对照](../periodic_template_matcher/results/class1_small_instances/verified640/target1/instance_crops.png)
（每行从左到右）及 [全图前后对照](../periodic_template_matcher/results/class1_small_instances/verified640/target1/comparison.png)
（左旧右新，黄色 class 1，绿色 class 2）。没有读取目标人工 GT，不宣称整图或全集精度。

复现工具 [benchmark_periodic_instances.py](../tools/benchmark_periodic_instances.py) 在同一个
复用 Matcher 中只开关新阶段；配置固定 640、`scale=null`、`template_workers=1`，
逐图串行，分别预热后 AB/BA 三轮，保存实现 SHA256 和双方完整输出。计时只含暖机
`infer()`，不含初始化与 PNG 导出；不能拿历史不同环境的 TN8PR2 10 秒作本轮基线。

问题图中位数 **5.228→5.391 秒**，约增加 **0.163 秒（3.1%）**；新阶段自身约
0.117 秒，ECC 调用数 **28→28**。既有 TN8PR2 的最终标签和身份未改变，新阶段
未新增前景；此前修复的 class 2 支持仍保留。本轮不增加图片并行、工作分辨率或 ECC，
也未下调阈值。原始计时见
[本轮串行报告](../periodic_template_matcher/results/class1_small_instances/verified640/benchmark.json)。

另三张控制图单独在同一实现下完成串行 AB/BA 三轮：

| 控制图 | 前→后中位数 | 新阶段中位数 | 新增 class 1 |
| --- | ---: | ---: | ---: |
| CN01MA / 42 | 10.972→12.271 s | 0.156 s | 0 |
| CS01AA / 94 | 12.278→12.973 s | 0.317 s | 26,473（4 个） |
| CS026A / 90 | 5.551→6.905 s | 0.072 s | 0 |

控制图总耗时波动明显（第三张旧模式 11.115/5.551/5.477 秒），不能把总差值都归因于
仅约 0.07–0.32 秒的新阶段，也不能保证全集固定 3.1% 开销。CS01AA 的另外四个同类
漏项也通过独立证据恢复，自身原像素 NCC 为 0.9755–0.9832；见
[控制图局部对照](../periodic_template_matcher/results/class1_small_instances/controls640/target2/instance_crops.png)。
五张图的所有旧前景、class 2、身份、valid、基础快照及全局矩阵均保持不变；TN8PR2、
CN01MA 和 CS026A 的最终标签逐像素不变。证据分为两份独立输出，不把未完成的终端
批次当成五图报告；[汇总记录](../periodic_template_matcher/results/class1_small_instances/validation.json)
核对两批实现 SHA256 相同，且本图旧输出与修复前冻结 baseline 逐像素一致。

小实例、周期权限、多源融合、流水线和旋转相关 **72 项测试全部通过**。
新增小实例回归 20 项，覆盖实际本图、原像素/结构/边缘独立门槛、源轴权限、未知与遮挡、
源裁边、预算内禁止自举、别名去重、亚像素离散采样及转置/裁剪恢复。
完整 Python 回归 **265 项，264 通过，仅原有内部 reference 目录检查失败**，无新增失败；
没有删除该无关目录或修改断言，见
[完整回归记录](../periodic_template_matcher/results/class1_small_instances/verified640/regression.json)。
图库需要重启并强制重算问题图；代码变更不自动淘汰旧缓存，历史结果目录未覆盖。

## 前次修复：保留 640 并修复 TN8PR2 的 class 2 断裂

当前业务配置由用户设为 `max_side=640`、`scale=null`，本轮保持不变；没有较高分辨率
回退、图片级并行、阈值下调或增加 ECC 迭代预算。旧的 800/640 对照仅为下节历史实验。

### 根因与修复

问题图为 TN8PR2 / AFE564H02RA / 文件索引 0。**800 自身也漏标，不能把 800 输出
当作正确答案；只在共同已知像素上对照会排除两者共享的未知缺口。**

1. D 的主出现靠近底部，仅从它上下各一个周期不能提供上部完整支持。常规副本完成后，
	若仍有缺口，从真实可见的同身份、完整源参考重选一个画布内周期原点；新填充只补前景。
2. 短片段局部 ECC 的微小仿射斜率误差在多个周期后累积。用已验证的源/目标向量提出
	`L * source_period = target_period` 最小变化约束，固定原观测中心、限制源轮廓角点
	六个工作像素位移；只用于晶格所属的那个模板，禁止跨源套用。原模型仍参与竞争。
3. A/D 的外观异常内部不能直接通过健康模板 NCC。只有同一源连通轮廓两端的同身份
	可见像素分别通过原像素 NCC、结构 NCC、边缘门槛，才迁移两端间的真实源前景。
	最近端点仍在异常中时可向外逐窗检查，但总未验证间隔不超过配置周期预算；不能跨越
	其他身份，不能外推单侧边界，不能用大分量替小分量借分，不认证背景。
4. 冻结普通验收后的身份/valid 快照；新桥接前景不能再充当后续端点。按源参考去重，
	复用残差成功/失败缓存；原始像素证据不合格时跳过昂贵上下文评分，ECC 救援和分块
	复验仍保留。普通副本已补齐时直接跳过原点排名，限制投影缓存为一个候选。

实现见 [support_anchors.py](../periodic_template_matcher/_internal/objects/support_anchors.py)、
[completion.py](../periodic_template_matcher/_internal/objects/completion.py)。空间记录包含
真实模板/参考与半开 `target_bbox`；转置和裁剪恢复均同步处理。

### 实图恢复及速度代价

最终证据在
[串行对比报告](../periodic_template_matcher/results/support640_lattice_verified/benchmark.json)，
问题图见 [640 前后叠加](../periodic_template_matcher/results/support640_lattice_verified/target1/comparison.png)
（左旧、右新，绿色 class 2）。相同 640 配置、同一复用 Matcher、逐图串行 AB/BA 三轮，
`template_workers=1`；计时只含暖机后的 `infer()`，不含初始化、哈希和 PNG 导出。
旧补全模块在内存中从保存快照载入，不替换生产文件，报告保存旧实现 SHA256。
复现工具为 [benchmark_object_support.py](../tools/benchmark_object_support.py)。

| 实图 | 修复前中位数 | 修复后中位数 | 补全阶段前→后 | 新增 class 2 前景 |
| --- | ---: | ---: | ---: | ---: |
| TN8PR2 / AFE564H02RA / 0 | 9.380 s | 10.065 s | 1.656→2.321 s | 81,542 |
| CN01MA / 42 | 9.089 s | 8.946 s | 0.005→0.005 s | 0 |
| CS01AA / 94 | 9.955 s | 9.457 s | 1.923→1.420 s | 0 |
| CS026A / 90 | 9.129 s | 9.180 s | 0.938→0.807 s | 0 |

问题图 class 2 从 **196,770→278,312 像素**，原有前景、已确认身份和已知像素均无丢失。
D/A/B 三条源轮廓在 ROI 内均恢复为单个连通分量，贯穿有效高度；不是只增加总面积。
D 在原图 `[478,942,581,1188)` 内补入 12,419 个桥接像素，A 在
`[1301,834,1459,1578)` 内补入 29,716；其余恢复由直接外观验收支持。
D 两端原始 NCC 约 0.852/0.932，A 两端约 0.756/0.751，结构和边缘亦分别达标。
主模板、DAB 顺序、全局矩阵、基础标签/valid 均不变；另外三张最终标签无变化。
问题图 ECC **30→30 次**，另外三张亦未增加；没有通过更多 ECC 或 800 工作图换质量。

**不是零开销修复：**问题图增加约 **0.685 秒（7.3%）**。当前只证明这些输入的速度与
恢复情况，不承诺所有图片固定小于某秒数，不把旧环境的 5 秒或历史图库 15 秒当作本轮
对照。原图直接补全的离线计时不等于完整 Matcher 计时（后者设有既有 OpenCV 线程策略）。
本轮未读取目标人工标注，源轮廓连续性不是 GT 像素精度；图中其他不具备证据的未知区域
仍保留，不能据此宣称整图所有语义标签完整或全部 9,577 张都无误。

新增回归覆盖跨模板来源、部分主参考、组件隔离、错误身份屏障、局部孔洞、单侧损坏、
预算内外端点、不可见远距离原点、周期仿射/奇异拒绝、重复拟合/已完成短路、ROI 缩放和
坐标恢复，并把该真实图加入 640 串行轮廓连续性测试（无资产时跳过）。

最终 Python 全量回归 **245 项，244 通过，1 项已知失败**，无新增失败：
[既有包目录检查](../tests/test_package.py) 仍发现遗留内部 reference 目录，未删除无关目录
或放宽断言。新增 23 项补全锚点测试均通过，包含这张真实 TN8PR2 的完整串行推理；
真实 SD-1 旋转及既有白色遮挡、多源、部分参考回归也通过。
[完整回归记录](../periodic_template_matcher/results/support640_lattice_verified/regression.json)
保存失败详情与执行总数。

图库需重启加载新代码，并取消“跳过已完成”后强制重算问题图：**代码改动本身不改变旧
结果缓存签名**，历史结果、原业务输出和旧性能证据均未覆盖。

## 历史实验：直接将 max_side 从 800 降到 640（尚未修复上述缺口）

**当时结论：不能把未修复的 640 当作与 800 效果相当的直接替换；当时保留 800。
当前用户配置已是 640，且已实现上节修复，本节数值不代表修复后的输出。**
本轮只改变 `max_side`，ROI ECC、共享周期、候选、阈值及有效区不变；另统一
`template_workers=1` 做串行 AB/BA 三轮。对照工具见
[benchmark_resolution.py](../tools/benchmark_resolution.py)，不读取目标人工标注。

三张图在 640 下均通过匹配，ABC/DAB 身份顺序不变，两轴周期都验证通过，ROI 内
4,659,424 像素仍全部有预测，新增/丢失已知像素均为 0。但这不足以证明掩膜质量不变：

| 实图 | 类别 1 IoU（640 相对 800） | 类别 2 IoU（640 相对 800） | 不同标签像素 | 主模板是否切换 |
| --- | ---: | ---: | ---: | --- |
| CN01MA / 42 | 82.27% | 90.10% | 202,712 | 是 |
| CS01AA / 94 | 80.57% | 97.48% | 191,528 | 是 |
| CS026A / 90 | 99.21% | 95.10% | 21,401 | 否 |

前两张主模板从 SD1_1371308 切到 SD1_1258237，模板排名和最终源标注随分辨率改变；
对象选中的参考与周期出现集合也可能改变。640 的匹配分数更高，**不等于精度更高**。
上述 IoU 是输出一致性，不是 GT 精度，不能据此认定 800 一定更准，也不能宣称 640 无退化。
若只需要粗定位/身份顺序，这三张结果可用；若要求接近 800 的像素级标签，则证据不足。

暖机 `infer()`（不含初始化/哈希/导出）中位数：第二张 7.60→5.92 秒，第三张
7.32→6.14 秒，约降低 22%/16%。第一张两组交替实验都有显著负载扰动：第一组
800 为 7.84/15.94/14.54 秒，640 为 12.23/13.18/7.39 秒；重跑分别为
6.29/12.08/12.55 与 9.93/9.82/9.85 秒，**不据此声称稳定加速比例**。

本地证据在 `periodic_template_matcher/results/resolution_640/primary` 和
`first_recheck`，含双方配置、全尺寸标签/身份/有效区、数组哈希及对照报告。
首张可视化见
[800/640 标签对照图](../periodic_template_matcher/results/resolution_640/primary/comparison_target1.png)。
紫色仅表示新旧标签不同，不表示人工确认的错误。旧实验 JSON 中按周期出现序号算出的
局部角点位移不可解释为同一物体的实际漂移；分辨率改变会重排候选，现工具不报告该量。
新增对照工具 3 项测试、既有性能/共享周期 36 项测试合计 39 项通过。

后续若要兼顾 640 的速度与精细效果，建议单独验证“640 粗定位 + 较高分辨率局部精配/模板
复核”的分级路径，而不是降低验收阈值。上节最终修复未采用较高分辨率分级路径；
当前配置保持用户选择的 640。

## 联合利用类别周期与同身份周期

当前 SD-1 OpenCV 配置另行启用 `options.periodic.shared_periods: true`，通用 API 默认关闭。
本轮对照双方均已开启 ROI ECC，不能把此前整图 ECC 的耗时再次当作基线；仍然逐图串行，
不减少工作分辨率、外观候选、配置周期步数、70 次迭代预算或验收阈值。

### 周期信息怎样复用

- 类别 1 允许源图 x/y 两轴重复，使用整数格点 `H + n·target_period_x + m·target_period_y`。
	类别 2 只允许同一身份沿源图 y 重复；ABCD 是横向身份顺序，不是允许横向复制同一身份。
- 显式周期、类别锚点和图像自相关仍按原优先级给出初值。每轴联合验证模板自平移、目标图
	自平移，以及已匹配的同身份/同模板参考的多次出现。出现间距先折叠整数倍，仅作提案，
	不能单独认证周期。配置的源向量保持不动，估计向量及目标周期只做有界平移校准。
- 保存源向量与目标向量，按源坐标执行类别权限。源图 y 周期经过斜切后即使目标 x 分量
	不再接近零，也不会被误当作横向身份周期。转置布局同步交换向量坐标。
- 已联合验证的轴共用一个周期格点，不再逐个 x/y/对角副本做仿射 ECC；未验证的轴及包含
	该轴的组合仍回退原逐副本校验。全部副本仍需原图像评分及新增区域的局部证据。
- 对象定位中，同一身份、同一模板参考的周期别名可复用一个已验收几何修正，仍按每个出现
	的原始局部评分及位移阈值验收；不通过则执行原 ECC。不同身份或不同参考不互借纠偏。
- 对象支持补全按身份/模板/参考/周期方向缓存一次残差平移；失败也缓存，避免每个碎片重试。
	每个区域仍检查自身可见像素 NCC、结构和边缘，粘贴实际验收变换的掩膜；白色缺陷不提供
	外观证据，但有足够同区域可见证据时仍迁移真实源标签。完全无证据、失配边缘保持未知。
- 不再给不满足源轴权限、又不能提供任何未知标签的副本计算无用几何。基础标签已知时，
	仍保留对象补全需要的 y 几何；不是关闭对象补全换时间。

### 串行对照及变化

[共享周期对照工具](../tools/benchmark_periodic_sharing.py) 固定 `template_workers=1`，
同进程按 AB/BA 交替，各 3 轮，计时只含复用 Matcher 的 `infer()`，不含初始化/哈希/导出。
三张图全部验收，模板排名、全局注册矩阵、ABC/DAB 顺序及有效区覆盖率保持不变。

| 实图 | 独立副本中位数 | 共享周期中位数 | 延迟降低 | 已知标签不同像素 | 类别 2 IoU（相对旧输出） |
| --- | ---: | ---: | ---: | ---: | ---: |
| TNXFL0 / AFE55CN01MA / 42 | 6.504 s | 5.867 s | 9.8% | 4,957 | 98.6371% |
| TN4FL0 / AFE55CS01AA / 94 | 8.236 s | 7.190 s | 12.7% | 6,440 | 98.3172% |
| TN4FL0 / AFE55CS026A / 90 | 8.158 s | 6.838 s | 16.2% | 3,273 | 99.3438% |

首张第一组测量受系统负载干扰（独立 7.30/16.04/16.99 秒，共享 10.70/14.79/16.65 秒），
表中首张使用随后独立重跑的三轮，双方分别 6.63/6.50/6.40 与 5.87/5.87/5.74 秒。
第二张独立首轮也达 17.62 秒；应参考完整记录和中位数，不保证不同负载下固定延迟。

每张共同已知像素均为 4,659,424，validity 差异为 0；身份像素仍分别改变 3,531/4,466/1,907。
类别 1 的新旧输出 IoU 为 99.8361%/99.7734%/99.8430%。**这些均不是人工 GT 精度，不能
声称准确率提高或无退化**。首张同参考变换在完整目标画布角点的最大位移为 26.19 原图像素，
后两张局部对象变换未变；画布角点不是实际对象支持区，仍不得称全部变化都不足一个像素。

首张剖析从 40 次 ECC 降到 29 次，`_refine` 20→15 次；周期补全约 2.06→1.48 秒，
对象支持约 0.92→0.62 秒，联合周期校准本身约 0.28 秒。对象定位约 2.39→2.52 秒，
**尚没有加快全部 step3**：仍需逐身份外观搜索及无法复用的局部校正。
同次 `infer()` 约 6.99→6.29 秒；分项是剖析样本，不等同于上表的暖机计时。

另行实跑原配置的图库 `Jobs` 队列（两张串行、复用 Matcher、独立新结果目录），均成功，
界面口径约 **6.08/6.34 秒，不含 PNG 保存**。保留了原有模板级和 OpenCV 内部线程策略，
未新增图片级并行。

证据在 `periodic_template_matcher/results/periodic_sharing/`（本地 Git 忽略）：
`before_profile`、`after_profile`、`verified`、`verified_first_recheck`、`gallery_actual`。
对照工具保存配置快照、双方全尺寸输出、14 个数组哈希及标签/身份/变换差异。

新增共享周期契约 16 项全部通过；既有补全/部分周期/OpenCV/循环身份等 58 项另以默认共享
开启重跑也全部通过。最终全量回归 **219 项，218 通过，1 项失败**：仅有已知遗留内部目录
检查失败，未放宽断言或删除无关目录。完整测试记录见本轮验证输出。C++ 内核未实现共享
周期，不能宣称开启时 Python/C++ 等价。

修改后需重启图库；新配置进入缓存签名，原缓存与旧耗时不会原地更新。设
`shared_periods: false` 可恢复独立副本行为（ROI ECC 是另一个独立开关）。

## step2/step3 的局部 ROI ECC 加速（允许数值差异）

**本节与下文历史无损优化不同。用户明确同意采用局部 ROI ECC，允许浮点及标签差异。**
通用 `MatcherConfig.ecc_roi` 默认为 `false`；当前 SD-1 OpenCV 配置已显式设置
`options.step3.ecc_roi: true`，图库原启动入口即可使用。设回 `false` 可恢复整图 ECC。
修改配置后需重启图库。配置内容进入缓存签名，旧结果不会被覆盖或作为新配置缓存复用。

### 新剖析与根因

本轮以 Codex 更新后的实际代码及当前配置重新建立基线，没有用下文旧配置的 18 秒结果充当对照。
首张图的全局粗匹配约 0.59 秒，全局注册阶段约 1.56 秒；但包括局部对象和周期补全，
总共调用约 40 次 `findTransformECC`，约耗时 2.98 秒。历史慢图 AFE55CS026A、索引 90
在当前配置下共调用 45 次 ECC，约耗时 5.60 秒。小局部 ROI 原来仍在 614×800 或 307×400
整张画布迭代，才是本轮有价值的优化点。

当前每个模板只枚举 1 个角度、1 个尺度、3 个平移峰；`refine_candidates=8` 是上限，
不代表每模板实际执行 8 个全局 ECC。仅重写 step2 或缓存金字塔不能把整个流程变快数倍。

### 实现及约束

- step2 复用补零的目标掩膜；FFT 路径和晶格自相关复用同一调用内的正变换频谱。
	晶格的正 FFT 从 8 次降到 3 次，保持每次乘积、逆变换及 dtype 的原运算顺序。
- 对象局部 ECC 共享目标强度金字塔，**每个对象仍单独生成局部 support**。
	既有的位移拒绝判定提前到重复打分之前，不更改阈值。
- `ecc_roi=true` 时，仅当 support 包围盒加边距后小于整张画布 70% 且各边至少 32 像素才裁剪。
	原分辨率、半尺度/全尺度两层、70 次迭代上限、`epsilon=1e-5`、5×5 ECC 高斯参数均不变。
	没有减少候选、搜索角度/尺度、周期步数，也没有降低验收阈值。
- 窗口保留 24 个当前层像素的边距。ECC 的 target→warped-source 仿射通过
	`T @ W_local @ inv(T)` 换回完整工作图坐标，再沿用原有模板→目标组合公式。
- ROI 求解异常、非有限变换或最终映射越过人工窗口的过滤边距时，重试原整图 ECC。
	真实图像边缘不被当作人工窗口边缘。全图 NCC、边缘评分、奇异值和对象/周期位移验收仍执行。
- **这不是等价性证明**：最终边距检查不约束每次中间迭代，裁剪会改变归约顺序、边界及局部最优解。
	因此必须按近似算法验证，不可套用下文逐字节一致的结论。
- 不增加多图并行，不新增线程池；原有模板执行策略和 OpenCV 内部线程策略保留。
	C++ matcher 尚未实现此选项，不能假定开启后的 Python/C++ 结果一致。

仅保留前述无损缓存时，两张实图交替 3 轮只改善约 0.7%/1.5%，不作为明显加速宣传。
另行试过只让 ECC 用单个 OpenCV 线程（反而更慢），以及只禁用 ECC 的 IPP（无明显收益），均未采用。

### ROI 模式的实测效果与差异

可复现实验见 [tools/benchmark_ecc_roi.py](../tools/benchmark_ecc_roi.py)：
同进程、逐图串行、`template_workers=1`、全图/ROI 按 AB/BA 交替，各 3 轮。
**计时仅包括复用 Matcher 的 `infer()`，不含初始化、哈希和导出**；不是图库完整耗时。

| 实图 | 全图 ECC 中位数 | ROI ECC 中位数 | 已知标签不同像素 | 类别 2 IoU（相对旧输出） |
| --- | ---: | ---: | ---: | ---: |
| TNXFL0 / AFE55CN01MA / 42 | 8.624 s | 7.357 s | 226 | 99.9181% |
| TN4FL0 / AFE55CS01AA / 94 | 9.304 s | 8.164 s | 853 | 99.7651% |
| TN4FL0 / AFE55CS026A / 90 | 13.775 s | 8.438 s | 100 | 99.9676% |

前两张中位数分别改善约 14.7%/12.3%。第三张受显著系统扰动：三轮全图为
11.75/13.78/30.51 秒，ROI 为 8.27/8.44/17.50 秒，**不能将其中位数差外推为稳定 39% 收益**。
先前另一次稳定的两轮实验中，第三张为约 10.85→7.23 秒（约 33%）；ECC 自身约 5.35→1.73 秒，
调用数仍为 45。首张和第二张的 ECC 调用数也仍为 40/42，没有通过跳过调用来制造收益。

三张图的验收状态、模板排名、全局注册矩阵、ABC/DAB 身份顺序保持不变，
每张 4,659,424 个共同已知像素、有效区差异 0；身份像素仍分别有 211/618/94 处变化。
第二张局部对象变换的可见中心线采样位移最高约 2.82 个原图像素，标签变化最远距离旧边界约
5.79 像素，**不能把变化描述为全部逐位相同或必然不足一个像素**。
这些 IoU 是新旧输出对照，不是人工 GT 精度，也不证明所有 9,577 张图片都无退化。

另外直接运行图库的 `Jobs` 队列路径（原配置的模板策略，新的独立结果目录，无并发图片）：
前两张均成功，界面口径耗时约 **7.63/8.07 秒，不含 PNG 保存**。
不应把旧图库记录的 15 秒直接与这两个数相减当作本轮优化收益。

证据目录（本地 Git 忽略）：
`periodic_template_matcher/results/registration_speed_codex/` 下的 `before_profile`、
`after_profile`、`alternating_warm.json` 为无损缓存阶段；
`roi_alternating.json` 为先前两轮 ROI 测量；`roi_verified` 为完整三轮及双方导出结果；
`gallery_actual` 为真实队列结果。`--reuse-matcher` 已加入原单图基准，首轮仍包含初始化，
`--profile` 会在报告内记录 `_coarse_candidates`、`_refine`、`_score` 与 ECC 等分项。

最终 Python 全量回归 **203 项，202 通过，1 项失败**：已有的内部包目录检查发现遗留
`reference` 目录；未放宽断言或删除无关目录。当前配置下真实旋转回归已通过。
性能契约文件 20 项通过，覆盖局部 support 隔离、坐标换算、窗口逃逸/异常回退、
真实图边缘恢复、稠密/空掩膜回退、FFT 逐值一致、迭代参数不变及基准报告的重复参考完整性。
另外在测试进程中强制默认启用 ROI，对多源补全、对象补全、周期补全、部分周期、
OpenCV 后端及循环身份的既有 46 项测试重跑，全部通过；未修改这些测试的断言。

---

**下文为此前无损实现及历史审查记录，不能用来证明 ROI 模式逐位一致。**

## 后续优化与配置预加载

现在可以用 `PeriodicTemplateMatcher(config_path)` 直接初始化文件推理。初始化读取并校验模板、标注、有效区和默认目标，同时准备配准特征、周期晶格与 ECC 的目标金字塔。之后通过 `infer()` 获取结果，或用 `infer_to(output_dir, target=path)` 推理并导出。图库在同一队列内复用一个 Matcher，每张图保留自己的配置快照和结果。

配准与周期补全复用同一目标和有效区的 ECC 金字塔。变换相同的路径复用投影或分数，对象候选复用掩膜统计及相关搜索的补零输入。每个区域仍独立计分。阈值、候选数量、工作分辨率与 ECC 迭代配置沿用原值。

后续审查在启用 IPP 的 OpenCV 4.14.0 上复现了距离变换裁剪造成的舍入差异，并确认差异能进入 `edge_alignment`。当前实现已恢复整图距离变换并加入稀疏边缘回归用例。下文第 5 项及第 3.5 节记录的是前一轮实现。

相对本轮改动前的实现，两张实图的 14 个数组哈希和完整 `registration` 均一致。串行测量包含结果导出和哈希。

| 实图 | 本轮改动前中位数 | 当前含初始化 | 当前复用 Matcher | 每种模式轮数 |
| --- | ---: | ---: | ---: | ---: |
| TNXFL0，索引 42 | 13.14 s | 12.75 s | 12.02 s | 3 |
| TN4FL0，索引 94 | 12.93 s | 12.83 s | 11.89 s | 2 |

复用列排除一次性初始化。当时单独测得初始化约为 0.75–0.89 秒。TN4FL0 另一组测量的波动超过了冷启动收益，当前样本量只支持把约 8% 的延迟降低作为复用实例的实测参考。对照时将改动前后的完整 Python 包分别载入同一进程，并交替执行。两张图的最终代码另外通过了对应历史快照的严格对照。

## 范围

优化 Python matcher **单张图片的计算延迟**。没有新增多图并行、进程池或后台并发推理；图库仍逐图执行。没有调整 max_side、scale、匹配阈值、候选数量、周期步数或 ECC 迭代次数，也没有关闭对象/周期补全。

## 变更

1. 对象与周期补全的组件、局部重试片改为「坐标 + 紧凑掩膜」。不再为每个小区域反复创建、扫描原图尺寸的空白掩膜。
2. 缩放紧凑掩膜时从 OpenCV 获取原图到工作图的最近邻采样索引，保持完整画布的像素对齐；不直接缩放裁剪图。组件编号、遍历顺序、分块网格和局部验证规则保持不变。
3. 候选支持区投影延迟至对象布局选定后，避免给最终不会使用的候选创建整图支持掩膜。
4. 同一候选的源图灰度、边缘和有效区投影在局部打分间复用。每个局部区域仍独立重算目标有效区、NCC 和对称边缘距离，不能借用其他区域的匹配证据。
5. 对称距离变换限制到包含双方全部边缘的矩形；外部没有边缘零点或距离采样，不改变距离值及取样顺序。
6. 同一源图/目标对复用已有工作分辨率特征。若多模板共同倍率与局部配对倍率不同，保持原有配对分辨率，另行准备。原始缩放 ROI 与特征提取腐蚀后的安全掩膜分别保留，不可互换。

## 测量方法与结果

测量工具 `tools/benchmark_single_image.py` 使用独立配置快照，并将 `template_workers` 设为 1。
新报告的 `seconds` 涵盖加载、推理、验证哈希和非图片导出，PNG 编码与写盘单独记录在
`image_save_seconds`。`--profile` 分析加载、推理和验证哈希。输出目录必须为新目录。

本节历史表格的耗时包含 PNG 保存，与新报告的 `seconds` 口径有别。

当前配置目标（TNXFL0，文件索引 42）：同一进程内交替执行旧版本与优化版本，各 3 次。旧算法从 Git HEAD 读取后仅在测量进程内临时绑定，没有回退或改写工作区文件。

| 轮次 | 旧版本 / 秒 | 优化版本 / 秒 |
| --- | ---: | ---: |
| 1 | 18.423 | 13.505 |
| 2 | 18.956 | 13.391 |
| 3 | 18.315 | 13.496 |
| 中位数 | **18.423** | **13.496** |

单图延迟降低 **26.7%**，相当于约 **1.37 倍**速度。另一个真实样本（TN4FL0，AFE55CS01AA，文件索引 94）从 20.807 秒降至 13.422 秒，约降低 **35.5%**；该样本仅单次对照，不能当作整个数据集平均值。

旧图库中该样本曾耗时 48 秒，受当时运行环境影响，不能用来宣称这次优化降到 13 秒的收益。测量期间也观察到一次 20.5 秒的系统扰动，因此采用交替重复测量的中位数。不同图片、硬件和后台负载的实际耗时仍会变化，尚未达到秒级内推理。

所有上述比较均要求 **14 个 MatchResult 数组逐字节 SHA256 相同，以及完整 registration 诊断的 JSON 值相同**，包括变换、分数、身份与局部补全记录。实际性能样本都是匹配通过的图片，并未对拒绝图片进行真实耗时/哈希对照；拒绝相关行为依赖既有回归测试，仍需扩展独立验证。最终版本通过了两张真实图片及三轮交替对照；开发中的不一致候选版本未采用。

测量输出保存在 `periodic_template_matcher/results/single_image_speed/`（本地生成，Git 忽略）。用于结论的目录是 `before_serial`、`before_slow`、`after_exact_slow` 与 `alternating_before_0..2`、`alternating_after_0..2`；其他目录属于剖析或开发中间版本，不应当作发布结果。

## 回归与使用

- 精确采样、组件顺序、非连续掩膜、独立 ROI 分数、距离变换、投影次数和特征复用契约见 `tests/test_single_image_speed.py`。
- 最终完整 Python 回归共 177 项，175 项通过，无新增失败；新增 7 项无损优化契约均通过。
- 已有对象补全、白色遮挡、局部纠偏、跨模板补全、身份周期和 OpenCV 后端测试继续验证判定行为。
- 完整 Python 回归仍有两项此前独立复现的问题：遗留 `reference/__pycache__` 目录使包布局检查失败，当前 SD-1 目标 CCW 旋转被拒绝；本次不放宽这些断言。
- 原有 matcher CLI、数组 API 和图库自动使用这些核心优化，无需改业务配置。已运行的图库服务需重启后加载新模块；缓存中的历史结果和耗时不会自动变更。若需要测新耗时，取消「跳过已完成」并重跑少量图片。
- 当前剩余耗时主要是原有 ECC 与对象候选搜索；本次不以减少迭代、降低分辨率或删除验证换取速度。

## 1. 审查基线与文件范围

旧算法基线提交：**2954a2f7f18a3b6301e1ca9d4fc493e3c4c13f6e**。交替测量时使用的是该提交的四个数值内核，结合当时工作区的业务配置和原始图片。不要把旧提交里的默认 target 当成测量 target。

| 文件 | 关键符号 | 本次变化 |
| --- | --- | --- |
| [periodic_template_matcher/_internal/segmentation/regions.py](../periodic_template_matcher/_internal/segmentation/regions.py) | `MaskRegion`、`component_regions()`、`RegionResize` | 新增紧凑区域和完整画布的精确最近邻采样 |
| [periodic_template_matcher/_internal/objects/completion.py](../periodic_template_matcher/_internal/objects/completion.py) | `complete_object_support()` | 区域/重试片改为紧凑表示；缓存源灰度与同候选投影 |
| [periodic_template_matcher/_internal/objects/matching.py](../periodic_template_matcher/_internal/objects/matching.py) | `match_ordered_objects()`、`_refinement_features()`、`refine_object_transform()` | 延迟支持图投影；复用特征并保留原始 ROI |
| [periodic_template_matcher/_internal/registration/scoring.py](../periodic_template_matcher/_internal/registration/scoring.py) | `ScoreWarp`、`_edge_distances()`、`_score()` | 源投影惰性缓存与有界距离变换 |
| [periodic_template_matcher/_internal/segmentation/periodic.py](../periodic_template_matcher/_internal/segmentation/periodic.py) | `complete_periodic()` | 紧凑区域验证和候选局部投影复用 |
| [tests/test_single_image_speed.py](../tests/test_single_image_speed.py) | 三组测试类 | 7 项新增正确性/重复计算契约 |
| [tools/benchmark_single_image.py](../tools/benchmark_single_image.py) | `benchmark()` | 串行计时、结果快照、逐值诊断比较 |

本次没有修改 C++ 内核、模型阈值、业务配置或原图。工作区已有的 .gitignore、配置改动，以及图库本身的新增文件，不属于本轮数值优化；审查时请区分。新增文件尚未提交，仅运行普通 `git diff` 会漏掉它们，应结合 `git status --short`。

### 不增加并行的准确含义

- 没有引入多图片同时推理，图库 `Jobs` 仍在单个后台线程中按队列逐张运行。
- 没有引入进程池、新线程池或后台预计算。
- 性能对照把已有 `template_workers` 设置为 1，连模板配准也串行测量。
- **产品中既有的模板级并行配置和 OpenCV 内部线程策略并没有被删除或重写。** 原配置 `template_workers=0` 的既有行为仍可能使用模板级线程；不能把此次工作描述成“整个产品完全没有任何多线程”。

## 2. 热点定位依据

优化前对当前目标执行一次 `cProfile`，包含推理和导出，总计约 22.65 秒。该次剖析保留原配置的模板执行策略，仅用于找热点，不用于计算最终 26.7% 收益。

| 热点 | 累计耗时（约） | 可观察到的重复工作 |
| --- | ---: | --- |
| 对象支持补全 | 9.11 秒 | 多个组件/重试片各自持有原图大小的掩膜；反复全图布尔运算和求和 |
| 对象定位 | 6.10 秒 | 未选中的候选也提前投影 source support；局部配准重复提取特征 |
| 周期补全 | 3.77 秒 | 同一变换为多个局部区域重复投影、做距离变换 |

累计耗时含嵌套调用，不能把热点内部函数的累计耗时再次相加当作总时间。当前目标原图为 2056×2464；小区域若始终表示成整图掩膜，单个数组约有 506 万个像素。

## 3. 各项优化的等价性依据与审查风险

### 3.1 紧凑组件与局部重试片

旧方案对每个组件构造 `components == component_id` 的全图布尔数组，并反复对整图求和、筛选 evidence、生成重试片。

新方案一次连通域分析后保存组件的 `(y, x, mask)`。`mask` 只含其包围盒；通过 `slices` 读取同一坐标处的图像与有效区，通过 `approved[slices]` 写回完整画布。分块仍以组件的真实左上界为起点，末端裁掉的只是组件之外的零。

**必须保持：**

- 连通性和组件顺序不变。周期补全按原组件顺序处理，对象补全保留原 pending 栈的后进先出顺序。
- 组件内孔洞、对角连接、目标边缘截断不能被包围盒填满。
- evidence 的像素顺序仍是原图行优先顺序，不能为了性能重排 NCC 向量。
- 布尔切片赋值真正修改目标数组，而不是修改临时拷贝。
- 子块仍使用相同 tile_size、相同网格原点和相同接受/纠偏规则。

**复杂度依据：** 区域自身的扫描/分配从反复遍历整图变成遍历各自包围盒。一次全图连通域与最终全图输出仍然存在；工作分辨率的局部掩膜也仍然逐区域分配，不能声称所有全图分配都消失。

**待进一步审查：** `connectedComponents()` 改为 `connectedComponentsWithStats()` 后，当前环境/测试中的编号相同，但还应验证其他 OpenCV 版本、极碎掩膜及不同连通域实现的编号顺序。

### 3.2 完整画布的最近邻采样

直接缩放裁剪 mask 会改变像素对齐，不能用于替代旧实现。

`RegionResize` 先对原图 x/y 坐标序列执行 OpenCV `INTER_NEAREST` 缩放，获得输出每列、每行对应的原图索引。每个区域用 `searchsorted()` 定位落在包围盒内的输出范围，再按这些原图索引读取区域 mask；包围盒之外的工作画布保持零。后续膨胀和上下文打分仍在原工作画布上进行。

**审查重点：** 下采样时整个窄区域未被采到、上采样时重复索引、单像素区域、贴边区域、非连续切片，以及浮点坐标序列与 uint8 掩膜的最近邻采样是否在目标 OpenCV 平台一致。

### 3.3 延迟候选支持图投影

布局选择仍按原算法生成候选可见 mask、分数、覆盖率、边界和间距；它不使用候选的 `_support`。因此删除候选阶段的 `_support` 投影，仅在最终 selected 候选处计算支持区；若候选需要局部 ECC，则在 ECC 后用最终变换计算。

**审查重点：** 全部布局评分/排序/去重路径是否确实不读取 `_support`；完整周期模式与 partial 模式都要检查，避免把“未选中无需计算”误用到最终需要融合的同身份候选。

### 3.4 同候选源投影复用

`ScoreWarp` 缓存源归一化强度、源边缘、源 support 的投影；同一 `H`、同一 source、同一工作画布之间复用。强度和边缘为惰性求值：低重叠提前返回时不新增不必要投影。

仍然逐区域重新计算目标 ROI 与投影 source ROI 的交集、3×3 腐蚀、coverage、NCC、边缘距离和局部一致性。ECC 纠偏后的新变换不复用旧变换的投影。

**审查重点：** 缓存对象仅限当前候选；构造后 `H`/源特征没有被原地修改；注入的 work_source/work_support 与该 H 的投影完全相同。当前类没有对错误的外部缓存/H 组合做运行时校验，正确性依赖内部调用约束。

### 3.5 有界对称距离变换

输入 source/target edges 已经与当前 valid 相交。取双方边缘并集的包围盒，只有包围盒面积低于完整画布 75% 时才裁剪。框内包含全部零点（真实边缘）和全部需要查询的边缘像素；框外是距离变换的非零背景，没有新增可匹配边缘。

仍然使用 `DIST_L2`、maskSize=5，仍拼接 source→target 与 target→source 两个方向的距离，顺序不变。评分中的边缘指数、分位数、NCC 和全画布 4×4 tile 一致性未修改。

**审查重点：** OpenCV 近似距离变换在紧贴裁剪边界、极远稀疏边缘和不规则边缘上的逐值一致性；不能只用理想欧氏距离的几何论证代替实现验证。空边缘情况由 `_score()` 的双方至少 40 个边缘判断挡住，独立调用 `_edge_distances()` 的边界契约尚未扩展。

### 3.6 局部配准特征复用

对象定位已提取 source/target 特征。如果全模板共同工作倍率与该 source/target 对的历史配对倍率完全相同，则局部配准复用已有特征和坐标矩阵；否则按原配对倍率另行准备一次。

**关键陷阱：** `_features()` 返回的是腐蚀后的 safe mask，原始缩放 ROI 不能被它替代。最终实现单独缓存 raw resized source mask，并在 `_refinement_features()` 返回中保留 raw target mask。

开发中曾将 safe mask 错当成 raw ROI，导致三类输出数组改变；严格对照发现后已修正。失败版本的 `after_final` 数据不是最终通过证据。这也是 Codex 需要重点复查的地方。

**审查重点：** 多模板尺寸不同导致 global/pairwise scale 不同、原图灰度/RGB/BGRA、显式有效区、x/y 布局转置、数组是否被 ECC 或其他步骤原地修改。

## 4. 验证契约与证据边界

### 输出快照

工具在 `save_result()` 前遍历 MatchResult 数据类的 ndarray 字段，保存 shape、dtype 和按逻辑行优先字节序得到的 SHA256；并对 registration 做 JSON 序列化/反序列化后比较，避免 Python tuple 与磁盘 JSON list 的表示差异被误报。

实际通过样本记录的 14 项为：`input_image`、`template_image`、`warped_template`、`geometric_valid`、`mapped_labels`、`label_valid`、`completed_template`、`completed_valid`、`periodic_valid`、`label_periodic_valid`、`object_identities`、`object_valid`、`base_labels`、`base_valid`。

**边界：**

- 比较的是数组内容、shape/dtype 和完整 registration JSON 值，不是 PNG 文件的编码字节，也不是全部导出元数据字节。
- 当前 harness 不直接记录 `None` 数组字段，也没有在差异摘要中单独列出“仅旧快照有的新缺失字段”；总快照比较仍会失败。若审查拒绝/无标注场景，应检查这些字段和 prediction_status，而不是固定要求 14 个数组。
- 它不证明 source/template 被零副作用使用，不证明跨平台位级一致，也不代替 GT 精度评估。
- 实际对照只有两张 SD-1 实图，不能外推成对全部 9,577 张图的等价证明或平均性能提升。

### 新增测试与覆盖

| 测试 | 覆盖 |
| --- | --- |
| `test_nearest_resize_equals_full_canvas_at_all_boundaries` | 缩小/放大/原倍率、单像素、四边、完整区域 |
| `test_connected_components_preserve_ids_order_holes_and_diagonals` | 组件顺序、孔洞、对角连接、贴边组件 |
| `test_no_full_canvas_per_region_and_noncontiguous_tiles` | 小区域存储、非连续子块 |
| `test_bounded_edge_distances_match_full_frame_including_image_edges` | 原距离算法与裁剪算法逐值比较 |
| `test_reused_transform_has_exact_scores_with_independent_roi_evidence` | 缓存前后分数一致，ROI 独立，3 次 source warp |
| `test_low_overlap_does_not_compute_intensity_or_edges` | 低重叠惰性返回只做 support warp |
| `test_cached_pair_features_preserve_exact_ecc_result` | 缓存前后 ECC 矩阵一致；不重复提特征；raw ROI 与 safe mask 分离 |

这些测试是有限 fixture，不是对全输入空间的证明；尤其建议补充随机掩膜 property tests、不同模板尺寸及实际拒绝场景。

## 5. Codex 可复现的检查步骤

### 5.1 先查看差异，再运行契约测试

以下命令均从仓库根目录执行；测试/测量可能需要已有 SD-1 资产和当前虚拟环境。审查期间不要提交、清理或覆盖用户已有改动。

```powershell
git status --short
git diff 2954a2f7f18a3b6301e1ca9d4fc493e3c4c13f6e -- periodic_template_matcher/_internal/objects/completion.py periodic_template_matcher/_internal/objects/matching.py periodic_template_matcher/_internal/registration/scoring.py periodic_template_matcher/_internal/segmentation/periodic.py
.venv/Scripts/python.exe -m unittest tests.test_single_image_speed tests.test_object_completion tests.test_periodic_completion tests.test_ordered_objects tests.test_partial_instance_cycle tests.test_multisource_completion -v
```

请另行读取表中的未跟踪新增文件，普通 diff 不包含其正文。完整回归可运行 `-m unittest discover -s tests -t . -v`，核对新增失败而非把此前两项失败直接归因于本轮。

### 5.2 对当前实现做独立测量/旧快照对照

以下输出目录须未存在；若已存在，换一个新目录名。不要用业务 `sd1_opencv` 或已有 benchmark 目录作为 output。

```powershell
.venv/Scripts/python.exe tools/benchmark_single_image.py --output periodic_template_matcher/results/single_image_speed/codex_current --repeats 3 --compare periodic_template_matcher/results/single_image_speed/before_serial
.venv/Scripts/python.exe tools/benchmark_single_image.py --output periodic_template_matcher/results/single_image_speed/codex_profile --profile
```

只有当业务配置、图片内容、模板标注和运行库环境与保存 before 快照时相同，严格比较才有意义。先核对 before 的 config.snapshot.json；若资产/配置变了，应重新制作同输入的 old/new 基线，不能直接放宽数值容差。

`--target` 可指定其他图片，但不能把带单图 target_valid_mask 的配置盲目套用到新 target。本文的两张性能样本使用 target_valid_height，没有目标掩膜。

### 5.3 核对已有交替对照，或独立重建旧版本

已有证据：每个 `alternating_before_N`、`alternating_after_N` 目录都有 config.snapshot.json、benchmark.json 和 run1/result.json。先比较双方配置快照，再核对时间、数组哈希和全部诊断。

当时旧函数通过固定基线代码在临时 namespace 中执行，替换以下四个入口，测完恢复，不涉及修改业务数据：

- `periodic_template_matcher.matcher.match_ordered_objects`
- `periodic_template_matcher.matcher.complete_object_support`
- `periodic_template_matcher._internal.segmentation.transfer.complete_periodic`
- `periodic_template_matcher._internal.registration.core._score`

旧对象/周期/补全函数的 `_score` 引用也绑定旧 scorer，避免旧内核混入新投影/距离算法。所有 runs 的数组/诊断还与预先保存的 before_serial 快照一致。

**注意：** 自动交替 runner 当时是一次性的测量代码，目前仓库工具只提供单版本重复测量与快照比较，不能声称 CLI 已自动实现上述替换。Codex 可独立实现 runner，或在独立 Git worktree 中使用固定旧提交，复制 benchmark 工具后以同一个冻结配置快照和相同资产测量；不要在当前工作区强行 checkout/reset。

异机器的绝对路径会使 registration 的路径字符串不同，应区分路径迁移与数值差异；不能因此忽略全部诊断。应冻结统一输入路径，或明确仅规范化具体路径字段，保留所有算法诊断的严格比较。

### 5.4 本机验证环境事项

本机此前观察到 Python `platform._wmi_query` 卡住。已有测试/测量仅在验证进程内临时让该函数抛 OSError，以触发标准 Windows fallback；没有修改生产 matcher、系统设置或模型计算。tempfile 测试另有 Administrator 与 ADMINI~1 路径别名问题，完整回归 runner 使用解析后的 tempfile.tempdir。

若正常命令可运行，不必添加这些 workaround；如遇环境问题，应与数值算法失败分开记录。没有使用 profiler 的耗时才用于上表的收益结论。

## 6. 建议给 Codex 的审查任务

请独立执行以下审查，**先报告证据和问题，不自动放宽阈值、减少候选或删除验证**：

1. 逐项检查上述六类优化是否保持原判定、坐标、像素和处理顺序。
2. 检查候选缓存作用域、raw ROI/safe mask、同身份多候选融合与纠偏变换使用是否正确。
3. 对最近邻采样和有界距离变换补充随机及极端边界逐值对照。
4. 验证不同尺寸模板、y 轴布局、ROI 孔洞/边条、无标注、拒绝、完全遮挡、薄条和局部纠偏。
5. 检查测量工具是否存在统计偏差、输入不一致、哈希字段遗漏、配置误用或“旧函数实际引用新内核”的问题。
6. 将真实实图对照扩展到不同目录、接受/拒绝与耗时区间；单图串行交替测量，报告中位数和波动，不把全数据集并行吞吐量当作单图延迟。
7. 如发现不等价，应给出最小复现、变化的数组/诊断路径和原因；不得将不一致说成“可接受”而跳过用户确认。
8. 输出结论应区分：已验证等价、条件性等价、尚未覆盖、实际不等价、性能收益可信度，以及是否需要回滚某一项优化。
