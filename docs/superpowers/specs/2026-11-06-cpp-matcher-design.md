# periodic_template_matcher C++ 迁移设计（待确认）

## 目标和已确认范围

用户已选择“完整 C++ 算法内核 + Python 外层”，要求功能与当前 Python 版本对应、能够断点调试，并持续说明准备工作、实现结果以及 C++ 所必需的内容。

在新目录 `periodic_template_matcher_cpp/` 实现原生算法；原 Python 代码保留为基准。Python 继续负责 JSON/LabelMe 读取、图像文件输入、结果导出及查看器。C++ 不回调 Python 算法；未迁移功能必须明确报错，不能悄悄调用原版并声称已完成迁移。

“对应”包含数组 API 的校验语义、七步执行顺序、全部配置控制的算法分支、接受/拒绝状态、结果字段与评价。现有 Tk 查看器和文件格式不重写。线程调度不要求实现相同，但 worker 上限与稳定排序语义需要保持。

## 方案选择

1. 完整 C++ 算法内核 + Python IO/查看器：用户已选；可在同一批数组上对照，降低外围变动。
2. 全部独立 C++：不选；需额外重写配置、LabelMe、导出和 UI。
3. 只替换几个热点：不选；不能满足本次完整功能对应的目标。

绑定方式由实现方建议采用 pybind11：数组、可选结果及嵌套诊断较多，相比手写 ctypes 更容易表达生命周期和异常。C++ 核心本身不依赖 Python，pybind11 仅位于边界层。教学目录 `c++_speed_test` 不替换。

## 当前准备情况和环境事实

- 已阅读公共入口、配置、数据契约、流水线、配准内核及相关测试，列出模块对应关系。
- 当前工作区包含大量既有未提交/暂存变更；不得清理、覆盖或把它们混入迁移提交。迁移基准应记录当前文件内容哈希，不能只记录 Git HEAD。
- 本机 Python 3.12.13 x64、NumPy 2.5.4、cv2 4.14.0；requirements 中声明 OpenCV 4.13.0.92，与实际环境不同。基准报告记录实际版本，不先改动用户环境掩盖差异。
- 已有局部 Zig 编译器可编译纯 C++ 示例；在 PATH 中尚未发现 CMake、MSVC、GDB/LLDB。现有 `.vscode/launch.json` 只配置 Python 调试。
- 尚未确认可用的 OpenCV C++ 开发包。Python wheel 的存在不能视为 C++ 头文件/链接库已经就绪。
- 已执行 `.venv/Scripts/python.exe -m unittest discover -s tests -t .`：161 项测试，87.095 秒，全部通过，退出码 0。环境与源码哈希记录在同目录 `2026-11-06-cpp-matcher-baseline.json`。当前未实现新的业务 C++ 模块。

## 结构与 Python 对应

```text
periodic_template_matcher_cpp/
  CMakeLists.txt
  CMakePresets.json
  include/ptm/           # 配置、输入/输出及核心 API
  src/
    config.cpp
    pipeline.cpp
    preparation.cpp
    registration/       # core、correlation、opencv、scoring
    objects/            # appearance、layout、matching、axes、completion、fusion
    segmentation/       # transfer、periodic、evaluation
    utils/              # images、masks、regions 的数值功能
  bindings/             # pybind11 数组转换及结果包装
  python/               # Python facade 和现有 IO 的适配
  debug/                # 不依赖 Python 的 C++ 调试入口
  tests/                # C++ 单元测试、跨语言对照及 fixture 工具
  docs/                 # 构建、断点操作、模块对照、迁移进度
```

具体文件数量可随实现调整；不为完全相同的目录外形增加空模块。

| Python 职责 | C++ 对应功能 | 验收重点 |
| --- | --- | --- |
| config / preparation / ValidRegion | 配置结构、校验、ROI 与标注知识归一化 | 灰度/BGR/BGRA，uint16 标签，非法输入，原数组不变 |
| registration/core | 统一尺度、特征、角度/缩放候选、多模板排名 | 原图坐标，稳定同分顺序，共享目标预处理 |
| correlation / opencv | FFT 与 OpenCV 部分重叠搜索 | 正负位移、交集归一化、多峰抑制、周期诊断 |
| scoring | NCC、边缘、局部一致性、受限 ECC | 分数公式、ECC 方向及失败处理 |
| objects/appearance / matching / layout / axes | 外观候选、循环顺序 DP、多参考与转置 | 完整/局部周期、边缘截断、身份编码、x/y 排列 |
| segmentation/transfer / periodic | 标签投影、分类别周期权限、逐副本验证 | 最近邻标签，无证据区域保持未知 |
| objects/completion | 实例支持补全、白色缺陷规则 | 不沿身份排列轴复制，几何批准与标签批准分离 |
| objects/fusion / pipeline | 融合、快照、提前返回和 objects-only 分支 | 原有优先级、accepted_objects_only、诊断字段 |
| segmentation/evaluation | 评价 ROI、混淆计数、IoU/Dice/总分 | GT 不进入匹配，拒绝/无标注返回无分数 |

## 接口和数据所有权

- C++ 核心接收明确的 Config、图像/掩码、对象参考和 ROI 结构，返回 MatchResult/EvaluationResult；输入按只读处理。
- Python 包装保留 `PeriodicTemplateMatcher.match/evaluate/save_result` 的调用形状，并转换成现有 MatchResult/EvaluationResult 数据契约，以复用导出和查看器。
- Python 绑定不向外暴露 STL/OpenCV 内存布局；使用 pybind11 转换 NumPy 数组，入口校验 dtype、形状、步长和配对数量。非连续输入规范化后处理。
- 同步调用期间保持数组所有者存活；必要时复制源数组，C++ 不修改调用者输入。
- 输出数组须拥有独立存储，或由明确的 owner/capsule 管理；核心局部 cv::Mat 销毁后不能遗留悬空 NumPy 视图。
- 进入纯 C++ 密集计算后释放 GIL；线程不访问 Python 对象，不在各 worker 中修改 OpenCV 的进程级线程设置。
- 异常在绑定边界转换；错误中包含阶段名。无法提供结果时保留与原版一致的 None/状态语义。
- 不省略拒绝结果的几何输出，不将未知标注当成背景，也不将对象局部通过当成全局语义通过。

## 数值对应规则

- 保留原公式、阈值、候选遍历顺序、稳定排序、坐标方向及插值规则。
- NumPy FFT 以双精度工作；C++ FFT 使用 OpenCV 双精度 DFT，对相关方向、补零尺寸和位移索引逐项对照。
- 显式实现 NumPy percentile 的线性插值语义、Python round 的平分取偶语义，以及 float32/float64 的关键转换；不直接用近似不同的替代函数。
- 矩阵统一表示“模板原图 → 目标原图”。周期向量只乘线性部分，不加平移。
- 匹配分数和诊断数值允许受库版本/浮点误差影响；必须报告偏差及是否改变分支，不能自动放宽接受阈值。

## 构建与依赖（提议）

- C++17，CMake，OpenCV C++，pybind11；诊断/调试 fixture 使用 nlohmann/json。
- OpenCV 核心模块至少包括 core、imgproc、photo（inpaint）、video（ECC）；独立调试入口读图再使用 imgcodecs。模块最终以实际移植调用为准。
- 优先为 Windows x64 配置 MSVC 与同工具链构建的 OpenCV；Linux 配置 GCC/Clang 与 Linux OpenCV。不得假设已有 Zig 能链接任意 MSVC C++ 二进制。
- 首先探测本机已有工具与开发包，再决定本地构建依赖或安装 SDK。实际安装情况在进度文档记录，不能标记为已准备完毕。
- 运行时产物按 OS/架构分开，Windows 的 .pyd/.dll 不用于 Linux；发布依赖 DLL/.so，开发头文件和编译器不随部署必须提供。

## 可调试意味着什么

1. 提供与 Python 无关的 C++ 调试可执行文件，读取由 Python IO 准备的同一份 fixture，调用同一核心库。
2. Debug 构建关闭核心优化、生成源码调试符号；可在特征提取、候选搜索、ECC、周期验证和融合处断点，查看 cv::Mat、矩阵、候选分数。
3. Python 调用版本提供带符号的构建；Windows 默认与普通发行版 Python 兼容的运行库，不要求用户安装 Debug Python。可附加原生调试器检查绑定调用。
4. VS Code 提供可复用的构建/启动配置与断点操作说明；保留原有 Python launch 配置，不覆盖。未安装原生调试器时必须明确提示，不能仅凭能编译宣称可以调试。
5. Debug 用于单步，Release/RelWithDebInfo 用于性能测量；Debug 耗时不作为加速依据。

## 分阶段实施及完成定义

1. 基线与构建：记录当前源文件哈希、库版本、原版测试；探测并准备原生开发环境；编译最小 OpenCV 核心调用，确认调试符号及断点。
2. 基础与配准：输入/配置、特征、FFT/OpenCV 搜索、ECC、多模板决策和直接标签迁移；逐层比较。
3. 完整语义：周期补全、实例外观与顺序、多模板局部周期、支持补全、融合和评价；覆盖原有关键测试。
4. 集成：Python API/IO/查看器复用、独立 C++ 调试入口、部署说明、完整差异报告。

每阶段交付状态明确：准备完成、已实现未验证、已实现并验证、未实现。即使第二阶段能运行，也不能声称完整 C++ 复现已完成。

## 验证标准

- 原版测试先跑一遍；既有失败如实记录，迁移不得通过改 Python 阈值来掩盖。
- 小型确定性 fixture：数组维度、掩码、取整/分位数、相关图、NCC、评价计数等按类型给出精确或严格数值对照。
- 默认目标：相关/分数 `atol=1e-4, rtol=1e-5`；矩阵比较以原图控制点位移不超过 0.1 像素为初始目标，并分别记录 ECC 的偏差。该容差是实施起点，任何调整必须有证据。
- 接受状态、模板选择、身份名称/编码和配置分支在确定性用例中要求一致；分数接近阈值的用例单独显示中间量。
- 整数标签、有效掩码、身份图在确定性/固定变换用例中逐像素一致；真实图像如因浮点边界不同出现差异，报告像素数和位置，不能仅凭 IoU 接近声称完全一致。
- 核心覆盖：平移/旋转/缩放、同分模板、纹理不足、无标注、ROI、FFT/OpenCV、完整/局部身份周期、多模板参考、横纵排列、白色缺陷、objects-only、周期副本拒绝、GT 隔离及输入不变。
- 有真实 SD-1 文件时运行原配置和旋转用例；无数据时明确跳过原因，不把跳过当通过。
- 实际执行 Python 导入调用、独立 C++ 运行和至少一次源码断点命中，才能分别声明接口、原生运行和调试验证成功。
- 全部功能对照通过后，再用 Release 构建测量预处理、配准、实例、补全及端到端耗时，不预先承诺加速倍数。

## 给用户的必需性说明

| 内容 | 为什么需要 | 是否 C++ 本身强制 |
| --- | --- | --- |
| C++ 编译器/链接器 | 将源码变成原生库/程序 | 必需 |
| OpenCV C++ 头文件和链接库 | 当前算法使用 OpenCV | 本项目必需；普通 C++ 不强制 |
| 运行依赖的 DLL/.so | 动态链接版本执行所需代码 | 取决于链接方式 |
| 导出/绑定接口、正确数据类型和生命周期 | 让 Python 安全调用 C++ | 跨语言调用必需；实现形式可选 |
| 调试器、符号与源码 | 实现断点、单步、变量检查 | 本次 debug 要求必需 |
| CMake | 管理多个平台、模块和构建配置 | 工具选择，不是语言强制 |
| pybind11 | 简化复杂数组/对象绑定 | 工具选择；可用 ctypes 等替代 |
| 对照测试/基线 | 证明复现功能对应 | 本次验收必需，不是语言强制 |

## 待确认

本设计沿用用户已经选择的范围。确认后进入实施计划和代码实现；如用户要求直接执行，则按本设计记录的实现方建议推进，并在实际限制出现时及时说明。
