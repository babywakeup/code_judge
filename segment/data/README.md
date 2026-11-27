# PSCD 数据集

只提供加载和在线预处理代码，不复制/下载原始图片；不依赖参考项目、`py_utils`
或其 `data_factory`。固定划分沿用参考项目：train 616、val 77、test 77 张原图。
划分 ID 紧凑存储在 [splits.py](splits.py)，训练集是 770 个原始 ID 中排除验证/测试后的集合。

## 本地目录

默认从仓库根目录的 `data_/PSCD` 查找；自动识别解压产生的下一层 `PSCD`。
也可以显式传入其他路径。数据根目录中应有 `t0`、`t1`、`mask_t0`、`mask_t1`，
同一样本使用相同 PNG 文件名。代码只要求当前选择的掩码目录存在。

## 使用

```python
from torch.utils.data import DataLoader
from segment.data import CroppedPSCD, DiffViewPSCD, PSCD

train = CroppedPSCD(mode="train", hflip_prob=0.5)
loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=4)
batch = next(iter(loader))
# pixel_values / pixel_values_pair: float32 [B,3,224,224]，范围 [0,1]
# labels: int64 [B,224,224]，0=未变化，1=变化
outputs = model(**batch)  # model 是 SceneChangeForSemanticSegmentation
outputs.loss.backward()

validation = CroppedPSCD(mode="val")
shifted = DiffViewPSCD(mode="test", adjacent_distance=-1)
full = PSCD(mode="test")
before, after, labels = full.get_raw_sample(0)  # uint8 HWC，uint8 HWC，int64 HW
```

- `PSCD`：整图；processor 默认中心裁剪到 14 的整数倍，不缩放整图。
- `CroppedPSCD`：每张原图默认生成 15 个 224×224 重叠块，横向步长 57、前缀 1。
  对应长度 train=9240、val=test=1155。裁剪名称如 `00000000.3.png` 只是样本标识，
  不会生成这些文件。
- `DiffViewPSCD`：将 t0 第 i 块与 t1 第 i+distance 块配对；默认 ±1 位移 57 像素，
  每张原图保留 14 对。非重叠区域的 t0 标签置 0，复现参考策略，不设为 ignore。
  仅支持 t0 监督；普通数据集可用 `mask_source="t1"` 切换监督。
- `processor=` 可传自定义 `SceneChangeProcessor`；默认不做 ImageNet 标准化。
  mask 已二值化成 0/1，不需要额外的 0/255 映射。
- `filenames=["00000000.png", ...]` 可明确覆盖固定划分，用于本地子集和测试。
- 同步水平翻转使用 PyTorch 随机数；验证/测试默认关闭，训练也须显式设置概率。

## 原始尺寸与参考协议的区别

本地下载数据的原图实际为 **4096×1152**；参考加载器硬编码 **1024×224**，
但没有提供把高分辨率图片转换到该尺寸的步骤。

本实现的裁剪版新增 `resize_panorama=True`（默认）：先将双图和标签同步缩放到
**1024×224**，再滑窗裁剪。RGB 用 PIL 双线性插值，二值标签用最近邻，保持 0/1。
这会改变原图宽高比，是当前数据的适配策略，**不声称与作者未公开的预处理一致**。
已有 1024×224 数据不会重复缩放；设置 `resize_panorama=False` 可严格要求输入该尺寸。
整图 `PSCD` 及它的 `get_raw_sample()` 不应用这一缩放。

processor 默认不做 mean/std 标准化；每个样本输出无 batch 轴的 Tensor 字典，
可直接由默认 DataLoader collate 成模型输入。整图尺寸较大，注意显存消耗。

## 可视化

[PSCD Viewer](../../apps/pscd_viewer/README.md) 可并排查看实际处理后的 Before / After /
Label RGB、监督坐标标签叠加，以及整图/裁剪/错位模式。仓库根目录启动：

```bash
python -m apps.pscd_viewer --no-browser
```

打开 http://127.0.0.1:7877 。默认使用当前本地 PSCD 数据，不加载模型或生成预测。

## 测试

仓库根目录运行（无下载，使用合成图片）：

```bash
python -m unittest segment.tests.test_pscd -v
```

## 来源与条款

固定划分与裁剪/错位协议来自 Chad Lin 的 MIT 项目
https://github.com/ChadLin9596/Robust-Scene-Change-Detection ，代码许可证见
[../LICENSE](../LICENSE) 和 [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
PSCD 原始数据另有使用条款：仅用于学术研究，未经许可不得再分发，论文使用须引用
Sakurada et al., *Weakly Supervised Silhouette-based Semantic Scene Change Detection* (2020)。
详见 https://kensakurada.github.io/pscd/term_of_use.html 及下载数据中的条款。

## VL-CMU-CD 数据集

`VLCMUCD` 和 `DiffViewVLCMUCD` 加载本地 VL-CMU-CD-binary255 数据，不会下载或复制原图。
默认识别仓库内的 `data_/VL-CMU-CD2/VL-CMU-CD-binary255`，也可传入该目录或其父目录。
数据需要采用以下结构，且 `t0`、`t1`、`mask` 中同名 PNG 一一配对：

```text
VL-CMU-CD-binary255/
├── train/{t0,t1,mask}/
└── test/{t0,t1,mask}/
```

```python
from torch.utils.data import DataLoader
from segment.data import DiffViewVLCMUCD, VLCMUCD

train = VLCMUCD(mode="train")
validation = VLCMUCD(mode="val")
test = VLCMUCD(mode="test")
batch = next(iter(DataLoader(train, batch_size=4, shuffle=True)))
# pixel_values / pixel_values_pair: float32 [B,3,504,504] by default
# labels: int64 [B,504,504], 0=未变化，1=变化
outputs = model(**batch)

shifted = DiffViewVLCMUCD(mode="train", adjacent_distance=1)
```

- train/val 从 `train` 目录读取，并按参考项目固定的 group ID 划分（train 87 组、val 11 组）；test 使用 `test` 目录的所有样本。索引 ID 已随本模块存储，不依赖运行时存在 `参考资料/`。
- 默认 `SceneChangeProcessor` 将 512×512 RGB 中心裁剪至 patch-size 14 的倍数，即 504×504；不改变比例、不做 ImageNet 标准化。`get_raw_sample(index)` 返回原始 uint8 RGB 与 int64 二值标签。
- 标签采用 `mask > 0` 二值化。binary255 发布版的灰度标签含 0 到 255 之间的中间值；所有非零值都映射到变化类 1，与参考加载器行为一致。
- `DiffViewVLCMUCD` 按参考协议在排序后的索引上移位：train/val 步长 4，test 步长 1；只保留 group ID 与视角角度都相同的图像对，标签来自 t0。`adjacent_distance` 可为任意非零整数；若配置与样本排序不匹配、没有可用配对，会报错。
- `filenames=[...]` 可覆盖固定 split，用于选取该目录下的 PNG 子集。`hflip_prob` 默认关闭；需要数据增强时可在训练集显式设置。

离线合成数据测试：

```bash
python -m unittest segment.tests.test_vl_cmu_cd -v
```

VL-CMU-CD 的 split/错位加载协议参考 Chad Lin 的 MIT 项目
https://github.com/ChadLin9596/Robust-Scene-Change-Detection 。数据文件的使用条件以其来源页面和数据发布许可为准。
