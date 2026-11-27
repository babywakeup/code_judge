# Transformers 场景变化分割模型

将参考项目 **Robust-Scene-Change-Detection** 的主模型迁移为原生 Hugging Face
模型，不依赖参考项目的 `py_utils`、`torch.hub`、DataParallel 或运行时下载源码。

## 支持范围

模型按 Transformers CLIP 的具名模型家族方式组织，当前实现位于
[models/scene_change](models/scene_change/__init__.py)：配置、模型、双图处理器和可选 PEFT helper。
顶层只保留包初始化和 [demo.py](demo.py) 两个 Python 文件，不再保留重复的兼容转发文件。
模型类仍可通过 `from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation` 使用；
具名模型也可直接从 `segment.models.scene_change` 导入；旧的 `segment.model` 源码目录已移除。

目录结构：

```text
segment/
├── __init__.py            # 模型导出与 HF AutoClass 注册
├── demo.py                # 薄入口：参数解析与训练/smoke 分发
├── models/
│   ├── assets.py          # 模型家族共用的本地骨干路径解析
│   └── scene_change/
│       ├── __init__.py
│       ├── configuration_scene_change.py
│       ├── modeling_scene_change.py
│       ├── image_processing_scene_change.py
│       └── peft_utils.py
├── data/                  # 数据集与预处理适配
├── training/
│   ├── config.py          # CLI 参数、模型与数据集构建
│   ├── trainer.py         # SceneChangeTrainer(Trainer)、流式分割指标
│   └── runner.py          # TrainingArguments 配置、训练/验证/保存编排
├── tools/                 # checkpoint 转换与随机输入 smoke 检查
└── tests/                 # 离线协议、模型及 Trainer 测试
```

- `SceneChangeConfig(PreTrainedConfig)`：采用 CLIP 同样的 `@strict` 声明式字段，
  `__post_init__` 整理默认值与嵌套骨干，`validate_architecture` 集中检查配置约束。
- `SceneChangePreTrainedModel`：HF 元信息、参数初始化、骨干冻结与加载后的冻结恢复。
- `SceneChangeForSemanticSegmentation(SceneChangePreTrainedModel)`：特征、交互、解码与输出编排；
  损失独立为 `scene_change_loss`，字典/tuple 输出统一使用 `ModelOutput`。
- `SceneChangeImageProcessor(BaseImageProcessor)`，别名 `SceneChangeProcessor`：双图同步预处理、
  标签同步裁剪、保存加载、预测后处理。单一图像模态，不需要 tokenizer / 多模态 `ProcessorMixin`。
- `AutoConfig`、`AutoModelForSemanticSegmentation`、`AutoImageProcessor`、`AutoProcessor`。
- PEFT LoRA：交互层 Q/K/V/output 投影，及可选 DINOv2 骨干 Q/K/V 投影；卷积分割头完整训练并随 adapter 保存。
- 原项目 `dino2 + cross_attention` 与 `dino2 + single_cross_attention1` 的本地 `.pth` 转换。
  DINOv2 small/base/large/giant 架构配置与标准非 register-token 权重键映射。

**没有迁移** ResNet、CoAttention、MTF、TemporalAttention 等消融模型；不支持 DINOv2 register-token 变体。
当前不是 Transformers 内置单图 segmentation pipeline：推理必须同时传入两张图。

这里借鉴 CLIP 的模块边界，而非改成 CLIP 的图文架构：不额外引入 tokenizer、重复的
`ProcessorMixin` 或依赖 Transformers 私有懒加载机制的小型包装层。
原 `model_type="scene_change"` 和参数键名保持不变，已有完整模型与 PEFT 权重仍按原语义加载。

## PSCD 数据加载

新增独立的 [data](data/README.md) 子包，提供 `PSCD`（整图）、`CroppedPSCD`（默认
15 个重叠块）和 `DiffViewPSCD`（错位视角），沿用参考固定划分，不复制原始图片。
默认查找仓库下的 `data_/PSCD`，输出可直接通过 `DataLoader` 送入模型。
原始高分辨率数据到参考裁剪尺寸的适配策略与使用示例见上述文档。

### 环境与验证

需要 Python 3.12+；依赖版本范围见 [requirements.txt](requirements.txt)。LoRA 需要 `peft`，
不使用 LoRA 时可不安装它。本实现面向 Transformers 5.x，不保证兼容 4.x。

在仓库根目录安装依赖并运行离线测试：

```bash
python -m pip install -r segment/requirements.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest segment.tests.test_scene_change -v
```

## 架构与兼容语义

```text
image A ── shared HF Dinov2Model ── normalized patch Q/K/V/token ── branch A ┐
                                                                        ├─ concat ─ 3×3 conv + ReLU ─ 1×1 classifier ─ bilinear upsample
image B ── shared HF Dinov2Model ── normalized patch Q/K/V/token ── branch B ┘
```

交互层保留参考模型的 post-norm residual、4 倍宽度 ReLU FFN 和 dropout；
把 `nn.MultiheadAttention.in_proj_weight` 拆为 `q_proj`、`k_proj`、`v_proj`，
以便 PEFT 直接注入，不使用副作用 forward hooks，也不会在插入 LoRA 后因为旧的冻结缓存截断梯度。

`attention_mode` 明确区分：

| 值 | 行为 |
|---|---|
| `cross`（新配置默认） | A 查询 B 的 K/V；B 查询 A 的 K/V；每层两分支使用更新前的输入 |
| `legacy_self` | 精确保留原双分支调用中 **K/V 两次交换后实际执行自注意力** 的行为，转换旧权重自动选择 |
| `single_cross` | 原单分支：A 查询 B，仅 A 的结果进入解码器；保留无 padding 的 3×3 卷积 |

另外保留参考模型分类卷积后的 ReLU（`classifier_relu=True`）。新训练如需不受限的 logits，
可显式设为 `False`；对转换后的旧模型不要随意改变。
`backbone_interpolation="reference"` 保留原 torch.hub DINOv2 bicubic 插值的 0.1 offset；
新模型默认 `"hf"`，采用当前 HF 的目标尺寸插值。转换旧权重会设置前者。

双图输入尺寸必须相同，为 `[B,3,H,W]`，H/W 是 patch size 的正整数倍。
默认输出跟随处理后的输入；`output_size=[H,W]` 可固定输出。
**logits 为 `[B,num_labels,H,W]`**，而不是原项目的 `[B,H,W,num_labels]`。
`output_hidden_states=True` 返回交互前的两个归一化 patch 特征 `[B,N,C]`，不返回全部骨干层。

调试特征时，可直接调用 `model.get_image_features(pixel_values, layer_index=None, facet=None)`，
默认使用第一个分支的层和 facet，不执行交互层或分割头。输入校验、解码和损失分别在
`_validate_inputs`、`_decode_features`、`scene_change_loss`，主 `forward` 只保留清晰的数据流。

## 1. 从预训练骨干初始化

以下示例只读取已有本地 DINOv2 权重，**变化分割头仍是随机初始化，必须训练**。

```python
from segment import SceneChangeForSemanticSegmentation, SceneChangeProcessor

model = SceneChangeForSemanticSegmentation.from_backbone_pretrained(
    "../../hf_cache/facebook/dinov2-small",
    feature_layer=11,
    facet="query",
    facet_pair="query",
    num_attention_heads=1,
    num_attention_blocks=1,
    attention_mode="cross",
    classifier_relu=False,
    freeze_backbone=True,
    backbone_load_kwargs={"local_files_only": True},
)
processor = SceneChangeProcessor(size=504)

model.save_pretrained("outputs/change_base")
processor.save_pretrained("outputs/change_base")
```

### 配置驱动的 DINOv2 / DINOv3 ViT 初始化

`backbone_config` 通过 `AutoConfig` 根据 `model_type` 还原，不再固定为 DINOv2。
支持标准 DINOv2 small/base/large/giant，以及 DINOv3 ViT（包括 register tokens 和 RoPE）。
不支持 DINOv2 register-token 变体、DINOv3 ConvNeXt 或带下游任务头的骨干权重。

```python
from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation

config = SceneChangeConfig(
    backbone_name_or_path="../../hf_cache/facebook/dinov3-vitb16-pretrain-lvd1689m",
    use_pretrained_backbone=True,
    classifier_relu=False,
)
model = SceneChangeForSemanticSegmentation.from_config(config)
```

- `use_pretrained_backbone=True`：直接调用 `AutoModel.from_pretrained` 读取本地骨干权重，
  不再先构建另一份随机骨干再复制。交互层和变化分割头仍随机初始化。
- `use_pretrained_backbone=False`（默认）：使用 `AutoModel.from_config` 随机初始化骨干；
  若只提供来源目录，仅读取其中的结构配置，不读取权重。
- 显式提供 `backbone_config` 且关闭预训练开关时，不需要来源目录。
- 直接构造 `SceneChangeForSemanticSegmentation(config)` 和 HF 的
  `AutoModelForSemanticSegmentation.from_config(config)` 遵循同一初始化开关。
- 所有来源加载均限制 `local_files_only=True`，缺失目录或不完整骨干权重会报错，不下载、不静默回退。

`from_pretrained` 用于加载已经保存的**完整变化模型**：即使保存配置的预训练开关开启，
恢复也不会再次读取原始骨干目录，完整 checkpoint 可独立使用。
不要把 DINOv2/DINOv3 骨干目录传给完整变化模型的 `from_pretrained`。

DINOv3 返回的特征去除 CLS 和全部 register tokens。`query/key/value` 表示原生线性投影结果，
其中 Q/K 在 RoPE 之前提取；前面的完整编码层正常执行原生 RoPE 注意力。
`token` 表示所选编码层的输出，不额外应用骨干最终的 LayerNorm。
`backbone_interpolation="reference"` 仅适用于 DINOv2。

本轮适配模型加载与计算接口，不改变训练数据预处理策略。使用真实 DINOv3 数据训练时，
还须按其预处理配置启用对应的归一化；当前训练 Demo 不会自动继承骨干的预处理配置。

仓库根目录的 [test_backbone_loading.py](../test_backbone_loading.py) 默认对本地 DINOv2-small、
DINOv2-base 和 DINOv3-ViT-B/16 分别检查预训练/随机初始化、四种特征、矩形网格与有限损失：

```bash
python test_backbone_loading.py
python test_backbone_loading.py --roundtrip
python test_backbone_loading.py --backbone ../../hf_cache/facebook/dinov2-large --mode pretrained
```

预训练检查逐个精确对比本地 safetensors 参数，不额外加载第二份骨干。
`--roundtrip` 额外保存并重新加载完整任务模型，需要临时磁盘空间；默认仅使用 CPU。

## 2. 保存后的完整模型加载与推理

```python
import torch
from PIL import Image
import segment  # 注册当前进程中的 Auto 类
from transformers import AutoModelForSemanticSegmentation, AutoProcessor

model = AutoModelForSemanticSegmentation.from_pretrained(
    "outputs/change_base", local_files_only=True,
).eval()
processor = AutoProcessor.from_pretrained("outputs/change_base")
image_before = Image.open("before.png").convert("RGB")
image_after = Image.open("after.png").convert("RGB")
inputs = processor(images=image_before, images_pair=image_after, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
masks = processor.post_process_semantic_segmentation(outputs)
mask = masks[0]  # H,W int64：0 未变化，1 变化
```

也可以直接使用三个显式类的 `from_pretrained`。
`save_pretrained` 会保存相关 Python 源码与 `auto_map`，因此发布到 Hub 或把目录复制给其他项目后，
可以不用 `import segment`，使用上述 Auto 类并添加 `trust_remote_code=True` 加载。
**仅对自己审查过的代码启用 trust_remote_code。** 完整模型离线加载不会额外去下载骨干。

## 3. 图像与标签预处理

默认与参考数据管线一致：uint8 RGB 转 `[0,1]`、中心裁剪到 patch-size 倍数、不做 ImageNet normalize。

- 输入支持 PIL RGB、numpy HWC/CHW、torch CHW/HWC，以及批量列表/4D 张量。
- 浮点 `[0,1]` 图像不会再次除以 255；如需完全禁用缩放，传 `do_rescale=False`。
- 歧义布局须传 `input_data_format="channels_first"` 或 `"channels_last"`；OpenCV 的 BGR 须先转 RGB。
- `size=None` 各对图根据自身尺寸裁剪；批量尺寸不同不能直接 stack，应设固定 `size` 或分别处理。
- 固定 `size` 时，小图中心补零；标签补 `ignore_index`，不把补出的边框当作背景训练。
- 两图及标签的原始尺寸必须一致。标签只做同步几何操作，**不进行 1/255 缩放**。
- 标签必须是整数类别 ID，默认 0/1，**255 表示忽略**。

**重要：常见的 0/255 二值 PNG 中，255 代表变化前景，不是忽略区域。必须显式映射！**
不映射时，所有 255 像素都会按默认 ignore index 忽略。

```python
processor = SceneChangeProcessor(size=504, mask_value_map={0: 0, 255: 1})
batch = processor(image_before, image_after, segmentation_maps=binary_mask)
outputs = model(**batch)
outputs.loss.backward()
```

需要保留 255 忽略标签时，不配置上述映射。映射未覆盖的值会成为 `ignore_index`。
标签输出为 `[B,H,W]` int64，损失是可选加权 CE；全忽略批次返回可反向传播的零损失。
当标签与输出尺寸不一致时，先将 logits 双线性插值到标签尺寸再算损失，不对离散标签做双线性插值。

`post_process_semantic_segmentation(outputs, target_sizes=[(height,width), ...])`
可将 logits 插值后 argmax，但**不是把中心裁剪恢复到原图坐标**；原边界已被裁掉，不能仅通过 resize 恢复。

## 4. LoRA 微调与恢复

```python
from peft import get_peft_model, PeftModel
from segment import SceneChangeForSemanticSegmentation
from segment.models.scene_change.peft_utils import get_lora_config

# 基座应是已经训练/转换过的变化检测模型，或即将训练的初始化模型。
base = SceneChangeForSemanticSegmentation.from_pretrained("outputs/change_base")
# 如果基座是刚初始化的随机分割头，必须先保存它，再保存 adapter。
model = get_peft_model(base, get_lora_config(
    base, r=8, lora_alpha=16, lora_dropout=0.05,
    adapt_backbone=False,   # True：同时向实际可达的 DINOv2 Q/K/V 注入 LoRA
))
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 接普通 PyTorch 训练循环，或使用 Transformers Trainer。
# batch 包含 pixel_values、pixel_values_pair、labels。
loss = model(**batch).loss
loss.backward()
# optimizer.step() 等由训练循环或 Trainer 执行
model.save_pretrained("outputs/change_adapter")
processor.save_pretrained("outputs/change_adapter")

# 恢复：同一基座 + adapter，不能只加载 adapter 当作完整模型。
base = SceneChangeForSemanticSegmentation.from_pretrained("outputs/change_base")
model = PeftModel.from_pretrained(base, "outputs/change_adapter", is_trainable=True)
# 推理可使用 is_trainable=False；或 merge_and_unload 后另存完整模型。
```

辅助函数不设置 NLP `TaskType`，使用 PEFT 通用 wrapper 保留双图接口。
`modules_to_save=["fusion","classifier"]` 保证变化分割头完整训练/恢复。
骨干适配只选择特征提取真正经过的层和投影；选择最后层 `query` facet 时，最后层 K/V 不会被调用。
冻结是动态检查参数状态，不会将已经插入的 LoRA 在 `no_grad` 内运行。
**`set_backbone_trainable` 仅在注入 adapter 前调用，注入后不要再用它重新冻结整个骨干。**

使用 `Trainer` 时，数据集每个样本返回两个 `[3,H,W]` 张量和一个 `[H,W]` 的 `labels` 张量。
PEFT 通用 wrapper 使用 `TrainingArguments(label_names=["labels"], ...)` 明确标注标签名；
普通训练不传 `TaskType.SEQ_CLS` / `FEATURE_EXTRACTION` 之类 NLP 任务类型。

## 5. 真实数据训练 Demo

[demo.py](demo.py) 仅负责入口分发，实际使用
[SceneChangeTrainer](training/trainer.py)（继承 `transformers.Trainer`），不再手写训练循环。
优化器、调度器、梯度累积、AMP、设备放置、验证和 checkpoint 均交由 Trainer 管理。
默认使用本地 VL-CMU-CD 的固定 train/val 划分，
以 `../../hf_cache/facebook/dinov2-small` 中的预训练权重初始化骨干，并从随机变化分割头开始训练。
模型路径相对于仓库根目录解析，且骨干始终以 `local_files_only=True` 加载，不会连接 Hugging Face 下载。
如需使用 base，可传 `--backbone ../../hf_cache/facebook/dinov2-base`。可指定 `--data-root`、
`--device`、batch size、epoch、学习率、AMP 和输出目录；训练后保存完整 HF 模型、processor 与验证指标。
PSCD 可通过 `--dataset pscd --data-root data_/PSCD` 使用；数据默认处理尺寸保留各自 dataset 行为。
每个 epoch 自动验证并保存 checkpoint，最多保留两个；结束后保存完整 HF 模型、processor、
Trainer 状态以及标准的 train/eval JSON 指标。验证按批次累计混淆矩阵，不保留整个数据集的 logits。
`--gradient-accumulation-steps` 和 `--gradient-checkpointing` 可控制显存占用。
如果已有完整变化模型，可用 `--init-model` 初始化新训练；不要把单独的 PEFT adapter 当完整模型传入。
要恢复优化器、调度器、步数与 RNG 状态，请使用 `--resume-from-checkpoint` 指向 Trainer checkpoint 目录。
`--device` 支持 `auto/cpu/cuda`，多卡选择由 `CUDA_VISIBLE_DEVICES` / Transformers 管理。

```bash
python -m segment.demo --dataset vl-cmu-cd --epochs 1 --batch-size 2 --output-dir outputs/vl-demo
python -m segment.demo --dataset pscd --data-root data_/PSCD --epochs 1
```

`--random-backbone --max-steps 2` 可用于小型离线训练管线检查，但该随机骨干不适用于精度评估。
不需要数据集的合成前向/反向检查可运行 `python -m segment.demo --smoke`；
实现位于 [tools/smoke.py](tools/smoke.py)。旧的顶层 smoke 与 converter 模块入口已移除。

调试时：参数和数据构建看 [training/config.py](training/config.py)，训练入口断点放在
[training/runner.py](training/runner.py)，损失检查、预测及验证指标断点放在
[training/trainer.py](training/trainer.py)。默认 `--workers 0`，便于单进程调试数据加载。

## 6. 原项目 checkpoint 转换

在仓库根目录：

```bash
python -m segment.tools.convert_checkpoint \
    --checkpoint /path/to/dinov2.2CrossAttn.CMU.pth \
    --output outputs/change_cmu
```

转换默认使用 `../../hf_cache/facebook/dinov2-*` 中的对应本地骨干，始终离线加载：

```bash
python -m segment.tools.convert_checkpoint \
    --checkpoint /path/to/reference.pth \
    --backbone /path/to/local/dinov2-small \
    --local-files-only \
    --output outputs/change_cmu
```

转换器处理参考脚本的 `{"model": ..., "args": {"model": ...}}` 格式：
去除 DataParallel `module.`、拆分 attention `in_proj` 和 DINO `qkv`、合并共享骨干重复键、
迁移卷积/LayerScale/MLP 名称、保留类别权重与旧模型输出尺寸。
不同双分支骨干权重、缺失分割头/应训练骨干权重、未知 state keys 都会报错。

原项目通常只保存**可训练参数**，不含冻结 DINOv2 权重，所以转换还须加载匹配的预训练骨干，
不能把随机骨干随缺失 checkpoint 保存。转换完成后输出目录包含全部参数，可完全离线 `from_pretrained`。
默认 `torch.load(weights_only=True)`；仅对可信旧文件在确有需要时加 `--allow-unsafe-pickle`，
不要对不可信 pickle 文件开启该选项。

目前没有自动下载原项目发布的变化检测权重，也没有声称随机模型就是预训练变化检测模型。
small 本地合成 checkpoint 的完整转换经过验证；发布权重以及 base/large/giant 的转换尚未实测。

## 测试覆盖

[tests/test_scene_change.py](tests/test_scene_change.py) 的离线 CPU 模型测试覆盖：

- 具名模型家族导出与继承、声明式/旧格式配置、独立损失和 tuple 输出约定。
- 前向/CE/反向、冻结与末层解冻、梯度检查点、多类别配置保存。
- Q/K/V/token 提取与骨干实际 forward hooks 对照。
- 参考代码 `TwoCrossAttention` 数值对照、完整权重键映射、严格缺失检查。
- 合成 DINO-small 规模参考 checkpoint + 本地骨干的端到端转换与完整模型重新加载。
- 处理器缩放、同步裁剪、补边、掩码映射、numpy 输出及后处理。
- Auto 类本地加载，及隔离进程中不导入 `segment` 的 `trust_remote_code` 加载。
- 骨干/交互层 LoRA 梯度、adapter 保存恢复、merge 后一致性，PEFT + `Trainer` CPU 单步。

[tests/test_backbones.py](tests/test_backbones.py) 另覆盖配置加载开关、完整 checkpoint 脱离原始骨干目录、
DINOv3 四类特征与原生 hooks 对照、BF16 输入类型、LoRA/梯度检查点与冻结策略恢复，
以及 DINOv2 四种尺寸的 meta-device 结构构造。large/giant 未在本地加载真实预训练权重。
真实 small/base/DINOv3-base 权重的逐参数加载和完整模型保存恢复由根目录脚本单独验证。

这些测试验证接口和转换逻辑，不验证真实数据集精度、GPU 性能或发布预训练权重的评测结果。
来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。
