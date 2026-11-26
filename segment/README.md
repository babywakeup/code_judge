# Transformers 场景变化分割模型

将参考项目 **Robust-Scene-Change-Detection** 的主模型迁移为原生 Hugging Face
模型，不依赖参考项目的 `py_utils`、`torch.hub`、DataParallel 或运行时下载源码。

## 支持范围

- `SceneChangeConfig(PretrainedConfig)`：保存骨干、特征层、facet、注意力、输出尺寸、冻结策略和损失配置。
- `SceneChangeForSemanticSegmentation(PreTrainedModel)`：`save_pretrained` / `from_pretrained`、
  safetensors、结构化 `ModelOutput`、训练损失、梯度检查点。
- `SceneChangeImageProcessor(BaseImageProcessor)`，别名 `SceneChangeProcessor`：双图同步预处理、
  标签同步裁剪、保存加载、预测后处理。单一图像模态，不需要 tokenizer / 多模态 `ProcessorMixin`。
- `AutoConfig`、`AutoModelForSemanticSegmentation`、`AutoImageProcessor`、`AutoProcessor`。
- PEFT LoRA：交互层 Q/K/V/output 投影，及可选 DINOv2 骨干 Q/K/V 投影；卷积分割头完整训练并随 adapter 保存。
- 原项目 `dino2 + cross_attention` 与 `dino2 + single_cross_attention1` 的本地 `.pth` 转换。
  DINOv2 small/base/large/giant 架构配置与标准非 register-token 权重键映射。

**没有迁移** ResNet、CoAttention、MTF、TemporalAttention 等消融模型；不支持 DINOv2 register-token 变体。
当前不是 Transformers 内置单图 segmentation pipeline：推理必须同时传入两张图。

### 环境与验证

开发及测试使用用户指定的 **mura_det / Python 3.12.14**：
`/data2/zrx/mura_det/.venv/bin/python`，PyTorch 2.14.0+cu126、Transformers 5.17.0。
依赖见 [requirements.txt](requirements.txt)，LoRA 需要 `peft`，不做 LoRA 时可不安装它。
本实现只在上述 Transformers 版本上做了运行验证，不声称已经验证 4.x。

在仓库根目录安装/运行：

```bash
/data2/zrx/mura_det/.venv/bin/python -m pip install -r segment/requirements.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data2/zrx/mura_det/.venv/bin/python -m unittest segment.tests.test_scene_change -v
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

## 1. 从预训练骨干初始化

这会读取 HF 本地缓存或下载 DINOv2 权重，**变化分割头仍是随机初始化，必须训练**。

```python
from segment import SceneChangeForSemanticSegmentation, SceneChangeProcessor

model = SceneChangeForSemanticSegmentation.from_backbone_pretrained(
    "facebook/dinov2-small",   # 也可以是本地 HF DINOv2 目录
    feature_layer=11,
    facet="query",
    facet_pair="query",
    num_attention_heads=1,
    num_attention_blocks=1,
    attention_mode="cross",
    classifier_relu=False,
    freeze_backbone=True,
    # backbone_load_kwargs={"local_files_only": True},
)
processor = SceneChangeProcessor(size=504)

model.save_pretrained("outputs/change_base")
processor.save_pretrained("outputs/change_base")
```

`SceneChangeForSemanticSegmentation(SceneChangeConfig(...))` 是完全随机初始化，**不会下载**骨干。
`from_pretrained` 则用于加载已经保存的**完整变化模型**，不要把 DINOv2 模型 ID 传给它冒充分割权重。

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
from segment.peft_utils import get_lora_config

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

## 5. 原项目 checkpoint 转换

在仓库根目录：

```bash
/data2/zrx/mura_det/.venv/bin/python -m segment.convert_checkpoint \
    --checkpoint /path/to/dinov2.2CrossAttn.CMU.pth \
    --output outputs/change_cmu
```

离线转换时，提前准备对应 HF DINOv2 目录：

```bash
/data2/zrx/mura_det/.venv/bin/python -m segment.convert_checkpoint \
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

[tests/test_scene_change.py](tests/test_scene_change.py) 的 **19 项离线 CPU 测试**覆盖：

- 前向/CE/反向、冻结与末层解冻、梯度检查点、多类别配置保存。
- Q/K/V/token 提取与骨干实际 forward hooks 对照。
- 参考代码 `TwoCrossAttention` 数值对照、完整权重键映射、严格缺失检查。
- 合成 DINO-small 规模参考 checkpoint + 本地骨干的端到端转换与完整模型重新加载。
- 处理器缩放、同步裁剪、补边、掩码映射、numpy 输出及后处理。
- Auto 类本地加载，及隔离进程中不导入 `segment` 的 `trust_remote_code` 加载。
- 骨干/交互层 LoRA 梯度、adapter 保存恢复、merge 后一致性，PEFT + `Trainer` CPU 单步。

这些测试验证接口和转换逻辑，不验证真实数据集精度、GPU 性能或发布预训练权重的评测结果。
来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。