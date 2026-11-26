"""双图变化分割的运行检查：预处理 -> 前向 -> loss -> 反向传播。

仓库根目录运行：python -m segment.demo
也可以直接运行本文件。默认使用 CPU 和小型随机模型，不下载任何权重。
"""

import argparse
from pathlib import Path
import sys

import torch
from transformers import Dinov2Config

# 兼容 VS Code 的「运行 Python 文件」及 python segment/demo.py。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation, SceneChangeProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1", help="运行设备，如 cpu 或 cuda:1")
    parser.add_argument("--model-size", choices=["tiny", "small"], default="tiny",
                        help="tiny 快速检查；small 使用标准 DINOv2-small 架构，均为随机权重")
    parser.add_argument("--pretrained", help="可选：加载已保存的完整变化模型 HF 目录")
    parser.add_argument("--image-size", type=int, default=56, help="模拟图像边长，默认 56")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用 --device cpu")

    if args.pretrained:
        model = SceneChangeForSemanticSegmentation.from_pretrained(args.pretrained, local_files_only=True)
        processor = SceneChangeProcessor.from_pretrained(args.pretrained, local_files_only=True)
        print(f"加载完整模型：{args.pretrained}")
    else:
        backbone = (Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=56)
                    if args.model_size == "tiny" else SceneChangeConfig().backbone_config)
        config = SceneChangeConfig(backbone_config=backbone, classifier_relu=False, dropout=0)
        model = SceneChangeForSemanticSegmentation(config)
        processor = SceneChangeProcessor(patch_size=backbone.patch_size)
        print(f"模型：{args.model_size}，随机初始化，仅验证运行，不代表检测效果")

    patch = model.config.backbone_config.patch_size
    if args.image_size < patch or args.image_size % patch:
        parser.error(f"--image-size 必须是 {patch} 的正整数倍")
    model = model.to(device).train()
    print(f"Python：{sys.executable}")
    print(f"设备：{device}")
    print(f"总参数：{sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 模拟一对 RGB 图像及 0/1 类别标签，在第二张图的中央制造变化。
    side = args.image_size
    before = torch.randint(0, 256, (3, side, side), dtype=torch.uint8)
    after = before.clone()
    labels = torch.zeros(side, side, dtype=torch.long)
    start, end = side // 4, side * 3 // 4
    after[:, start:end, start:end] = 255 - after[:, start:end, start:end]
    labels[start:end, start:end] = 1
    inputs = processor(before, after, segmentation_maps=labels).to(device)
    for name, value in inputs.items():
        print(f"{name}：shape={tuple(value.shape)}，dtype={value.dtype}")

    # 提供 labels 时，模型会计算交叉熵 loss。
    outputs = model(**inputs)
    if outputs.loss is None or not torch.isfinite(outputs.loss).item():
        raise RuntimeError("模型未返回有效 loss")
    if not torch.isfinite(outputs.logits).all().item():
        raise RuntimeError("logits 包含 NaN/Inf")
    prediction = outputs.logits.argmax(dim=1)
    print(f"logits：{tuple(outputs.logits.shape)}")
    print(f"预测掩码：{tuple(prediction.shape)}")
    print(f"loss：{outputs.loss.item():.6f}")

    outputs.loss.backward()
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(grad).all().item() for _, grad in gradients):
        raise RuntimeError("反向传播未产生有效梯度")
    if not any(torch.count_nonzero(grad).item() > 0 for _, grad in gradients):
        raise RuntimeError("所有梯度均为零")
    print(f"反向传播成功：{len(gradients)} 个参数张量有梯度")
    print(f"classifier.weight 梯度范数：{model.classifier.weight.grad.norm().item():.6f}")
    print("运行检查通过。未进行优化器更新，未保存文件。")


if __name__ == "__main__":
    main()