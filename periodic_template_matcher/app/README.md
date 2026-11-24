# Input / Mask 查看应用

应用只读取保存好的预测结果，不调用 matcher，也不依赖原始输入文件。推理结果与显示设置相互独立。

## 两阶段使用

在项目根目录运行：

```bash
# 1. 保存预测结果
python -m periodic_template_matcher --config periodic_template_matcher/config.sd1.labelme.opencv.json --out periodic_template_matcher/results/sd1_opencv

# 2. 打开桌面应用
python -m periodic_template_matcher.app --result periodic_template_matcher/results/sd1_opencv
```

`bash run.sh` 会依次执行这两步；第一步失败时不会打开旧结果。

也可以不提供 `--result`，在窗口中选择结果目录。CLI 同时接受 `result.json` 的完整路径。

Python API：

```python
result = matcher.match(templates, masks, input_image)
manifest = matcher.save_result(result, "periodic_template_matcher/results/my_prediction")
```

## 如何观察

- 视图：原图、类别 Mask、透明叠加、左右对照。左右对照时，两侧共用缩放和平移位置。
- 类别：默认显示所有前景；也可选择单个类别，包括背景 ID 0。颜色只用于显示，保存的 mask 仍是整数类别 ID。
- 来源：全部来源（重叠混色）、最终互斥标签、直接配准、周期补全、独立物体。
  新导出独立保存三套来源，互不覆盖；例如直接层的类别 1 可与对象层的类别 2 同时存在。
  按类别筛选可看完整单类掩膜，不再从最终标签反推来源。重叠仅表示假设冲突，并非已确认双方均正确。
  最终互斥标签仍对应原 `mapped_labels.png`，评价及旧 API 行为不变。
- 身份：启用类别 2 独立匹配后，可单独查看配置中的 N 个身份；
  类别 2 不跟随类别 1 周期复制。灰度身份图的 1..N 与语义类别 ID 分开使用。
- 跨边界前景：与未知区域相连的周期物体会补齐其在背景上的完整部分；这些新增前景也归入“周期补全”，已有非零标签不会被覆盖。
- 透明度：调整标签颜色与 input 的混合比例。
- 未知区域：紫色斜纹表示当前所选层的 validity 为 0，与背景类别 0 分开显示；最终层使用 `label_valid`。
- 像素查询：移动鼠标，窗口底部显示原图坐标、类别 ID、直接/周期来源和当前缩放。
- 缩放/平移：滚轮缩放，左键拖动；`F` 适应窗口，`1` 返回 100%。
- `Ctrl+O` 打开结果目录，`Ctrl+R` 重新读取文件。重新读取不会执行推理。

匹配被拒绝或最佳模板没有标注时，应用显示原因和 input，预测 mask 不存在；不会将无预测解释为全背景。

## 保存的数据

| 文件 | 含义 |
| --- | --- |
| `input.png` | 原始分辨率的 input，无损保存 |
| `mapped_labels.png` | `uint16` 类别 ID 图，只有有效预测时存在 |
| `label_valid.png` | 0/255 标签有效区域 |
| `label_periodic_valid.png` | 可选，周期补齐产生的标签区域 |
| `object_identities.png` | 可选，1..N 的独立物体身份图，0 为无已接受身份 |
| `object_valid.png` | 可选，独立物体分支的预测有效范围 |
| `base_labels.png` / `base_valid.png` | 可选，合并独立实例之前的基础标签及有效范围，保留原配准和周期补全结果 |
| `direct_labels.png` / `direct_valid.png` | 可选，直接语义匹配标签及自身有效区；启用独立对象分支时排除其类别，独立于最终融合 |
| `completion_labels.png` / `completion_valid.png` | 可选，融合前周期标签及后续已验收的小实例补全，不清除与对象层的重叠 |
| `object_labels.png` / `object_layer_valid.png` | 可选，独立对象前景及有效区，不受最终融合优先级裁剪 |
| `object_layer_identities.png` | 可选，独立层身份；与最终 `object_identities` 分开保留 |
| `geometric_valid.png` | 直接配准的图像有效重叠区域 |
| `matrix_template_to_target.npy` | 可选，模板原图到 input 原图的变换矩阵 |
| `result.json` | 匹配信息、预测状态、类别统计、相对文件清单 |

`save_result()` 也接受 `EvaluationResult`，并额外保存真值和评估指标。当前 app 重点展示 input 与预测 mask。

移动或分享时复制整个结果目录即可。应用只加载 manifest 指定的数据，不会重新打开原始模板或标注。
旧结果兼容加载，但无法从已经丢失的信息恢复独立层；更新后需要重新推理，而不只是重新读取。

## 环境

应用使用 Tkinter、NumPy 和 OpenCV。后两项沿用包的依赖；Windows 标准 Python 通常自带 Tkinter。可用 `python -m tkinter` 检查桌面支持。推理和保存结果不需要 Tkinter 或桌面环境。
