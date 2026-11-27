# PSCD 处理后数据 Viewer

并排展示一组 **Before RGB / After RGB / Label RGB**，直接读取
`segment.data` 的模型输入处理结果，不加载模型、不进行预测、不写数据文件。

仓库根目录运行（使用安装了 segment 依赖的 Python）：

```bash
python -m apps.pscd_viewer --no-browser
```

打开 http://127.0.0.1:7877 。默认自动识别本地 `data_/PSCD/PSCD`。
默认端口与已有 SD-1 / Matcher viewer 分开。
VS Code 远程环境可在“端口”面板转发 7877，或使用集成浏览器。

指定其他目录或端口：

```bash
python -m apps.pscd_viewer --dataset /path/to/PSCD --port 7877 --no-browser
```

## 功能

- Full / Cropped / Diff view 三种模式；train / val / test 固定划分。
- Before / After / Label 三联图：label 黑色=0 未变化，白色=1 变化，导出为 RGB PNG。
- 显示原图名、裁剪块编号、处理尺寸、样本数和变化像素占比。
- 样本翻页、索引跳转、滑动定位、上一/下一原图组。
- 错位距离 −3…+3（每档 57 像素）；Diff view 固定使用 mask_t0。
- 普通模式可切换 mask_t0 / mask_t1；标签叠加跟随监督来源的坐标。
  不把错位的 t0 标签叠加到 t1 上。
- 像素/平滑显示切换，各图单独下载 PNG。
- 快捷键：← / → 翻样本，A / D 切换原图组。

## 处理与显示

服务调用 `dataset[index]`，即使用真实 processor 输出，再将 `[0,1]` RGB
还原到 uint8 显示；标签按类别 ID 着色，不是直接展示原始 PNG。
增强关闭，不做 ImageNet normalization。

Cropped 和 Diff view 会按当前数据协议先将高分辨率原图缩放到 1024×224，
再生成 224×224 块。Full 模式只按 patch size 中心裁剪（本地原图输出为
4088×1148），不会缩放成小图；浏览器仅按卡片大小缩小显示。
高分辨率适配并非作者已验证的原始预处理，细节见
[../../segment/data/README.md](../../segment/data/README.md)。

只开放固定静态资源和只读 API，不接受任意文件路径。默认仅监听 loopback；
无鉴权，不建议用 `--host 0.0.0.0` 暴露到不可信网络。
使用数据仍须遵守 PSCD 的研究用途与禁止未经许可再分发条款。

## 测试

```bash
python -m unittest tests.test_pscd_viewer -v
```
