# VL-CMU-CD 处理后数据 Viewer

并排查看模型预处理后的 Before RGB、After RGB 和 Label。Viewer 只读本地数据，
不加载模型、不进行预测、不修改或复制数据文件。

在仓库根目录启动：

```bash
python -m apps.vl_cmu_cd_viewer --no-browser
```

默认打开 `http://127.0.0.1:7878`，与 PSCD Viewer 的 7877 端口分开。
默认读取仓库内的 `data_/VL-CMU-CD2/VL-CMU-CD-binary255`；也可指定根目录：

```bash
python -m apps.vl_cmu_cd_viewer --dataset /path/to/VL-CMU-CD-binary255 --port 7878
```

## 功能

- Paired 同名配对与 Diff view 序列错位配对。
- Train / Validation / Test 固定 group 划分。
- 展示实际 processor 输出的三联图、文件名、group、sequence、angle、尺寸与变化像素占比。
- 样本索引跳转、滑动定位、上一/下一样本及上一/下一 group。
- 标签叠加到 Before、像素显示切换、各图 PNG 下载。
- Diff 距离 −3…+3（不含 0）；train/val 索引步长 4，test 步长 1，仅显示同 group 且角度一致的配对。

Viewer 调用 `dataset[index]`，因此显示内容与训练的 `SceneChangeProcessor` 输出一致：
512×512 原图默认中心裁剪为 504×504；RGB `[0,1]` 还原为 PNG 展示，灰度 mask 用
`mask > 0` 映射为 0/1。错位视图的标签取自 t0，叠加也定位在 Before。

服务默认仅绑定 `127.0.0.1`，只提供固定静态资源和只读 API；无鉴权，不建议对不可信网络使用 `--host 0.0.0.0`。

测试：

```bash
python -m unittest tests.test_vl_cmu_cd_viewer -v
```