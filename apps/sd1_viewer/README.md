# SD-1 数据集观察器

在项目根目录运行（仅需 Python 3.10+ 标准库，无需安装依赖）：

```powershell
.venv\Scripts\python.exe -m apps.sd1_viewer
```

自动打开 http://127.0.0.1:7875 。首次启动扫描 XML 可能需要十几秒，终端按 Ctrl+C 停止。

指定其他数据集或端口：

```powershell
python -m apps.sd1_viewer --dataset D:\datasets\SD-1 --port 7876 --no-browser
```

数据目录必须包含 `imgs/` 和 `Annotations/`。按相对目录与同名文件配对，如 `imgs/TN4FL0/a.jpg` 对应 `Annotations/TN4FL0/a.xml`，不会混用其他目录下的同名文件。支持 JPG、JPEG、PNG、BMP、WebP（解码取决于浏览器）及 Pascal VOC XML。

- 左侧按文件名、目录、标注类别、配对状态筛选，列表分页，方向键切换图片。
- 中间滚轮以鼠标为中心缩放，拖动平移；F 适应窗口，100% 按钮恢复原始比例，B 显隐标注。
- 点击矩形框或右侧标注查看缺陷局部、坐标和 difficult 属性；重叠框优先选择面积较小者。
- 右侧展开文件信息和数据集概览，点击类别统计可筛选。类别统计是全部 XML 的标注框数量，包含缺少图片的记录。
- 缺图片、缺 XML、损坏 XML 会单独提示。标注坐标原样展示，不自动转换 VOC 坐标约定；超出原图的部分由画布裁剪。

应用只读，不执行推理或修改源文件。修改数据后重启服务重新索引。默认仅监听本机地址，所有页面资源均在本地，无外部服务依赖。

验证：

```powershell
.venv\Scripts\python.exe -m unittest tests.test_sd1_viewer
```
