# Python 调用 C++ 核心模块示例

这是独立的教学案例，不接入现有业务。Python 负责数据准备、参数校验、调用、结果处理和计时；只有模板匹配的耗时循环用 C++ 实现。

采用 `ctypes` 加载动态库，无需 pybind11 或 Python 开发头文件。依赖 NumPy，以及一个本机 C++ 编译器。Windows 可使用 MSVC Build Tools（安装“使用 C++ 的桌面开发”，并在 Python 环境安装 setuptools），或者 PATH 中的 g++ / clang++。Linux/macOS 使用 g++ / clang++。Python 与编译器生成的库必须具有相同架构（例如均为 x64）。

从项目根目录运行（Windows PowerShell，使用现有虚拟环境）：

```powershell
.\.venv\Scripts\python.exe .\c++_speed_test\build.py
.\.venv\Scripts\python.exe .\c++_speed_test\check.py
.\.venv\Scripts\python.exe .\c++_speed_test\demo.py
```

其他环境可进入此目录运行 `python build.py` 和 `python demo.py`。修改 C++ 后需要重新编译，并重新启动 Python 进程。生成的库和构建产物已在此目录的 `.gitignore` 中排除。

如果 Windows 没有编译器，也可以下载一个仅放在本示例构建目录内的工具链（约 100 MB，不需要配置系统 PATH）：

```powershell
.\.venv\Scripts\python.exe -m pip install ziglang --target .\c++_speed_test\build\toolchain
.\.venv\Scripts\python.exe .\c++_speed_test\build.py
```

`build.py` 会自动识别该位置的 Zig C++ 编译器。如果虚拟环境缺少 pip，可先执行 `.\.venv\Scripts\python.exe -m ensurepip`。

文件职责：

| 文件 | 职责 |
| --- | --- |
| `matcher.cpp` | C++ 核心：遍历候选位置并计算灰度差平方和 SSD，越小越好 |
| `build.py` | 编译为 Windows DLL、Linux SO 或 macOS dylib |
| `native_matcher.py` | Python 包装接口，以及用于对照的纯 Python 实现 |
| `demo.py` | 生成固定随机数据、植入模板、核对结果、比较耗时 |
| `check.py` | 核对随机输入、平分情况、非连续数组和非法输入 |

在同目录的 Python 程序中，调用方式为：

```python
from native_matcher import match_cpp

# image 和 template 为二维 numpy.uint8 灰度图
x, y, score = match_cpp(image, template)
```

坐标为模板左上角；相同分数时优先取从上到下、从左到右遇到的位置。连续数组直接传递底层内存地址；非连续切片由包装层复制为连续数组。调用期间 Python 保持数组存活，C++ 不保存这些指针。

可调整规模：

```powershell
.\.venv\Scripts\python.exe .\c++_speed_test\demo.py --size 128 --template-size 16 --repeats 3
```

程序先预热动态库，再报告多次调用的中位耗时。C++ 耗时包括 Python 包装和跨语言调用；不包括编译、首次加载和数据生成。纯 Python 的四层循环较慢，增大参数会显著增加等待时间。

**性能含义：** 这里比较的是相同 SSD 算法的 Python 像素循环和优化编译的 C++ 循环，展示如何只替换热点模块。它不是对现有项目的加速承诺，也不是与 OpenCV 的性能比较。`cv2.matchTemplate` 本身已有原生实现；该示例未实现旋转、缩放、归一化相关性或其他业务匹配功能。
