"""Local desktop inspection of input images and saved predicted masks."""
from __future__ import annotations

import base64
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..utils.types import PathInput
from .data import SavedPrediction, class_color, load_result, render_view, source_pixel


MODES = {'叠加': 'overlay', '左右对照': 'split', '原图': 'input', '类别 Mask': 'mask'}
SOURCES = {'全部预测': 'all', '直接配准': 'direct', '周期补全': 'periodic'}


class PredictionViewer(tk.Tk):
    """Stage 2: read persisted files, then inspect them without running match()."""

    def __init__(self) -> None:
        super().__init__()
        self.title('Prediction Inspector · input / mask')
        self.geometry('1280x900')
        self.minsize(900, 600)
        self.data: SavedPrediction | None = None
        self.scale = 1.
        self.origin = (0., 0.)
        self.fit_mode = True
        self._pending: str | None = None
        self._photo: tk.PhotoImage | None = None
        self._drag = (0, 0)
        self._class_ids: list[int | None] = [None]
        self.mode = tk.StringVar(value='左右对照')
        self.source = tk.StringVar(value='全部预测')
        self.opacity = tk.DoubleVar(value=.5)
        self.class_name = tk.StringVar(value='全部前景')
        self.unknown = tk.BooleanVar(value=True)
        self.summary = tk.StringVar(value='打开第一阶段保存的结果目录，查看 input 和预测 mask。')
        self.details = tk.StringVar(value='')
        self.pixel = tk.StringVar(value='滚轮缩放 · 左键拖动 · F 适应窗口 · 1 原始比例 · Ctrl+O 打开 · Ctrl+R 重新读取')
        self._build_widgets()
        self.bind('<Control-o>', lambda event: self.open_directory())
        self.bind('<Control-r>', lambda event: self.reload())
        self.bind('<Key-f>', lambda event: self.fit())
        self.bind('<Key-1>', lambda event: self.actual_size())

    def _build_widgets(self) -> None:
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill='x')
        for text, command in (('打开结果目录', self.open_directory), ('重新读取', self.reload),
                              ('适应窗口', self.fit), ('100%', self.actual_size)):
            ttk.Button(toolbar, text=text, command=command).pack(side='left', padx=(0, 8))
        ttk.Label(toolbar, textvariable=self.summary).pack(side='left', padx=10)
        controls = ttk.Frame(self, padding=(12, 0, 12, 10))
        controls.pack(fill='x')
        ttk.Label(controls, text='视图').pack(side='left')
        for variable, values, width in ((self.mode, list(MODES), 12),
                                         (self.source, list(SOURCES), 12)):
            combo = ttk.Combobox(controls, textvariable=variable, values=values,
                                 width=width, state='readonly')
            combo.pack(side='left', padx=(6, 14))
            combo.bind('<<ComboboxSelected>>', self._change_view)
        ttk.Label(controls, text='类别').pack(side='left')
        self.classes = ttk.Combobox(controls, textvariable=self.class_name,
                                    values=['全部前景'], width=22, state='readonly')
        self.classes.pack(side='left', padx=(6, 14))
        self.classes.bind('<<ComboboxSelected>>', lambda event: self.request_render())
        ttk.Label(controls, text='透明度').pack(side='left')
        ttk.Scale(controls, from_=0, to=1, variable=self.opacity, length=150,
                   command=lambda value: self.request_render()).pack(side='left', padx=6)
        ttk.Checkbutton(controls, text='标出未知区域', variable=self.unknown,
                         command=self.request_render).pack(side='left', padx=8)
        ttk.Label(self, textvariable=self.details, padding=(12, 0, 12, 8)).pack(fill='x')
        self.legend = ttk.Frame(self, padding=(12, 0, 12, 8))
        self.legend.pack(fill='x')
        self.canvas = tk.Canvas(self, background='#191e26', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self._image_item = self.canvas.create_image(0, 0, anchor='nw')
        self.canvas.bind('<Configure>', self._resize)
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', lambda event: self.zoom(1.2, event.x, event.y))
        self.canvas.bind('<Button-5>', lambda event: self.zoom(1 / 1.2, event.x, event.y))
        self.canvas.bind('<ButtonPress-1>', self._start_drag)
        self.canvas.bind('<B1-Motion>', self._pan)
        self.canvas.bind('<Motion>', self._inspect_pixel)
        ttk.Label(self, textvariable=self.pixel, padding=(12, 8)).pack(fill='x')

    def load(self, path: PathInput) -> None:
        """Replace the current dataset only after all saved files validate."""
        data = load_result(path)
        self.data = data
        self.title(f'Prediction Inspector · {data.manifest.parent.name}')
        meta = data.metadata
        best = meta['matches'][meta['ranking'][0]]
        state = '匹配通过' if meta.get('accepted') else '匹配拒绝'
        self.summary.set(f"{state}  |  分数 {best['match_score']:.2f}  |  {best.get('template_name', '模板 ' + str(best.get('template_index', 0)))}")
        height, width = data.image.shape[:2]
        if data.labels is None:
            reason = '匹配未通过，未导出预测 mask' if not meta.get('accepted') else '模板没有标注，未导出预测 mask'
            self.details.set(f'{width} × {height} px  |  {reason}  |  {data.manifest}')
        else:
            self.details.set(f'{width} × {height} px  |  标签覆盖 {data.valid.mean():.2%}  |  背景 ID=0；紫色斜纹=未知  |  {data.manifest.parent}')
        names = meta.get('class_names', {})
        self._class_ids = [None] + data.class_ids
        choices = ['全部前景'] + [f"{names.get(str(value), '背景' if value == 0 else '类别')} (ID {value})"
                                   for value in data.class_ids]
        self.classes.configure(values=choices)
        self.classes.current(0)
        for child in self.legend.winfo_children():
            child.destroy()
        for value in data.class_ids[:12]:
            b, g, r = class_color(value)
            swatch = tk.Label(self.legend, text=f'  ID {value}  ', background=f'#{r:02x}{g:02x}{b:02x}')
            swatch.pack(side='left', padx=(0, 8))
        self.fit()

    def open_directory(self) -> None:
        directory = filedialog.askdirectory(title='选择包含 result.json 的结果目录', parent=self)
        if directory:
            self._load_or_warn(directory)

    def _load_or_warn(self, path: PathInput) -> None:
        try:
            self.load(path)
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
            messagebox.showerror('无法读取结果', str(error), parent=self)

    def reload(self) -> None:
        if self.data is not None:
            self._load_or_warn(self.data.manifest)

    def _pane_size(self) -> tuple[int, int]:
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        return (max(1, width // 2) if self.mode.get() == '左右对照' else width), height

    def fit(self) -> None:
        self.fit_mode = True
        if self.data is not None:
            width, height = self._pane_size()
            ih, iw = self.data.image.shape[:2]
            self.scale = max(.001, min(width / iw, height / ih) * .96)
            self.origin = ((width - iw * self.scale) / 2, (height - ih * self.scale) / 2)
        self.request_render()

    def actual_size(self) -> None:
        if self.data is not None:
            width, height = self._pane_size()
            self.fit_mode = False
            self.scale = 1.
            self.origin = ((width - self.data.image.shape[1]) / 2,
                           (height - self.data.image.shape[0]) / 2)
            self.request_render()

    def _change_view(self, event: tk.Event) -> None:
        if self.fit_mode:
            self.fit()
        else:
            self.request_render()

    def _resize(self, event: tk.Event) -> None:
        if self.fit_mode:
            self.fit()
        else:
            self.request_render()

    def zoom(self, factor: float, x: int, y: int) -> None:
        if self.data is None:
            return
        pane_width, _ = self._pane_size()
        if self.mode.get() == '左右对照' and x >= pane_width:
            x -= pane_width
        new_scale = min(32., max(.01, self.scale * factor))
        ratio = new_scale / self.scale
        self.origin = (x - (x - self.origin[0]) * ratio, y - (y - self.origin[1]) * ratio)
        self.scale = new_scale
        self.fit_mode = False
        self.request_render()

    def _wheel(self, event: tk.Event) -> None:
        self.zoom(1.2 if event.delta > 0 else 1 / 1.2, event.x, event.y)

    def _start_drag(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y)

    def _pan(self, event: tk.Event) -> None:
        self.fit_mode = False
        self.origin = (self.origin[0] + event.x - self._drag[0],
                       self.origin[1] + event.y - self._drag[1])
        self._drag = (event.x, event.y)
        self.request_render()
        self._inspect_pixel(event)

    def _inspect_pixel(self, event: tk.Event) -> None:
        if self.data is None:
            return
        x = event.x
        pane_width, _ = self._pane_size()
        if self.mode.get() == '左右对照' and x >= pane_width:
            x -= pane_width
        x, y = source_pixel(x, event.y, self.scale, self.origin)
        height, width = self.data.image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            self.pixel.set(f'缩放 {self.scale:.0%}  |  图像范围外')
            return
        state = '无预测'
        if self.data.valid is not None:
            state = '未知（无有效标签）'
            if self.data.valid[y, x]:
                periodic = self.data.periodic is not None and self.data.periodic[y, x]
                state = f"类别 ID {self.data.labels[y, x]} · {'周期补全' if periodic else '直接配准'}"
        self.pixel.set(f'原图 x={x}, y={y}  |  {state}  |  缩放 {self.scale:.0%}  |  滚轮缩放 / 拖动平移')

    def request_render(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(25, self._render)

    def _render(self) -> None:
        self._pending = None
        if self.data is None:
            return
        index = self.classes.current()
        selected = self._class_ids[index] if index >= 0 else None
        size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        image = render_view(self.data, size, self.scale, self.origin,
                            mode=MODES[self.mode.get()], alpha=self.opacity.get(),
                            class_id=selected, source=SOURCES[self.source.get()],
                            show_unknown=self.unknown.get())
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Could not encode viewer frame')
        self._photo = tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.delete('caption')
        if self.mode.get() == '左右对照':
            for x, text in ((16, 'INPUT 原图'), (size[0] // 2 + 16, '预测 Mask 叠加')):
                self.canvas.create_text(x, 18, text=text, anchor='nw', fill='#edf2fa',
                                         font=('Microsoft YaHei UI', 11, 'bold'), tags='caption')
