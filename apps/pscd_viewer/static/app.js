"use strict";
const $ = id => document.getElementById(id);
const state = { index: 0, info: null, sample: null, controller: null, generation: 0 };

function query(index) {
  const params = new URLSearchParams();
  for (const key of ["variant", "split", "distance", "mask"]) params.set(key, $(key).value);
  if (index !== undefined) params.set("index", index);
  return params;
}
async function request(path, signal) {
  const response = await fetch(path, { signal });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}
function controls() {
  const diff = $("variant").value === "diff";
  $("distance").disabled = !diff;
  $("mask").disabled = diff;
  if (diff) $("mask").value = "t0";
}
function buttons() {
  const total = state.info?.total || 0;
  $("prev").disabled = $("prev-group").disabled = state.index <= 0 || !total;
  $("next").disabled = $("next-group").disabled = state.index >= total - 1 || !total;
}
function renderImages() {
  if (!state.sample) return;
  const images = state.sample.images;
  // Ground truth is aligned to its source coordinates, not always to After.
  const overlayTarget = state.sample.mask === "t0" ? "before" : "after";
  for (const key of ["before", "after", "label"]) {
    $(key).src = $("overlay").checked && key === overlayTarget ? images.overlay : images[key];
    $("download-" + key).href = images[key];
    $("download-" + key).download = state.sample.filename.replace(/\.png$/, "." + key + ".png");
  }
}
function render(sample) {
  state.sample = sample;
  state.index = sample.index;
  $("filename").textContent = sample.filename;
  $("position").textContent = `${sample.index + 1} / ${sample.total}`;
  $("size").textContent = sample.processed_size.join(" × ");
  $("ratio").textContent = (sample.changed_ratio * 100).toFixed(2) + "%";
  const crop = sample.crop_before === null ? "整图" : `Before crop ${sample.crop_before} → After crop ${sample.crop_after}`;
  $("detail").textContent = `${sample.original} · ${crop} · ${sample.changed_pixels.toLocaleString()} / ${sample.pixels.toLocaleString()} 变化像素`;
  $("index").value = $("scrubber").value = sample.index;
  document.body.classList.toggle("wide", sample.variant === "full");
  const resize = sample.resized ? `原始 ${sample.original_size.join("×")} → 1024×224（RGB 双线性 / Label 最近邻，非作者已验证预处理） → ` : "";
  const view = sample.variant === "diff" ? `错位 ${sample.distance * sample.stride}px；非重叠 t0 标签置 0。` : "双图与标签同步裁剪。";
  $("protocol").textContent = `${resize}${sample.processed_size.join("×")}。${view} RGB 从模型输入 [0,1] 还原；Label 白色=1。叠加跟随 ${sample.mask} 坐标。`;
  renderImages();
  buttons();
}
async function load(reset = false) {
  controls();
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const generation = ++state.generation;
  $("status").textContent = "处理中…";
  $("triptych").setAttribute("aria-busy", "true");
  $("error").hidden = true;
  try {
    if (reset || !state.info) {
      const info = await request("/api/info?" + query(), controller.signal);
      if (generation !== state.generation) return;
      state.info = info;
      state.index = 0;
      $("total").textContent = info.total.toLocaleString();
      $("originals").textContent = info.originals.toLocaleString();
      $("root").textContent = info.root;
      $("index").max = $("scrubber").max = info.total - 1;
    }
    state.index = Math.max(0, Math.min(state.info.total - 1, state.index));
    const sample = await request("/api/sample?" + query(state.index), controller.signal);
    if (generation !== state.generation) return;
    render(sample);
    $("status").textContent = "已同步 · 处理后数据";
  } catch (error) {
    if (generation !== state.generation || error.name === "AbortError") return;
    $("error").textContent = "无法加载：" + error.message;
    $("error").hidden = false;
    $("status").textContent = "加载失败";
  } finally {
    if (generation === state.generation) $("triptych").setAttribute("aria-busy", "false");
  }
}
function move(delta) {
  if (!state.info) return;
  state.index += delta;
  load();
}
for (const key of ["variant", "split", "mask", "distance"]) $(key).addEventListener("change", () => {
  state.info = null;
  load(true);
});
$("prev").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("prev-group").addEventListener("click", () => move(-(state.info?.crops_per_image || 1)));
$("next-group").addEventListener("click", () => move(state.info?.crops_per_image || 1));
$("jump").addEventListener("submit", event => { event.preventDefault(); state.index = Number($("index").value); load(); });
$("scrubber").addEventListener("change", () => { state.index = Number($("scrubber").value); load(); });
$("overlay").addEventListener("change", renderImages);
$("pixelated").addEventListener("change", () => $("triptych").classList.toggle("pixelated", $("pixelated").checked));
document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  const steps = { ArrowLeft: -1, ArrowRight: 1, a: -(state.info?.crops_per_image || 1), d: state.info?.crops_per_image || 1 };
  if (event.key in steps) { event.preventDefault(); move(steps[event.key]); }
});
buttons();
load(true);
