"use strict";
const $ = id => document.getElementById(id);
const state = { index: 0, info: null, sample: null, controller: null, generation: 0 };

function query(index) {
  const params = new URLSearchParams();
  for (const key of ["variant", "split", "distance"]) params.set(key, $(key).value);
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
}
function groupIndex() {
  if (!state.info) return -1;
  const starts = state.info.group_starts;
  let result = 0;
  for (let i = 1; i < starts.length && starts[i] <= state.index; i++) result = i;
  return result;
}
function buttons() {
  const total = state.info?.total || 0;
  const currentGroup = groupIndex();
  const groupCount = state.info?.group_starts.length || 0;
  $("prev").disabled = state.index <= 0 || !total;
  $("next").disabled = state.index >= total - 1 || !total;
  $("prev-group").disabled = currentGroup <= 0;
  $("next-group").disabled = currentGroup < 0 || currentGroup >= groupCount - 1;
}
function renderImages() {
  if (!state.sample) return;
  const images = state.sample.images;
  for (const key of ["before", "after", "label"]) {
    $(key).src = $("overlay").checked && key === "before" ? images.overlay : images[key];
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
  $("detail").textContent = `Group ${sample.group} · sequence ${sample.sequences} · angle ${sample.angle}° · ${sample.before_file} → ${sample.after_file} · ${sample.changed_pixels.toLocaleString()} / ${sample.pixels.toLocaleString()} 变化像素`;
  $("index").value = $("scrubber").value = sample.index;
  const distance = sample.variant === "diff" ? `；索引偏移 ${sample.distance * sample.stride}（${sample.distance} 个序列步长）` : "";
  $("protocol").textContent = `原图 ${sample.original_size.join("×")} → processor 中心裁剪至 ${sample.processed_size.join("×")}；RGB 为模型实际输入 [0,1] 还原显示，Label 按 mask > 0 二值化。${sample.variant === "diff" ? `Diff view 保留同 group、同角度的 t0/t1 配对，标签取 t0${distance}。` : "同名 t0/t1 配对，标签坐标与 Before 对齐。"}`;
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
      $("groups").textContent = info.originals.toLocaleString();
      $("root").textContent = info.root;
      $("index").max = $("scrubber").max = info.total - 1;
    }
    state.index = Math.max(0, Math.min(state.info.total - 1, state.index));
    const sample = await request("/api/sample?" + query(state.index), controller.signal);
    if (generation !== state.generation) return;
    render(sample);
    $("status").textContent = "已同步 · 模型预处理输出";
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
function moveGroup(delta) {
  if (!state.info) return;
  const starts = state.info.group_starts;
  const target = groupIndex() + delta;
  if (target >= 0 && target < starts.length) {
    state.index = starts[target];
    load();
  }
}
for (const key of ["variant", "split", "distance"]) $(key).addEventListener("change", () => {
  state.info = null;
  load(true);
});
$("prev").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("prev-group").addEventListener("click", () => moveGroup(-1));
$("next-group").addEventListener("click", () => moveGroup(1));
$("jump").addEventListener("submit", event => { event.preventDefault(); state.index = Number($("index").value); load(); });
$("scrubber").addEventListener("change", () => { state.index = Number($("scrubber").value); load(); });
$("overlay").addEventListener("change", renderImages);
$("pixelated").addEventListener("change", () => $("triptych").classList.toggle("pixelated", $("pixelated").checked));
document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
  const steps = { ArrowLeft: -1, ArrowRight: 1, a: -1, d: 1 };
  if (event.key in steps) { event.preventDefault(); move(steps[event.key]); }
});
buttons();
load(true);
