'use strict';
const $ = id => document.getElementById(id);
const state = {items: [], total: 0, offset: 0, limit: 80, current: null, image: null, selected: -1, scale: 1, x: 0, y: 0, fit: true, loadToken: 0, queryToken: 0};
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const statusNames = {paired: '图片 + 标注', image_only: '缺少标注', annotation_only: '缺少图片'};
function color(label) {let n = 0; for (const c of label) n = (n * 31 + c.charCodeAt(0)) >>> 0; return `hsl(${n % 360} 80% 65%)`;}
function node(tag, text, cls) {const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el;}
async function api(url) {const r = await fetch(url); if (!r.ok) throw new Error(`读取失败 (${r.status})`); return r.json();}
function params() {return new URLSearchParams({q: $('search').value.trim(), label: $('label').value, group: $('group').value, status: $('status').value, offset: state.offset, limit: state.limit});}
function message(text) {$('empty').textContent = text; $('empty').hidden = false;}
function renderList() {
  $('list').replaceChildren();
  for (const row of state.items) {
    const b = node('button', undefined, 'file' + (state.current?.id === row.id ? ' active' : ''));
    b.title = row.key; b.setAttribute('aria-current', String(state.current?.id === row.id));
    b.append(node('span', row.key.split('/').pop(), 'file-name'));
    const meta = node('span', undefined, 'meta'); meta.append(node('span', row.group), node('span', row.error ? '标注异常' : statusNames[row.status], row.error || row.status !== 'paired' ? 'warning' : ''));
    b.append(meta); b.onclick = () => select(row.id); $('list').append(b);
  }
  if (!state.items.length) $('list').append(node('div', '没有符合条件的记录', 'hint'));
  $('pageInfo').textContent = state.total ? `${state.offset + 1}–${state.offset + state.items.length} / ${state.total}` : '0 条记录';
  $('total').textContent = `${state.total.toLocaleString()} 条`;
  $('pagePrev').disabled = state.offset === 0; $('pageNext').disabled = state.offset + state.limit >= state.total;
  const index = state.items.findIndex(r => r.id === state.current?.id);
  $('prev').disabled = !state.total || (state.offset === 0 && index <= 0);
  $('next').disabled = !state.total || (state.offset + state.items.length >= state.total && index === state.items.length - 1);
}
async function query(pickLast = false) {
  const token = ++state.queryToken;
  try {
    const data = await api('/api/items?' + params()); if (token !== state.queryToken) return;
    state.items = data.items; state.total = data.total;
    if (state.items.length) await select(state.items[pickLast ? state.items.length - 1 : 0].id);
    else {++state.loadToken; state.current = null; state.image = null; state.selected = -1; $('filename').textContent = '没有符合条件的记录'; $('dimensions').textContent = ''; $('objects').replaceChildren(); $('objectCount').textContent = '0'; $('objectInfo').textContent = ''; $('paths').replaceChildren(); $('notice').hidden = true; draw(); drawCrop(); message('没有符合条件的记录，请调整筛选条件');}
    renderList();
  } catch (e) {message(e.message);}
}
async function select(id) {
  const token = ++state.loadToken;
  state.image = null; state.selected = -1; draw(); drawCrop(); message('正在加载图片…');
  try {
    const item = await api('/api/item/' + id); if (token !== state.loadToken) return;
    state.current = item; state.fit = true;
    $('filename').textContent = item.key; $('filename').title = item.key;
    $('dimensions').textContent = item.width && item.height ? `${item.width} × ${item.height}` : '';
    const warning = [item.status === 'annotation_only' ? '对应图片不存在，仍可查看标注坐标。' : '', item.status === 'image_only' ? '对应 XML 不存在。' : '', item.error ? '标注读取异常：' + item.error : ''].filter(Boolean).join(' ');
    $('notice').textContent = warning; $('notice').hidden = !warning;
    $('paths').replaceChildren(node('div', '图片：' + (item.image_path || '缺失')), node('div', '标注：' + (item.annotation_path || '缺失')));
    renderObjects(); renderList();
    if (!item.image_url) {message('此标注缺少对应图片'); return;}
    const image = new Image();
    image.onload = () => {if (token !== state.loadToken) return; state.image = image; $('dimensions').textContent = `${image.naturalWidth} × ${image.naturalHeight}`; $('empty').hidden = true; fit(); if (item.objects.length) chooseObject(0);};
    image.onerror = () => {if (token === state.loadToken) message('图片无法解码或文件已不存在');};
    image.src = item.image_url;
  } catch (e) {if (token === state.loadToken) message(e.message);}
}
function renderObjects() {
  $('objects').replaceChildren();
  const objects = state.current?.objects || []; $('objectCount').textContent = objects.length;
  $('objectInfo').textContent = objects.length ? '点击标注查看局部和坐标' : '当前记录没有可显示的标注';
  objects.forEach((obj, i) => {
    const b = node('button', undefined, 'object' + (i === state.selected ? ' active' : ''));
    const swatch = node('span', undefined, 'swatch'); swatch.style.background = color(obj.label);
    const name = node('div', obj.label); name.append(node('small', obj.box.map(v => Math.round(v)).join(', ')));
    b.append(swatch, name, node('span', `#${i + 1}`, 'number')); b.onclick = () => chooseObject(i); $('objects').append(b);
  });
}
function chooseObject(index) {
  state.selected = index; renderObjects();
  const obj = state.current.objects[index], [x1, y1, x2, y2] = obj.box;
  $('objectInfo').textContent = `${obj.label} · ${(x2-x1).toFixed(0)} × ${(y2-y1).toFixed(0)} px\n坐标 [${obj.box.join(', ')}] · difficult=${obj.difficult}`;
  draw(); drawCrop();
}
function sizeCanvas(c) {const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); const context = c.getContext('2d'); context.setTransform(dpr, 0, 0, dpr, 0, 0); return [r.width, r.height];}
function fit() {
  if (!state.image) return; state.fit = true;
  state.scale = Math.max(.01, Math.min((canvas.clientWidth - 40) / state.image.naturalWidth, (canvas.clientHeight - 40) / state.image.naturalHeight));
  center(); draw();
}
function center() {state.x = (canvas.clientWidth - state.image.naturalWidth * state.scale) / 2; state.y = (canvas.clientHeight - state.image.naturalHeight * state.scale) / 2;}
function draw() {
  const [w,h] = sizeCanvas(canvas); ctx.clearRect(0,0,w,h); if (!state.image) return;
  const s = state.scale; ctx.save(); ctx.translate(state.x,state.y); ctx.scale(s,s); ctx.drawImage(state.image,0,0);
  if ($('boxes').checked) (state.current?.objects || []).forEach((obj,i) => {
    const [x1,y1,x2,y2] = obj.box, selected = state.selected === i;
    ctx.strokeStyle = selected ? '#fff' : color(obj.label); ctx.fillStyle = color(obj.label); ctx.lineWidth = (selected ? 3 : 2) / s;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    if (selected) {ctx.globalAlpha = .13; ctx.fillRect(x1,y1,x2-x1,y2-y1); ctx.globalAlpha = 1;}
    if ($('names').checked) {const label = `${i+1} ${obj.label}`; ctx.font = `${12/s}px Segoe UI`; const tw = ctx.measureText(label).width; const top = Math.max(0,y1-20/s); ctx.fillRect(x1,top,tw+10/s,20/s); ctx.fillStyle='#10151d'; ctx.fillText(label,x1+5/s,top+14/s);}
  });
  ctx.restore(); $('zoom').textContent = `${Math.round(s*100)}%`;
}
function drawCrop() {
  const c = $('crop'), [w,h] = sizeCanvas(c), cx = c.getContext('2d'); cx.clearRect(0,0,w,h);
  const obj = state.current?.objects[state.selected]; $('cropEmpty').hidden = !!(obj && state.image);
  if (!obj || !state.image) return;
  const [x1,y1,x2,y2] = obj.box, padding = Math.max(x2-x1,y2-y1)*.18;
  const sx = Math.max(0,x1-padding), sy = Math.max(0,y1-padding), ex = Math.min(state.image.naturalWidth,x2+padding), ey = Math.min(state.image.naturalHeight,y2+padding);
  if (ex<=sx || ey<=sy) return;
  const scale = Math.min((w-20)/(ex-sx),(h-20)/(ey-sy)), dx = (w-(ex-sx)*scale)/2, dy = (h-(ey-sy)*scale)/2;
  cx.drawImage(state.image,sx,sy,ex-sx,ey-sy,dx,dy,(ex-sx)*scale,(ey-sy)*scale);
  cx.strokeStyle=color(obj.label); cx.lineWidth=1.5; cx.strokeRect(dx+(x1-sx)*scale,dy+(y1-sy)*scale,(x2-x1)*scale,(y2-y1)*scale);
}
function zoom(factor, x=canvas.clientWidth/2, y=canvas.clientHeight/2) {
  if (!state.image) return; state.fit=false;
  const s = Math.min(20,Math.max(.01,state.scale*factor)), f=s/state.scale;
  state.x=x-(x-state.x)*f; state.y=y-(y-state.y)*f; state.scale=s; draw();
}
async function navigate(direction) {
  const i = state.items.findIndex(r=>r.id===state.current?.id), next=i+direction;
  if (next>=0 && next<state.items.length) return select(state.items[next].id);
  if (direction<0 && state.offset>0) {state.offset=Math.max(0,state.offset-state.limit); await query(true);}
  if (direction>0 && state.offset+state.limit<state.total) {state.offset+=state.limit; await query();}
}
let drag = null;
canvas.addEventListener('pointerdown', e=>{drag={x:e.offsetX,y:e.offsetY,ox:state.x,oy:state.y,moved:false}; canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{
  if (drag) {const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y; if (Math.hypot(dx,dy)>3) drag.moved=true; if(drag.moved){state.fit=false;state.x=drag.ox+dx;state.y=drag.oy+dy;draw();}}
  if(state.image){const x=Math.floor((e.offsetX-state.x)/state.scale),y=Math.floor((e.offsetY-state.y)/state.scale); $('pointer').textContent=`坐标 (${x}, ${y}) · ${Math.round(state.scale*100)}%`;}
});
canvas.addEventListener('pointerup',e=>{
  if(drag && !drag.moved && state.image && $('boxes').checked){const x=(e.offsetX-state.x)/state.scale,y=(e.offsetY-state.y)/state.scale; const hits=state.current.objects.map((o,i)=>({o,i})).filter(({o})=>x>=o.box[0]&&x<=o.box[2]&&y>=o.box[1]&&y<=o.box[3]).sort((a,b)=>(a.o.box[2]-a.o.box[0])*(a.o.box[3]-a.o.box[1])-(b.o.box[2]-b.o.box[0])*(b.o.box[3]-b.o.box[1])); if(hits.length)chooseObject(hits[0].i);}
  drag=null;
});
canvas.addEventListener('pointercancel',()=>{drag=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY);},{passive:false});
$('fit').onclick=fit; $('actual').onclick=()=>{if(state.image){state.scale=1;state.fit=false;center();draw();}};
$('zoomIn').onclick=()=>zoom(1.25); $('zoomOut').onclick=()=>zoom(.8);
$('boxes').onchange=draw; $('names').onchange=draw;
$('prev').onclick=()=>navigate(-1); $('next').onclick=()=>navigate(1);
$('pagePrev').onclick=()=>{state.offset=Math.max(0,state.offset-state.limit);query();};
$('pageNext').onclick=()=>{state.offset+=state.limit;query();};
for(const id of ['label','group','status']) $(id).onchange=()=>{state.offset=0;query();};
let timer; $('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{state.offset=0;query();},250);};
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||e.ctrlKey||e.altKey||e.metaKey)return; if(['ArrowLeft','ArrowRight','f','F','b','B'].includes(e.key))e.preventDefault(); if(e.key==='ArrowLeft')navigate(-1);if(e.key==='ArrowRight')navigate(1);if(e.key.toLowerCase()==='f')fit();if(e.key.toLowerCase()==='b'){$('boxes').checked=!$('boxes').checked;draw();}});
new ResizeObserver(()=>{if(state.fit)fit();else draw();drawCrop();}).observe($('stage'));
async function init() {
  try {
    const summary = await api('/api/summary'); $('stats').replaceChildren();
    for(const [key,label] of [['images','图片'],['paired','已配对'],['annotation_only','缺少图片'],['invalid','标注异常']]){const el=node('span');el.append(node('b',summary[key].toLocaleString()),document.createTextNode(label));$('stats').append(el);}
    for(const group of summary.groups){const option=node('option',group);option.value=group;$('group').append(option);}
    $('overview').append(node('div',summary.root),node('div',`缺少标注：${summary.image_only} · 类别：${Object.keys(summary.labels).length}`),node('div','以下为全部 XML 的标注框数量（含缺图记录）'));
    for(const [label,count] of Object.entries(summary.labels)){
      const option=node('option',`${label} (${count})`);option.value=label;$('label').append(option);
      const row=node('div',undefined,'class-stat'),button=node('button',label);button.onclick=()=>{$('label').value=label;state.offset=0;query();};row.append(button,node('span',count));$('overview').append(row);
    }
    await query();
  }catch(e){message(e.message+'，请确认本地服务正在运行');}
}
init();
