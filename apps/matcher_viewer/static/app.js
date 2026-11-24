'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const filterNames = ['q', 'group', 'status', 'template', 'station', 'severity', 'date_from', 'date_to', 'min_score', 'sort'];
  const statuses = {
    pending: '待匹配', running: '匹配中', available: '匹配通过',
    rejected: '匹配未通过', no_annotations: '无标注', error: '运行异常'
  };
  const modeNames = {input: '原始图像', overlay: '分割叠加', mask: '预测掩膜', split: '原图 / 匹配结果对照'};
  const preferenceKey = 'aoi.matcher.viewer.v1';
  const state = {
    items: [], total: 0, offset: 0, limit: 80, columns: 5, filters: {},
    selected: new Set(), cards: new Map(), summary: null, token: null,
    view: {mode: 'overlay', alpha: '0.5', source: 'all', unknown: true},
    listController: null, listVersion: 0, loading: false, listLoaded: false, invalidFilters: false,
    summaryController: null, summaryVersion: 0, pollTimer: null,
    actionController: null, actionBusy: false, actionKind: '', filterVersion: 0,
    modalController: null, modalVersion: 0, modalId: null, detail: null, detailRevision: null,
    zoom: false, viewTimer: null, filterTimer: null, toastTimer: null
  };
  const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const count = value => number(value).toLocaleString('zh-CN');
  const finite = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const timingLabel = ({ timing_scope }) => timing_scope === 'excluding_image_save'
    ? '耗时（不含图片保存）' : timing_scope === 'including_image_save' ? '耗时（含图片保存）' : '耗时';
  const score = value => finite(value) ? Number(value).toFixed(3) : '—';
  const percent = value => finite(value) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function badge(status) {
    const node = el('span', undefined, 'badge');
    if (Object.hasOwn(statuses, status)) node.classList.add(status);
    node.append(el('span', undefined, 'status-dot'), document.createTextNode(statuses[status] || String(status || '未知状态')));
    node.title = '模板匹配验收状态，不代表缺陷判定';
    return node;
  }
  function showError(error) {
    $('errorText').textContent = error instanceof Error ? error.message : String(error);
    $('errorBanner').hidden = false;
  }
  function toast(text) {
    clearTimeout(state.toastTimer);
    $('toast').textContent = text; $('toast').hidden = false;
    state.toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
  }
  function aborted(error) { return error?.name === 'AbortError'; }
  async function request(url, options = {}) {
    let response;
    try { response = await fetch(url, {cache: 'no-store', ...options}); }
    catch (error) {
      if (aborted(error)) throw error;
      throw new Error('无法连接本地服务，请检查服务是否正在运行，再重试。');
    }
    let data;
    try { data = await response.json(); }
    catch (error) {
      if (aborted(error)) throw error;
      throw new Error(`服务返回了非 JSON 响应（HTTP ${response.status}）。`);
    }
    if (!response.ok) throw new Error(data?.error || `请求失败（HTTP ${response.status}）。`);
    return data;
  }
  function post(url, payload, signal) {
    if (!state.token) throw new Error('未取得操作令牌，请先重试连接。');
    return request(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Viewer-Token': state.token}, body: JSON.stringify(payload), signal});
  }
  function readFilters() {
    return Object.fromEntries(filterNames.map(name => [name, $(name).value.trim()]));
  }
  function queryString(filters, offset = state.offset, limit = state.limit) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filters)) if (value !== '') params.set(key, value);
    params.set('offset', String(offset)); params.set('limit', String(limit));
    return params.toString();
  }
  function savePreferences() {
    try { localStorage.setItem(preferenceKey, JSON.stringify({filters: readFilters(), limit: state.limit, columns: state.columns, view: state.view, skipCompleted: $('skipCompleted').checked})); }
    catch (_) { /* Private browsing / storage quotas must not break the viewer. */ }
  }
  function restorePreferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(preferenceKey) || 'null');
      if (!saved || typeof saved !== 'object') return;
      for (const name of filterNames) {
        const value = saved.filters?.[name];
        if (typeof value !== 'string') continue;
        if (['group', 'template', 'station', 'severity'].includes(name) && value) {
          const option = el('option', value); option.value = value; $(name).append(option);
        }
        $(name).value = value;
      }
      if (!$('sort').value) $('sort').value = 'key';
      if ([40, 80, 120, 200].includes(saved.limit)) state.limit = saved.limit;
      if ([4, 5, 6, 8].includes(saved.columns)) state.columns = saved.columns;
      if (saved.view && typeof saved.view === 'object') {
        if (Object.hasOwn(modeNames, saved.view.mode)) state.view.mode = saved.view.mode;
        if (['all', 'fused', 'direct', 'periodic', 'objects'].includes(saved.view.source)) state.view.source = saved.view.source;
        if (finite(saved.view.alpha)) state.view.alpha = String(Math.max(0, Math.min(1, Number(saved.view.alpha))));
        if (typeof saved.view.unknown === 'boolean') state.view.unknown = saved.view.unknown;
      }
      if (typeof saved.skipCompleted === 'boolean') $('skipCompleted').checked = saved.skipCompleted;
    } catch (_) { /* Ignore malformed or inaccessible saved settings. */ }
    $('pageSize').value = String(state.limit); $('columns').value = String(state.columns);
    $('gallery').style.setProperty('--columns', state.columns);
  }
  function syncViewControls() {
    document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === state.view.mode)));
    for (const id of ['source', 'detailSource']) $(id).value = state.view.source;
    for (const id of ['alpha', 'detailAlpha']) $(id).value = state.view.alpha;
    for (const id of ['alphaValue', 'detailAlphaValue']) $(id).textContent = `${Math.round(Number(state.view.alpha) * 100)}%`;
    for (const id of ['unknown', 'detailUnknown']) $(id).checked = state.view.unknown;
    $('detailZoom').setAttribute('aria-pressed', String(state.zoom));
    $('detailZoom').textContent = state.zoom ? '适应窗口' : '高清放大 · 1920px';
    $('detailStage').classList.toggle('zoomed', state.zoom);
  }
  function revision(row) {
    // Stable during a poll; invalidate only when this record's matcher result changes.
    return JSON.stringify([row.status, Boolean(row.result), row.score, row.coverage, row.template, row.elapsed, row.timing_scope, row.error, row.provenance]);
  }
  function effectiveMode(row) {
    // Pending / unprocessed records always have a useful original preview.
    return row.status === 'pending' || !row.result ? 'input' : state.view.mode;
  }
  function previewUrl(row, width, rev) {
    const mode = effectiveMode(row);
    const params = new URLSearchParams({mode, width: String(width)});
    if (mode !== 'input') {
      params.set('alpha', state.view.alpha); params.set('source', state.view.source);
      params.set('unknown', state.view.unknown ? '1' : '0');
      // Query cache-buster does not create a new request on unchanged summary polls.
      params.set('v', rev || revision(row));
    }
    return `/preview/${encodeURIComponent(row.id)}?${params}`;
  }

  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      const card = state.cards.get(entry.target.dataset.id);
      if (card) { card.visible = true; updateCardPreview(card); }
      observer.unobserve(entry.target);
    }
  }, {rootMargin: '240px'}) : null;

  function updateCardPreview(card) {
    if (!card.visible) return;
    const url = previewUrl(card.row, 480, card.revision);
    if (card.url === url) return;
    card.url = url;
    card.placeholder.hidden = false; card.placeholder.classList.remove('failed');
    card.placeholder.textContent = '正在加载预览…';
    const image = el('img', undefined, 'preview-image');
    image.alt = card.row.key || 'AOI 图像'; image.loading = 'lazy'; image.decoding = 'async';
    image.onload = () => { if (card.image === image) card.placeholder.hidden = true; };
    image.onerror = () => {
      if (card.image !== image) return;
      // A failed result preview falls back to a real input preview, not a blank card.
      if (effectiveMode(card.row) !== 'input' && !image.dataset.fallback) {
        image.dataset.fallback = '1'; card.placeholder.textContent = '结果预览不可用，读取原图…';
        const params = new URLSearchParams({mode: 'input', width: '480'});
        image.src = `/preview/${encodeURIComponent(card.row.id)}?${params}`;
      } else {
        image.hidden = true; card.placeholder.hidden = false;
        card.placeholder.classList.add('failed'); card.placeholder.textContent = '图像不可用 · 点击查看详情';
      }
    };
    card.image?.remove(); card.image = image; card.preview.prepend(image); image.src = url;
  }
  function createCard(row) {
    const root = el('article', undefined, 'card'); root.dataset.id = row.id;
    const previewArea = el('div', undefined, 'card-preview');
    const preview = el('button', undefined, 'preview-button'); preview.type = 'button';
    const placeholder = el('span', '预览按需加载', 'preview-placeholder'); preview.append(placeholder);
    const selectLabel = el('label', undefined, 'card-select');
    const checkbox = el('input'); checkbox.type = 'checkbox'; selectLabel.append(checkbox);
    const status = el('div', undefined, 'card-badge'); previewArea.append(preview, selectLabel, status);
    const body = el('div', undefined, 'card-body');
    const title = el('button', undefined, 'card-title'); title.type = 'button';
    const subline = el('div', undefined, 'card-subline');
    const metrics = el('div', undefined, 'card-metrics');
    const scoreNode = el('strong', undefined, 'card-score'); scoreNode.title = '匹配分数（后端原始量纲）';
    const template = el('span', undefined, 'card-template'); metrics.append(scoreNode, template);
    const foot = el('div', undefined, 'card-foot');
    const coverage = el('span'); const date = el('span'); foot.append(coverage, date);
    const error = el('div', undefined, 'card-error');
    body.append(title, subline, metrics, foot, error); root.append(previewArea, body);
    const card = {root, preview, placeholder, checkbox, status, title, subline, scoreNode, template, coverage, date, error, row, visible: !observer, image: null, url: '', revision: ''};
    preview.onclick = title.onclick = () => openDetail(card.row.id);
    checkbox.onchange = () => {
      if (checkbox.checked) state.selected.add(card.row.id); else state.selected.delete(card.row.id);
      updateSelection();
    };
    return card;
  }
  function updateCard(card, row) {
    card.row = row; card.revision = revision(row);
    card.status.replaceChildren(badge(row.status));
    const name = String(row.key || row.id).split(/[\\/]/).pop();
    card.title.textContent = name; card.title.title = row.key || row.id;
    card.preview.setAttribute('aria-label', `查看 ${name} 的匹配详情`);
    card.checkbox.setAttribute('aria-label', `选择 ${name}`);
    card.subline.textContent = [row.group, row.station, row.severity, row.provenance === 'imported' ? '历史结果' : ''].filter(Boolean).join(' · ') || '未分组';
    card.subline.title = [row.key, row.group, row.station, row.severity].filter(Boolean).join(' / ');
    card.scoreNode.textContent = score(row.score);
    card.template.textContent = row.template || '未匹配模板'; card.template.title = row.template || '';
    card.coverage.textContent = `覆盖 ${percent(row.coverage)}`;
    card.date.textContent = row.date || '—'; card.date.title = [row.date, finite(row.elapsed) ? `${timingLabel(row)} ${Number(row.elapsed).toFixed(2)}s` : ''].filter(Boolean).join(' · ');
    card.error.hidden = !row.error; card.error.textContent = row.error || ''; card.error.title = row.error || '';
    updateCardPreview(card);
  }
  function renderGallery() {
    const liveIds = new Set(state.items.map(row => row.id));
    for (const [id, card] of state.cards) if (!liveIds.has(id)) {
      observer?.unobserve(card.root); card.image && (card.image.onload = card.image.onerror = null);
      card.root.remove(); state.cards.delete(id);
    }
    for (const id of state.selected) if (!liveIds.has(id)) state.selected.delete(id);
    state.items.forEach((row, index) => {
      let card = state.cards.get(row.id);
      if (!card) { card = createCard(row); state.cards.set(row.id, card); }
      updateCard(card, row);
      // Reconcile in place: never destroy stable images or reset scroll on each poll.
      const atIndex = $('gallery').children[index];
      if (atIndex !== card.root) $('gallery').insertBefore(card.root, atIndex || null);
      if (!card.visible) observer?.observe(card.root);
    });
    $('emptyState').hidden = Boolean(state.items.length);
    $('emptyTitle').textContent = '没有符合条件的图像';
    $('emptyText').textContent = '调整筛选条件，或重新扫描数据目录。';
    $('emptyReset').hidden = false;
    updatePagination(); updateSelection();
    if (state.modalId) {
      const row = state.items.find(item => item.id === state.modalId);
      if (!row) closeDetail();
      else {
        updateDetailNavigation();
        if (state.detailRevision !== revision(row) && state.detail) loadDetail(row.id);
      }
    }
  }
  function updatePagination() {
    const pages = Math.max(1, Math.ceil(state.total / state.limit));
    const page = Math.floor(state.offset / state.limit) + 1;
    $('filteredTotal').textContent = `${count(state.total)} 张`;
    $('rangeInfo').textContent = state.total ? `显示 ${count(state.offset + 1)}–${count(state.offset + state.items.length)} / ${count(state.total)} 张` : '0 张图像';
    $('pageInfo').textContent = `第 ${page} / ${count(pages)} 页`;
    $('pageInput').value = String(page); $('pageInput').max = String(pages);
    $('pagePrev').disabled = state.loading || state.offset === 0;
    $('pageNext').disabled = state.loading || state.offset + state.limit >= state.total;
    $('pageGo').disabled = state.loading || !state.total;
    $('filterCount').textContent = String(filterNames.filter(name => name !== 'sort' && state.filters[name]).length);
  }
  function updateSelection() {
    const selected = state.selected.size;
    $('selectionCount').textContent = selected ? `已选 ${count(selected)} 张` : '未选择';
    $('clearSelection').disabled = !selected;
    $('selectPage').checked = Boolean(state.items.length) && selected === state.items.length;
    $('selectPage').indeterminate = selected > 0 && selected < state.items.length;
    $('selectPage').disabled = state.loading || !state.items.length;
    for (const [id, card] of state.cards) {
      card.checkbox.checked = state.selected.has(id); card.checkbox.disabled = state.loading;
      card.root.classList.toggle('selected', state.selected.has(id));
    }
    updateActions();
  }
  function updateActions() {
    const active = Boolean(state.summary?.job?.active);
    const blocked = state.actionBusy || state.loading || state.invalidFilters || !state.token || active;
    $('runPage').disabled = blocked || !state.items.length;
    $('runFiltered').disabled = blocked || !state.total;
    $('runPage').textContent = state.selected.size ? `▶ 匹配已选 (${state.selected.size})` : '▶ 匹配本页';
    $('runFiltered').textContent = state.actionKind === 'job' ? '正在提交…' : '▶ 匹配全部筛选';
    $('stopJob').disabled = state.actionBusy || !state.token || !active;
    $('refresh').disabled = state.actionBusy || !state.token || active;
    $('refresh').title = active ? '请等待任务结束后重新扫描' : '重新扫描数据目录';
  }
  async function loadItems({quiet = false} = {}) {
    clearTimeout(state.filterTimer);
    state.listController?.abort(); const controller = new AbortController(); state.listController = controller;
    const version = ++state.listVersion; const filters = {...state.filters};
    if (!quiet) {
      state.loading = true; $('gallery').classList.add('loading'); $('gallery').setAttribute('aria-busy', 'true');
      $('loadingLabel').textContent = '正在读取…'; updatePagination(); updateSelection();
    }
    try {
      const data = await request(`/api/items?${queryString(filters)}`, {signal: controller.signal});
      if (version !== state.listVersion) return;
      if (!Array.isArray(data.items) || !finite(data.total)) throw new Error('图库接口格式不正确：需要 total 与 items。');
      state.total = Math.max(0, number(data.total));
      if (state.total && state.offset >= state.total) {
        state.offset = Math.floor((state.total - 1) / state.limit) * state.limit;
        return loadItems({quiet});
      }
      if (!state.total) state.offset = 0;
      state.items = data.items.map(row => ({...row, id: String(row.id)}));
      state.listLoaded = true; renderGallery();
    } catch (error) {
      if (!aborted(error) && version === state.listVersion) {
        showError(error);
        if (!quiet) {
          // A failed new filter/page must not leave actionable rows from the old query.
          state.items = []; state.total = 0; state.offset = 0; renderGallery();
        }
        if (!state.items.length) {
          $('emptyState').hidden = false; $('emptyTitle').textContent = '图库加载失败';
          $('emptyText').textContent = '请检查本地服务，使用上方“重试连接”恢复。'; $('emptyReset').hidden = true;
        }
      }
    } finally {
      if (version === state.listVersion) {
        state.loading = false; $('gallery').classList.remove('loading'); $('gallery').setAttribute('aria-busy', 'false');
        $('loadingLabel').textContent = ''; updatePagination(); updateSelection();
      }
    }
  }
  function options(id, values) {
    const select = $(id), current = select.value;
    const list = Array.isArray(values) ? values.map(String) : [];
    const existing = [...select.options].slice(1).map(option => option.value);
    if (JSON.stringify(existing) === JSON.stringify(list)) return;
    const first = select.options[0].cloneNode(true); select.replaceChildren(first);
    for (const value of list) { const option = el('option', value); option.value = value; select.append(option); }
    // Keep a restored/active filter even if a rescan temporarily removes its option.
    if (current && !list.includes(current)) { const option = el('option', `${current}（当前筛选）`); option.value = current; select.append(option); }
    select.value = current;
  }
  function renderJob(job = {}) {
    const total = Math.max(0, number(job.total)), done = Math.max(0, number(job.done));
    $('jobDot').classList.toggle('active', Boolean(job.active));
    $('jobTitle').textContent = job.active ? (job.cancelled ? '正在停止 · 等待当前图像完成' : '匹配任务运行中') : job.cancelled ? '任务已停止' : total ? '匹配任务已结束' : '暂无运行任务';
    $('jobProgressText').textContent = total || job.active ? `${count(done)} / ${count(total)} · ${total ? Math.min(100, Math.round(done / total * 100)) : 0}% · 逐张推理` : '';
    $('jobProgress').max = total || 1; $('jobProgress').value = Math.min(done, total);
    const current = typeof job.current === 'object' && job.current ? (job.current.key || job.current.id || '') : job.current;
    $('jobCurrent').textContent = job.active ? (job.cancelled ? '当前图像完成后停止。' : `当前：${current || '正在准备…'}`) : total ? `已处理 ${count(done)} 张${job.cancelled ? ' · 剩余排队图像未继续运行' : ''}` : '可按当前页或全部筛选结果启动匹配';
    $('jobCurrent').title = $('jobCurrent').textContent;
    $('jobErrors').textContent = number(job.errors) ? `运行异常 ${count(job.errors)} 张` : '';
    updateActions();
  }
  function summarySignature(summary) {
    return JSON.stringify([summary.total, summary.counts, summary.job?.active, summary.job?.total, summary.job?.done, summary.job?.errors, summary.job?.current, summary.job?.cancelled]);
  }
  function renderSummary(summary) {
    const counts = summary.counts || {};
    $('statTotal').textContent = count(summary.total); $('statAvailable').textContent = count(counts.available);
    $('statPending').textContent = count(counts.pending);
    $('statIssues').textContent = count(number(counts.rejected) + number(counts.error));
    for (const [id, key] of [['group', 'groups'], ['template', 'templates'], ['station', 'stations'], ['severity', 'severities']]) options(id, summary[key]);
    $('statusCounts').replaceChildren();
    for (const [status, label] of Object.entries(statuses)) {
      const row = el('div', undefined, 'legend-row'), button = el('button'); button.type = 'button';
      button.append(badge(status)); button.setAttribute('aria-label', `${label}，${count(counts[status])} 张，点击筛选`);
      button.onclick = () => { $('status').value = status; filtersChanged(); };
      row.append(button, el('strong', count(counts[status]))); $('statusCounts').append(row);
    }
    renderJob(summary.job || {});
  }
  function schedulePoll() {
    clearTimeout(state.pollTimer); state.pollTimer = null;
    if (state.summary?.job?.active) state.pollTimer = setTimeout(() => loadSummary({poll: true}), 2000);
  }
  async function loadSummary({poll = false, forceList = false} = {}) {
    clearTimeout(state.pollTimer); state.pollTimer = null;
    state.summaryController?.abort(); const controller = new AbortController(); state.summaryController = controller;
    const version = ++state.summaryVersion;
    try {
      const summary = await request('/api/summary', {signal: controller.signal});
      if (version !== state.summaryVersion) return;
      if (!finite(summary.total) || !summary.counts) throw new Error('概览接口格式不正确：需要 total 与 counts。');
      const changed = !state.summary || summarySignature(summary) !== summarySignature(state.summary);
      state.summary = summary; state.token = summary.token || null;
      $('connection').textContent = state.token ? '服务已连接' : '只读 · 操作令牌缺失'; $('connection').className = 'connection online';
      renderSummary(summary);
      // Do not steal a debounced filter's request or cancel its pending timer.
      if (!state.loading && !state.invalidFilters && (forceList || !state.listLoaded || (poll && changed))) {
        await loadItems({quiet: state.listLoaded && !forceList});
      }
    } catch (error) {
      if (!aborted(error) && version === state.summaryVersion) {
        $('connection').textContent = '连接异常'; $('connection').className = 'connection offline'; showError(error);
        if (!state.listLoaded && !state.loading) {
          $('gallery').setAttribute('aria-busy', 'false'); $('rangeInfo').textContent = '尚未读取图像';
          $('emptyState').hidden = false; $('emptyTitle').textContent = '无法连接匹配服务';
          $('emptyText').textContent = '请启动本地服务，然后点击上方“重试连接”。'; $('emptyReset').hidden = true;
        }
      }
    } finally { if (version === state.summaryVersion) schedulePoll(); }
  }
  function validFilters() {
    const filters = readFilters();
    if (filters.date_from && filters.date_to && filters.date_from > filters.date_to) {
      showError('日期范围不正确：起始日期不能晚于结束日期。'); return false;
    }
    if ($('min_score').validity.badInput) { showError('最低匹配分数必须是有效数字。'); return false; }
    return true;
  }
  function filtersChanged(delay = 0) {
    clearTimeout(state.filterTimer); state.listController?.abort(); ++state.listVersion; ++state.filterVersion;
    // Cancel a not-yet-submitted all-filtered count lookup; never submit stale filters.
    if (state.actionKind === 'count') state.actionController?.abort();
    state.selected.clear(); closeDetail();
    state.loading = true; $('gallery').classList.add('loading'); $('gallery').setAttribute('aria-busy', 'true');
    $('loadingLabel').textContent = '筛选中…'; updateSelection(); updatePagination(); savePreferences();
    state.filterTimer = setTimeout(() => {
      state.invalidFilters = !validFilters();
      if (state.invalidFilters) {
        state.loading = false; $('gallery').classList.remove('loading'); $('gallery').setAttribute('aria-busy', 'false');
        $('loadingLabel').textContent = '筛选条件无效 · 保留上次查询结果'; updateSelection(); updatePagination(); return;
      }
      state.filters = readFilters(); state.offset = 0; loadItems();
    }, delay);
  }
  function resetFilters() {
    $('filterForm').reset(); $('sort').value = 'key'; filtersChanged();
  }
  function goPage(page) {
    if (state.loading) return;
    const pages = Math.max(1, Math.ceil(state.total / state.limit));
    if (!Number.isInteger(page) || page < 1 || page > pages) { toast(`请输入 1–${pages} 之间的页码`); return; }
    closeDetail(); state.selected.clear(); state.offset = (page - 1) * state.limit;
    loadItems(); $('gallery').scrollIntoView({block: 'start', behavior: 'auto'});
  }
  async function runJob(allFiltered) {
    if (state.actionBusy || state.loading || state.summary?.job?.active || !validFilters()) return;
    const filters = {...state.filters}, filterVersion = state.filterVersion;
    const ids = state.selected.size ? [...state.selected] : state.items.map(row => row.id);
    const skip = $('skipCompleted').checked;
    const controller = new AbortController(); state.actionController = controller; state.actionBusy = true;
    state.actionKind = allFiltered ? 'count' : 'job'; updateActions();
    try {
      let actualCount = ids.length;
      if (allFiltered) {
        // Obtain the actual scope count from the server rather than stale header stats.
        const data = await request(`/api/items?${queryString(filters, 0, 1)}`, {signal: controller.signal});
        if (!finite(data.total)) throw new Error('无法确认全部筛选结果的数量。');
        actualCount = Math.max(0, number(data.total));
      }
      if (controller.signal.aborted || filterVersion !== state.filterVersion) return;
      if (!actualCount) { toast('没有可提交的图像'); return; }
      const scope = allFiltered ? '全部筛选结果' : state.selected.size ? '本页已选图像' : '当前页全部图像';
      const cacheHint = skip ? '将跳过当前配置的已完成结果。' : '将重新推理范围内的图像。';
      if (!window.confirm(`启动模板匹配？\n\n范围：${scope}\n当前范围：${count(actualCount)} 张图像\n逐张推理，${cacheHint}\n停止请求会在当前图像完成后生效。`)) return;
      state.actionKind = 'job'; updateActions();
      const payload = allFiltered ? {filters, skip_completed: skip} : {ids, skip_completed: skip};
      const job = await post('/api/jobs', payload, controller.signal);
      if (state.summary) state.summary.job = job;
      renderJob(job); schedulePoll();
      toast(`任务已提交 · 服务端任务数量 ${count(job.total)} 张`);
      await loadSummary({forceList: true});
    } catch (error) { if (!aborted(error)) showError(error); }
    finally { state.actionBusy = false; state.actionKind = ''; state.actionController = null; updateActions(); }
  }
  async function operation(kind) {
    if (state.actionBusy || !state.token) return;
    state.actionBusy = true; state.actionKind = kind; updateActions();
    try {
      await post(kind === 'stop' ? '/api/stop' : '/api/refresh', {});
      toast(kind === 'stop' ? '停止请求已发送，当前图像完成后停止。' : '数据目录已重新扫描');
      await loadSummary({forceList: true});
    } catch (error) { showError(error); }
    finally { state.actionBusy = false; state.actionKind = ''; updateActions(); }
  }

  function updateDetailNavigation() {
    const index = state.items.findIndex(row => row.id === state.modalId);
    $('detailPrev').disabled = index <= 0; $('detailNext').disabled = index < 0 || index >= state.items.length - 1;
    $('detailPosition').textContent = `本页 ${index + 1} / ${state.items.length}`;
  }
  function clearDetail() {
    state.detail = null; state.detailRevision = null;
    $('detailImageWrap').replaceChildren(el('div', '正在读取详情…', 'detail-placeholder'));
    for (const id of ['detailStatus', 'detailStats', 'classStats', 'downloads']) $(id).replaceChildren();
    $('metadataJson').textContent = ''; $('resultPath').textContent = '';
    $('metadataDetails').open = false; $('detailNotice').textContent = '正在读取详情…';
    $('detailNotice').className = 'detail-notice'; $('detailStage').scrollTo(0, 0);
  }
  function openDetail(id) {
    if (state.loading) return;
    if (!$('detailModal').open) { $('detailModal').showModal(); document.body.classList.add('modal-open'); }
    state.zoom = false; syncViewControls(); loadDetail(id);
  }
  function closeDetail() {
    state.modalController?.abort(); ++state.modalVersion; state.modalId = null;
    state.detail = null; state.detailRevision = null;
    $('detailImageWrap').replaceChildren();
    if ($('detailModal').open) $('detailModal').close();
    document.body.classList.remove('modal-open');
  }
  async function loadDetail(id) {
    state.modalController?.abort(); const controller = new AbortController(); state.modalController = controller;
    const version = ++state.modalVersion; state.modalId = id; clearDetail(); updateDetailNavigation();
    const row = state.items.find(item => item.id === id);
    $('detailTitle').textContent = row?.key || id;
    try {
      const item = await request(`/api/item/${encodeURIComponent(id)}`, {signal: controller.signal});
      if (version !== state.modalVersion || !$('detailModal').open) return;
      state.detail = {...item, id}; state.detailRevision = revision(row || item);
      renderDetail();
    } catch (error) {
      if (aborted(error) || version !== state.modalVersion) return;
      $('detailImageWrap').replaceChildren(el('div', '详情加载失败，请关闭后重试', 'detail-placeholder'));
      $('detailNotice').textContent = error.message; $('detailNotice').classList.add('warning');
    }
  }
  function metric(label, value, wide = false) {
    const node = el('div', undefined, 'metric' + (wide ? ' wide' : ''));
    node.append(el('span', label), el('strong', value ?? '—')); return node;
  }
  function classEntries(metadata) {
    const pixels = metadata.class_pixels, ids = metadata.class_ids;
    if (Array.isArray(pixels)) return pixels.map((value, index) => Array.isArray(value) && value.length === 2 ? value : [Array.isArray(ids) ? ids[index] ?? index : index, value]);
    if (pixels && typeof pixels === 'object') return Object.entries(pixels);
    return [];
  }
  function artifactKinds(item) {
    // Do NOT invent known mask names or link arbitrary paths. Only manifest entries
    // are eligible. Accept an array of kinds, entry objects, or a kind->path map.
    const manifest = item.manifest || item.result?.manifest || item.metadata?.manifest;
    const sources = [item.artifacts, item.artifact_kinds, item.result?.artifacts, item.result?.artifact_kinds, item.metadata?.artifacts, manifest?.artifacts, manifest?.artifact_kinds];
    if (Array.isArray(manifest)) sources.push(manifest);
    const kinds = new Set();
    for (const source of sources) {
      if (Array.isArray(source)) for (const entry of source) {
        const kind = typeof entry === 'string' ? entry : entry?.kind;
        if (typeof kind === 'string' && kind) kinds.add(kind);
      }
      else if (source && typeof source === 'object') for (const [kind, value] of Object.entries(source)) {
        if (value !== null && value !== false && value !== '') kinds.add(kind);
      }
    }
    return [...kinds];
  }
  function renderDetail() {
    const item = state.detail; if (!item) return;
    const metadata = item.metadata || item.result?.metadata || {};
    $('detailTitle').textContent = item.key || item.id;
    $('detailStatus').replaceChildren(badge(item.status), el('p', '匹配通过 / 未通过仅表示模板匹配验收，不代表缺陷判定。'));
    const objects = metadata.ordered_objects;
    const objectCount = Array.isArray(objects) ? objects.length : Array.isArray(objects?.objects) ? objects.objects.length : finite(metadata.object_count) ? metadata.object_count : '—';
    const stats = [
      metric('匹配分数', score(item.score)),
      metric('有效覆盖', percent(item.coverage)),
      metric(timingLabel(item), finite(item.elapsed) ? `${Number(item.elapsed).toFixed(2)} s` : '—'),
      metric('有序对象', objectCount),
      metric('匹配模板', item.template || '—', true)
    ];
    const { x_start, x_end, y_start, y_end } = item.valid_bounds || {};
    if ([x_start, x_end, y_start, y_end].every(Number.isInteger) && x_end > x_start && y_end > y_start) {
      stats.push(metric('输入有效区', `${x_end - x_start} × ${y_end - y_start} px · X [${x_start}, ${x_end}) · Y [${y_start}, ${y_end})`, true));
    }
    stats.push(metric('分组 / 工位', [item.group, item.station].filter(Boolean).join(' / ') || '—', true),
               metric('严重度 / 日期', [item.severity, item.date].filter(Boolean).join(' / ') || '—', true));
    $('detailStats').replaceChildren(...stats);
    const classes = classEntries(metadata);
    $('classStats').replaceChildren();
    if (metadata.mask_layers?.overlap_allowed) $('classStats').append(el('p', '以下统计为最终互斥标签；独立来源可重叠，请切换图层检查。', 'muted small-copy'));
    if (!classes.length) $('classStats').append(el('p', '暂无类别像素统计', 'muted small-copy'));
    for (const [id, pixels] of classes) {
      const row = el('div', undefined, 'class-row');
      row.append(el('span', `类别 ${id}`), el('strong', finite(pixels) ? `${count(pixels)} px` : JSON.stringify(pixels)));
      $('classStats').append(row);
    }
    $('downloads').replaceChildren();
    const labels = {input: '↓ 原图', input_image: '↓ 原图', prediction: '↓ 预测掩膜', prediction_mask: '↓ 最终互斥标签', predicted_mask: '↓ 预测掩膜', mask: '↓ 掩膜', input_mask: '↓ 输入掩膜', manifest: '↓ 结果清单', direct_labels: '↓ 直接匹配标签', direct_valid: '↓ 直接匹配有效区', completion_labels: '↓ 周期补全标签', completion_valid: '↓ 周期补全有效区', object_labels: '↓ 独立对象标签', object_layer_valid: '↓ 独立对象有效区', object_layer_identities: '↓ 独立对象身份'};
    const kinds = artifactKinds(item);
    for (const kind of kinds) {
      const link = el('a', labels[kind] || `↓ ${kind}`, 'download-link');
      link.href = `/artifact/${encodeURIComponent(item.id)}/${encodeURIComponent(kind)}`; link.download = ''; $('downloads').append(link);
    }
    if (!kinds.length) $('downloads').append(el('span', '清单未提供可下载文件', 'muted small-copy'));
    $('metadataJson').textContent = JSON.stringify({metadata, result: item.result}, null, 2);
    $('resultPath').textContent = item.result_path ? `结果路径：${item.result_path}` : '';
    if (item.provenance === 'imported') $('resultPath').prepend(document.createTextNode('历史冻结结果，未认证与当前配置相同。 '));
    renderDetailPreview();
  }
  function safeImageUrl(value) {
    if (typeof value !== 'string' || !value) return null;
    try {
      const url = new URL(value, location.href);
      return url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch (_) { return null; }
  }
  function renderDetailPreview() {
    const item = state.detail; if (!item || !$('detailModal').open) return;
    const version = state.modalVersion;
    const url = previewUrl(item, state.zoom ? 1920 : 960, state.detailRevision);
    const existing = $('detailImageWrap').querySelector('img');
    if (existing?.dataset.requestUrl === url) return;
    const image = el('img'); image.alt = `${item.key || item.id} · ${modeNames[effectiveMode(item)]}`;
    image.decoding = 'async'; image.draggable = false; image.dataset.requestUrl = url;
    const placeholder = el('div', '正在渲染预览…', 'detail-placeholder');
    $('detailImageWrap').replaceChildren(placeholder);
    $('renderCaption').textContent = modeNames[effectiveMode(item)];
    $('detailNotice').className = 'detail-notice';
    const note = item.error || (effectiveMode(item) === 'input' && state.view.mode !== 'input' ? '当前记录尚无匹配结果，显示原始图像。' : '预览设置与图库同步；对照模式为服务端原图 / 结果并排渲染。');
    $('detailNotice').textContent = note;
    if (item.error) $('detailNotice').classList.add('warning');
    const current = () => version === state.modalVersion && state.detail === item && image.dataset.requestUrl === url && $('detailImageWrap').contains(image);
    image.onload = () => { if (current()) placeholder.remove(); };
    image.onerror = () => {
      if (!current()) return;
      if (!image.dataset.fallback && effectiveMode(item) !== 'input') {
        image.dataset.fallback = 'input';
        $('detailNotice').textContent = '结果预览不可用，已回退到原图。'; $('detailNotice').classList.add('warning');
        image.src = `/preview/${encodeURIComponent(item.id)}?mode=input&width=${state.zoom ? 1920 : 960}`;
      } else if (image.dataset.fallback !== 'original' && safeImageUrl(item.image_url)) {
        image.dataset.fallback = 'original'; image.src = safeImageUrl(item.image_url);
      } else {
        image.remove(); placeholder.textContent = '图像无法读取或已不存在';
        $('detailNotice').textContent = '请检查图像文件，或重新扫描目录后重试。'; $('detailNotice').classList.add('warning');
      }
    };
    $('detailImageWrap').append(image); image.src = url;
  }
  function navigateDetail(direction) {
    if (!$('detailModal').open) return;
    const index = state.items.findIndex(row => row.id === state.modalId), next = index + direction;
    if (index >= 0 && next >= 0 && next < state.items.length) loadDetail(state.items[next].id);
  }
  function viewChanged(immediate = true) {
    syncViewControls(); savePreferences(); clearTimeout(state.viewTimer);
    const render = () => { for (const card of state.cards.values()) updateCardPreview(card); renderDetailPreview(); };
    if (immediate) render(); else state.viewTimer = setTimeout(render, 180);
  }

  $('filterForm').onsubmit = event => { event.preventDefault(); filtersChanged(); };
  for (const name of filterNames) $(name).addEventListener(name === 'q' || name === 'min_score' ? 'input' : 'change', () => filtersChanged(name === 'q' || name === 'min_score' ? 300 : 0));
  $('resetFilters').onclick = $('emptyReset').onclick = resetFilters;
  $('toggleFilters').onclick = () => { const open = $('filterPane').classList.toggle('open'); $('toggleFilters').setAttribute('aria-expanded', String(open)); };
  $('selectPage').onchange = () => { state.selected = new Set($('selectPage').checked ? state.items.map(row => row.id) : []); updateSelection(); };
  $('clearSelection').onclick = () => { state.selected.clear(); updateSelection(); };
  $('skipCompleted').onchange = savePreferences;
  $('runPage').onclick = () => runJob(false); $('runFiltered').onclick = () => runJob(true);
  $('stopJob').onclick = () => operation('stop'); $('refresh').onclick = () => operation('refresh');
  $('retry').onclick = () => { $('errorBanner').hidden = true; loadSummary({forceList: true}); };
  $('dismissError').onclick = () => { $('errorBanner').hidden = true; };
  $('pagePrev').onclick = () => goPage(Math.floor(state.offset / state.limit));
  $('pageNext').onclick = () => goPage(Math.floor(state.offset / state.limit) + 2);
  $('pageGo').onclick = () => goPage(Number($('pageInput').value));
  $('pageInput').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); goPage(Number($('pageInput').value)); } };
  $('pageSize').onchange = () => { state.limit = Number($('pageSize').value); state.offset = 0; state.selected.clear(); closeDetail(); savePreferences(); loadItems(); };
  $('columns').onchange = () => { state.columns = Number($('columns').value); $('gallery').style.setProperty('--columns', state.columns); savePreferences(); };
  document.querySelectorAll('[data-mode]').forEach(button => { button.onclick = () => { state.view.mode = button.dataset.mode; viewChanged(); }; });
  for (const id of ['source', 'detailSource']) $(id).onchange = () => { state.view.source = $(id).value; viewChanged(); };
  for (const id of ['alpha', 'detailAlpha']) $(id).oninput = () => { state.view.alpha = $(id).value; viewChanged(false); };
  for (const id of ['unknown', 'detailUnknown']) $(id).onchange = () => { state.view.unknown = $(id).checked; viewChanged(); };
  $('detailZoom').onclick = () => { state.zoom = !state.zoom; syncViewControls(); $('detailStage').scrollTo(0, 0); renderDetailPreview(); };
  $('closeDetail').onclick = closeDetail;
  $('detailPrev').onclick = () => navigateDetail(-1); $('detailNext').onclick = () => navigateDetail(1);
  $('detailModal').addEventListener('cancel', event => { event.preventDefault(); closeDetail(); });
  $('detailModal').addEventListener('close', () => { if (state.modalId) closeDetail(); });
  $('detailModal').addEventListener('click', event => {
    if (event.target !== $('detailModal')) return;
    const rect = $('detailModal').getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDetail();
  });
  document.addEventListener('keydown', event => {
    if (!$('detailModal').open || event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.key === 'Escape') { event.preventDefault(); closeDetail(); return; }
    if (event.target.closest('input, select, textarea, [contenteditable=true]')) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); navigateDetail(event.key === 'ArrowLeft' ? -1 : 1); }
  });
  let drag = null;
  $('detailStage').addEventListener('pointerdown', event => {
    if (!state.zoom || event.button !== 0 || event.pointerType === 'touch') return;
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY, left: $('detailStage').scrollLeft, top: $('detailStage').scrollTop};
    $('detailStage').setPointerCapture(event.pointerId); $('detailStage').classList.add('dragging'); event.preventDefault();
  });
  $('detailStage').addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    $('detailStage').scrollLeft = drag.left - event.clientX + drag.x;
    $('detailStage').scrollTop = drag.top - event.clientY + drag.y;
  });
  const endDrag = () => { drag = null; $('detailStage').classList.remove('dragging'); };
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) $('detailStage').addEventListener(event, endDrag);
  window.addEventListener('pagehide', () => {
    clearTimeout(state.pollTimer); state.listController?.abort(); state.summaryController?.abort(); state.modalController?.abort();
  });

  restorePreferences(); syncViewControls(); state.filters = readFilters(); updateActions(); loadSummary();
})();
