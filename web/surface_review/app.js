const $ = (id) => document.getElementById(id);
const viewNames = { palm: "手心", back: "手背", side: "侧面", unknown: "无法判断", invalid: "标注无效" };
const viewColors = { palm: "#2b8568", back: "#c16d48", side: "#a88332", unknown: "#838e9b", invalid: "#a3424c" };
const state = { items: [], index: 0, mode: "full", meta: null, toastTimer: null, imageToken: 0, inferToken: 0, inference: null, models: [] };

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => element.classList.remove("show"), 3200);
}

async function request(url, options) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("Content-Type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(data?.error || `请求失败 (${response.status})`);
  return data;
}

function newSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}

function settingsFromUrl() {
  const params = new URLSearchParams(location.search);
  return {
    split: params.get("split") || "all",
    view: params.get("view") || "all",
    size: params.get("size") || "100",
    mode: params.get("mode") || "balanced",
    seed: params.get("seed") || String(newSeed()),
    item: Number(params.get("item") || 0),
  };
}

function readSettings() {
  return {
    split: $("split-filter").value,
    view: $("view-filter").value,
    size: $("sample-size").value,
    mode: document.querySelector('input[name="mode"]:checked').value,
    seed: $("sample-seed").value,
  };
}

function setSettings(settings) {
  $("split-filter").value = settings.split;
  $("view-filter").value = settings.view;
  $("sample-size").value = settings.size;
  $("sample-seed").value = settings.seed;
  const mode = document.querySelector(`input[name="mode"][value="${settings.mode}"]`);
  (mode || document.querySelector('input[name="mode"][value="balanced"]')).checked = true;
}

function syncUrl() {
  const settings = readSettings();
  const params = new URLSearchParams({ ...settings, item: String(state.index) });
  history.replaceState(null, "", `?${params}`);
}

async function loadSample(preferredIndex = 0) {
  const settings = readSettings();
  if (!$("sampling-form").reportValidity()) return;
  const query = new URLSearchParams(settings);
  const result = await request(`/api/sample?${query}`);
  state.items = result.items;
  state.index = Math.min(Math.max(0, preferredIndex), Math.max(0, state.items.length - 1));
  $("queue-summary").textContent = `抽样池 ${result.pool_size.toLocaleString()} · 当前批次 ${state.items.length}`;
  render();
  syncUrl();
}

function current() { return state.items[state.index] || null; }

function renderQueue() {
  const list = $("queue-list");
  const scroll = list.scrollTop;
  list.replaceChildren();
  const done = state.items.filter((item) => item.review).length;
  $("queue-count").textContent = `${done} / ${state.items.length}`;
  if (!state.items.length) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = "当前条件没有可审核的标注。";
    list.append(empty);
    return;
  }
  state.items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `queue-item${index === state.index ? " active" : ""}`;
    button.setAttribute("aria-current", index === state.index ? "true" : "false");
    const number = document.createElement("span");
    number.className = "queue-index";
    number.textContent = String(index + 1).padStart(3, "0");
    const copy = document.createElement("span");
    copy.className = "queue-copy";
    const title = document.createElement("strong");
    title.textContent = `${viewNames[item.candidate_view]} · ${item.handedness === "left" ? "左手" : "右手"}`;
    const subtitle = document.createElement("small");
    subtitle.textContent = `${item.split.toUpperCase()} · #${item.id}`;
    copy.append(title, subtitle);
    const status = document.createElement("span");
    status.className = `queue-state${item.review ? " done" : ""}`;
    status.title = item.review ? "已审核" : "待审核";
    button.append(number, copy, status);
    button.addEventListener("click", () => {
      selectItem(index);
      if (matchMedia("(max-width: 760px)").matches) setMobileTab("image");
    });
    list.append(button);
  });
  list.scrollTop = scroll;
  list.children[state.index]?.scrollIntoView({ block: "nearest" });
}

function fitFullFigure(item) {
  const stage = $("image-stage");
  const scale = Math.min((stage.clientWidth - 32) / item.width, (stage.clientHeight - 32) / item.height);
  if (!Number.isFinite(scale) || scale <= 0) return;
  $("full-figure").style.width = `${Math.floor(item.width * scale)}px`;
  $("full-figure").style.height = `${Math.floor(item.height * scale)}px`;
}

function fitCropCanvas() {
  const stage = $("image-stage");
  const size = Math.max(0, Math.min(stage.clientWidth - 32, stage.clientHeight - 32, 640));
  $("crop-canvas").style.width = `${size}px`;
  $("crop-canvas").style.height = `${size}px`;
}

function drawCrop(image, item) {
  const canvas = $("crop-canvas");
  const context = canvas.getContext("2d");
  const [x, y, width, height] = item.bbox;
  const cropSize = Math.max(width, height) * 2.25;
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  context.fillStyle = "#263430";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, centerX - cropSize / 2, centerY - cropSize / 2, cropSize, cropSize, 0, 0, canvas.width, canvas.height);
  const scale = canvas.width / cropSize;
  context.strokeStyle = "#f4bf64";
  context.lineWidth = 3;
  context.strokeRect((x - centerX + cropSize / 2) * scale, (y - centerY + cropSize / 2) * scale, width * scale, height * scale);
  if (state.inference?.image_id === item.image_id) {
    state.inference.models.forEach((model) => {
      const slot = $("model-a").value === model.id ? 0 : 1;
      context.save();
      context.strokeStyle = slot === 0 ? "#16a6a0" : "#e05377";
      context.lineWidth = 3;
      if (slot === 1) context.setLineDash([9, 6]);
      for (const detection of model.detections) {
        const [x1, y1, x2, y2] = detection.bbox_xyxy;
        context.strokeRect((x1 - centerX + cropSize / 2) * scale, (y1 - centerY + cropSize / 2) * scale, (x2 - x1) * scale, (y2 - y1) * scale);
      }
      context.restore();
    });
  }
}

function renderPredictions() {
  const container = $("prediction-overlays");
  const legend = $("model-legend");
  container.replaceChildren();
  legend.replaceChildren();
  const groundTruth = document.createElement("span");
  groundTruth.className = "ground-truth-key";
  groundTruth.textContent = "原标注";
  legend.append(groundTruth);
  const item = current();
  if (!item) return;
  const image = $("full-image");
  if (state.inference?.image_id !== item.image_id) {
    if (image.complete && image.naturalWidth) drawCrop(image, item);
    return;
  }
  state.inference.models.forEach((model) => {
    const slot = $("model-a").value === model.id ? 0 : 1;
    const key = document.createElement("span");
    key.className = `model-result-key slot-${slot === 0 ? "a" : "b"}`;
    const top = model.detections[0];
    const name = top ? ({ palm: "手", left_palm: "左手", right_palm: "右手" }[top.class_name] || top.class_name) : "无检测";
    key.textContent = `${slot === 0 ? "A" : "B"} · ${name}${top ? ` ${top.score.toFixed(2)}` : ""}${model.detections.length > 1 ? ` · ${model.detections.length} 个` : ""}`;
    key.title = model.label;
    legend.append(key);
    for (const detection of model.detections) {
      const [x1, y1, x2, y2] = detection.bbox_xyxy;
      const box = document.createElement("div");
      box.className = `prediction-overlay slot-${slot === 0 ? "a" : "b"}`;
      box.style.left = `${x1 / item.width * 100}%`;
      box.style.top = `${y1 / item.height * 100}%`;
      box.style.width = `${(x2 - x1) / item.width * 100}%`;
      box.style.height = `${(y2 - y1) / item.height * 100}%`;
      const label = document.createElement("span");
      const name = { palm: "手", left_palm: "左手", right_palm: "右手" }[detection.class_name] || detection.class_name;
      label.textContent = `${slot === 0 ? "A" : "B"} ${name} ${detection.score.toFixed(2)}`;
      box.append(label);
      container.append(box);
    }
  });
  if (image.complete && image.naturalWidth) drawCrop(image, item);
}

function selectedModels() {
  return [$("model-a").value, $("model-b").value].filter(Boolean);
}

function renderModelIdentity() {
  for (const slot of ["a", "b"]) {
    const model = state.models.find((item) => item.id === $(`model-${slot}`).value);
    $(`model-${slot}-identity`).textContent = model?.label || "未选择模型";
    $(`model-${slot}-identity`).title = model?.run_id || "";
    $(`model-${slot}`).title = model ? `${model.label}\nRun ID: ${model.run_id}` : "不显示模型推理结果";
  }
}

async function runInference() {
  const token = ++state.inferToken;
  const item = current();
  state.inference = null;
  renderPredictions();
  const modelIds = selectedModels();
  if (!item || !modelIds.length) return;
  const status = document.createElement("span");
  status.textContent = "推理中…";
  $("model-legend").append(status);
  try {
    const result = await request("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: item.image_id, model_ids: modelIds, score_threshold: Number($("score-threshold").value) }),
    });
    if (token !== state.inferToken || current()?.image_id !== item.image_id) return;
    state.inference = { image_id: item.image_id, models: result.models };
    renderPredictions();
  } catch (error) {
    if (token !== state.inferToken) return;
    renderPredictions();
    toast(error.message);
  }
}

async function loadModels() {
  const result = await request("/api/models");
  state.models = result.models;
  for (const select of [$("model-a"), $("model-b")]) {
    for (const model of state.models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      option.title = model.id;
      select.append(option);
    }
  }
  let preferred = [];
  try { preferred = JSON.parse(localStorage.getItem("surfaceReviewModels") || "[]"); } catch { /* Ignore stale browser settings. */ }
  $("model-a").value = state.models.some((model) => model.id === preferred[0]) ? preferred[0] : (state.models[0]?.id || "");
  $("model-b").value = state.models.some((model) => model.id === preferred[1] && model.id !== $("model-a").value) ? preferred[1] : "";
  renderModelIdentity();
}

function renderImage(item) {
  const token = ++state.imageToken;
  const figure = $("full-figure");
  const image = $("full-image");
  const canvas = $("crop-canvas");
  if (!item) {
    figure.hidden = true;
    canvas.hidden = true;
    return;
  }
  const [x, y, width, height] = item.bbox;
  const box = $("bbox-overlay");
  box.style.left = `${x / item.width * 100}%`;
  box.style.top = `${y / item.height * 100}%`;
  box.style.width = `${width / item.width * 100}%`;
  box.style.height = `${height / item.height * 100}%`;
  $("box-label").textContent = `#${item.id}`;
  fitFullFigure(item);
  fitCropCanvas();
  image.onload = () => { if (token === state.imageToken) drawCrop(image, item); };
  image.onerror = () => { if (token === state.imageToken) toast("图片加载失败"); };
  image.src = `/api/image/${item.image_id}`;
  figure.hidden = state.mode !== "full";
  canvas.hidden = state.mode !== "crop";
  $("image-source").textContent = item.source_image;
  $("image-shape").textContent = `${item.width} × ${item.height} px`;
}

function renderReview(item) {
  const status = $("review-status");
  const checked = document.querySelector('input[name="verdict"]:checked');
  if (checked) checked.checked = false;
  $("review-note").value = item?.review?.note || "";
  $("candidate-label").textContent = item ? viewNames[item.candidate_view] : "—";
  $("candidate-indicator").style.background = item ? viewColors[item.candidate_view] : "#929f9b";
  $("candidate-meta").textContent = item
    ? `当前字段：${viewNames[item.view]}${item.facing == null ? "" : ` · 朝向分数 ${item.facing.toFixed(2)}`}`
    : "—";
  $("handedness-value").textContent = item ? (item.handedness === "left" ? "左手" : "右手") : "—";
  $("annotation-id").textContent = item ? `#${item.id}` : "—";
  status.textContent = item?.review ? "已审核" : "待审核";
  status.classList.toggle("done", Boolean(item?.review));
  if (item?.review) {
    const chosen = document.querySelector(`input[name="verdict"][value="${item.review.view}"]`);
    if (chosen) chosen.checked = true;
    $("saved-detail").textContent = `${item.review.reviewer} · ${new Date(item.review.reviewed_at_utc).toLocaleString("zh-CN")}`;
  } else {
    $("saved-detail").textContent = "";
  }
  $("save-review").disabled = !item;
}

function render() {
  const item = current();
  state.inference = null;
  renderQueue();
  $("item-split").textContent = item ? `${item.split.toUpperCase()} · ${item.handedness === "left" ? "LEFT" : "RIGHT"}` : "—";
  $("item-title").textContent = item ? item.source_image : "没有样本";
  $("item-position").textContent = `${item ? state.index + 1 : 0} / ${state.items.length}`;
  $("previous-item").disabled = state.index <= 0 || !item;
  $("next-item").disabled = state.index >= state.items.length - 1 || !item;
  renderImage(item);
  runInference();
  renderReview(item);
}

function selectItem(index) {
  if (index < 0 || index >= state.items.length) return;
  state.index = index;
  render();
  syncUrl();
}

function setImageMode(mode) {
  state.mode = mode;
  $("show-crop").classList.toggle("selected", mode === "crop");
  $("show-full").classList.toggle("selected", mode === "full");
  $("show-crop").setAttribute("aria-pressed", mode === "crop");
  $("show-full").setAttribute("aria-pressed", mode === "full");
  renderImage(current());
}

function setMobileTab(tab) {
  document.body.dataset.mobileTab = tab;
  document.querySelectorAll(".mobile-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  if (tab === "image" && current()) {
    fitFullFigure(current());
    fitCropCanvas();
  }
}

function setFiltersOpen(open) {
  document.body.classList.toggle("filters-open", open);
  $("mobile-filter-toggle").setAttribute("aria-expanded", String(open));
}

async function saveReview(event) {
  event.preventDefault();
  const item = current();
  if (!item) return;
  const reviewer = $("reviewer-name").value.trim();
  const view = document.querySelector('input[name="verdict"]:checked')?.value;
  if (!reviewer) { $("reviewer-name").focus(); return toast("请填写审核人"); }
  if (!view) return toast("请选择实际可见朝向");
  const button = $("save-review");
  button.disabled = true;
  try {
    const firstReview = !item.review;
    item.review = await request("/api/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotation_id: item.id, reviewer, view, note: $("review-note").value }),
    });
    localStorage.setItem("surfaceReviewer", reviewer);
    if (firstReview && state.meta) {
      state.meta.reviewed += 1;
      $("global-progress").textContent = `已审核 ${state.meta.reviewed.toLocaleString()} / ${state.meta.total.toLocaleString()}`;
    }
    toast("审核已保存");
    const next = state.items.findIndex((candidate, index) => index > state.index && !candidate.review);
    if (next >= 0) selectItem(next);
    else render();
  } catch (error) {
    button.disabled = false;
    toast(error.message);
  }
}

async function boot() {
  try {
    state.meta = await request("/api/metadata");
    await loadModels();
    $("dataset-name").textContent = state.meta.dataset;
    $("global-progress").textContent = `已审核 ${state.meta.reviewed.toLocaleString()} / ${state.meta.total.toLocaleString()}`;
    $("reviewer-name").value = localStorage.getItem("surfaceReviewer") || "";
    const settings = settingsFromUrl();
    setSettings(settings);
    await loadSample(settings.item);
  } catch (error) { toast(error.message); }
}

$("sampling-form").addEventListener("submit", (event) => {
  event.preventDefault();
  loadSample().then(() => setFiltersOpen(false)).catch((error) => toast(error.message));
});
$("new-sample").addEventListener("click", () => {
  $("sample-seed").value = String(newSeed());
  loadSample().then(() => setFiltersOpen(false)).catch((error) => toast(error.message));
});
$("mobile-filter-toggle").addEventListener("click", () => setFiltersOpen(!document.body.classList.contains("filters-open")));
$("previous-item").addEventListener("click", () => selectItem(state.index - 1));
$("next-item").addEventListener("click", () => selectItem(state.index + 1));
$("show-crop").addEventListener("click", () => setImageMode("crop"));
$("show-full").addEventListener("click", () => setImageMode("full"));
$("review-form").addEventListener("submit", saveReview);
for (const select of [$("model-a"), $("model-b")]) {
  select.addEventListener("change", () => {
    if ($("model-a").value && $("model-a").value === $("model-b").value) {
      select.value = "";
      toast("同一模型只需选择一次");
    }
    localStorage.setItem("surfaceReviewModels", JSON.stringify([$("model-a").value, $("model-b").value]));
    renderModelIdentity();
    runInference();
  });
}
$("score-threshold").addEventListener("input", () => { $("threshold-value").textContent = Number($("score-threshold").value).toFixed(2); });
$("score-threshold").addEventListener("change", runInference);
document.querySelectorAll(".mobile-tabs button").forEach((button) => button.addEventListener("click", () => setMobileTab(button.dataset.tab)));
window.addEventListener("resize", () => {
  if (current()) {
    fitFullFigure(current());
    fitCropCanvas();
  }
});
new ResizeObserver(() => {
  if (current()) {
    fitFullFigure(current());
    fitCropCanvas();
  }
}).observe($("image-stage"));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setFiltersOpen(false);
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.key === "ArrowLeft") selectItem(state.index - 1);
  if (event.key === "ArrowRight") selectItem(state.index + 1);
});

boot();
