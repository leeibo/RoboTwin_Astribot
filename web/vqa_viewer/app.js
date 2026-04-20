const state = {
  index: null,
  selectedTaskKey: null,
  selectedType: "object_search",
  selectedSampleIdx: null,
  taskSearch: "",
  storageFilter: "all",
  sampleSearch: "",
  sampleEpisode: "all",
  sampleStage: "all",
  sampleSubtask: "all",
  historyOnly: false,
  selectedImageIdx: null,
};

const caches = {
  sampleLists: new Map(),
  sampleDetails: new Map(),
};

const elements = {
  serverStatus: document.getElementById("serverStatus"),
  refreshIndexButton: document.getElementById("refreshIndexButton"),
  globalStats: document.getElementById("globalStats"),
  taskSearchInput: document.getElementById("taskSearchInput"),
  storageSelect: document.getElementById("storageSelect"),
  taskList: document.getElementById("taskList"),
  taskCountBadge: document.getElementById("taskCountBadge"),
  selectionTitle: document.getElementById("selectionTitle"),
  selectionSubtitle: document.getElementById("selectionSubtitle"),
  taskTypeTabs: document.getElementById("taskTypeTabs"),
  sampleSearchInput: document.getElementById("sampleSearchInput"),
  episodeFilter: document.getElementById("episodeFilter"),
  stageFilter: document.getElementById("stageFilter"),
  subtaskFilter: document.getElementById("subtaskFilter"),
  historyOnlyCheckbox: document.getElementById("historyOnlyCheckbox"),
  sampleSummary: document.getElementById("sampleSummary"),
  prevSampleButton: document.getElementById("prevSampleButton"),
  nextSampleButton: document.getElementById("nextSampleButton"),
  sampleList: document.getElementById("sampleList"),
  detailTitle: document.getElementById("detailTitle"),
  detailMetaLine: document.getElementById("detailMetaLine"),
  detailStatChips: document.getElementById("detailStatChips"),
  copySampleJsonButton: document.getElementById("copySampleJsonButton"),
  copyPromptButton: document.getElementById("copyPromptButton"),
  imagePanelHint: document.getElementById("imagePanelHint"),
  imageStage: document.getElementById("imageStage"),
  thumbnailRail: document.getElementById("thumbnailRail"),
  parsedResponse: document.getElementById("parsedResponse"),
  videoGrid: document.getElementById("videoGrid"),
  userMessage: document.getElementById("userMessage"),
  assistantMessage: document.getElementById("assistantMessage"),
  metadataGrid: document.getElementById("metadataGrid"),
  rawSampleJson: document.getElementById("rawSampleJson"),
  lightbox: document.getElementById("lightbox"),
  lightboxImage: document.getElementById("lightboxImage"),
  lightboxCaption: document.getElementById("lightboxCaption"),
  lightboxClose: document.getElementById("lightboxClose"),
};

function taskKey(taskName, storageName) {
  return `${taskName}::${storageName}`;
}

function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}

async function fetchJson(path, params = {}) {
  const response = await fetch(apiUrl(path, params));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
}

function parseUrlState() {
  const params = new URLSearchParams(window.location.search);
  state.selectedTaskKey = params.get("task_key");
  state.selectedType = params.get("type") || state.selectedType;
  state.selectedSampleIdx = params.has("sample_idx") ? Number(params.get("sample_idx")) : null;
  state.taskSearch = params.get("task_search") || "";
  state.storageFilter = params.get("storage") || "all";
  state.sampleSearch = params.get("sample_search") || "";
  state.sampleEpisode = params.get("episode") || "all";
  state.sampleStage = params.get("stage") || "all";
  state.sampleSubtask = params.get("subtask") || "all";
  state.historyOnly = params.get("history_only") === "1";
}

function syncUrlState() {
  const params = new URLSearchParams();
  if (state.selectedTaskKey) params.set("task_key", state.selectedTaskKey);
  if (state.selectedType) params.set("type", state.selectedType);
  if (Number.isInteger(state.selectedSampleIdx)) params.set("sample_idx", String(state.selectedSampleIdx));
  if (state.taskSearch) params.set("task_search", state.taskSearch);
  if (state.storageFilter && state.storageFilter !== "all") params.set("storage", state.storageFilter);
  if (state.sampleSearch) params.set("sample_search", state.sampleSearch);
  if (state.sampleEpisode !== "all") params.set("episode", state.sampleEpisode);
  if (state.sampleStage !== "all") params.set("stage", state.sampleStage);
  if (state.sampleSubtask !== "all") params.set("subtask", state.sampleSubtask);
  if (state.historyOnly) params.set("history_only", "1");
  const query = params.toString();
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
  window.history.replaceState(null, "", nextUrl);
}

function selectedTask() {
  if (!state.index || !state.selectedTaskKey) return null;
  return state.index.tasks.find((task) => taskKey(task.task_name, task.storage_name) === state.selectedTaskKey) || null;
}

function sampleListCacheKey() {
  const task = selectedTask();
  if (!task) return null;
  return `${taskKey(task.task_name, task.storage_name)}::${state.selectedType}`;
}

function detailCacheKey(sampleIdx) {
  const base = sampleListCacheKey();
  return base ? `${base}::${sampleIdx}` : null;
}

function setStatus(text, tone = "neutral") {
  elements.serverStatus.textContent = text;
  elements.serverStatus.dataset.tone = tone;
}

function renderGlobalStats() {
  if (!state.index) return;
  const cards = [
    { label: "Tasks", value: state.index.task_count },
    { label: "Storages", value: state.index.storages.length },
    { label: "Object Search", value: state.index.totals.object_search || 0 },
    { label: "Angle Delta", value: state.index.totals.angle_delta || 0 },
    { label: "Compression", value: state.index.totals.memory_compression_vqa || 0 },
  ];
  elements.globalStats.innerHTML = cards
    .map(
      (card) => `
        <div class="stat-card">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
        </div>
      `
    )
    .join("");
}

function filteredTasks() {
  if (!state.index) return [];
  const needle = state.taskSearch.trim().toLowerCase();
  return state.index.tasks.filter((task) => {
    if (state.storageFilter !== "all" && task.storage_name !== state.storageFilter) {
      return false;
    }
    if (!needle) return true;
    return `${task.task_name} ${task.storage_name}`.toLowerCase().includes(needle);
  });
}

function renderStorageOptions() {
  if (!state.index) return;
  const options = ['<option value="all">全部存储目录</option>'];
  for (const storage of state.index.storages) {
    options.push(`<option value="${storage}">${storage}</option>`);
  }
  elements.storageSelect.innerHTML = options.join("");
  elements.storageSelect.value = state.storageFilter;
}

function renderTaskList() {
  const tasks = filteredTasks();
  elements.taskCountBadge.textContent = String(tasks.length);
  if (!tasks.length) {
    elements.taskList.innerHTML = '<div class="empty-state">没有匹配的任务</div>';
    return;
  }
  elements.taskList.innerHTML = tasks
    .map((task) => {
      const active = taskKey(task.task_name, task.storage_name) === state.selectedTaskKey ? "active" : "";
      const chips = [
        `episodes ${task.episode_count}`,
        `object ${task.sample_counts.object_search || 0}`,
        `angle ${task.sample_counts.angle_delta || 0}`,
        `compress ${task.sample_counts.memory_compression_vqa || 0}`,
      ];
      return `
        <article class="task-card ${active}" data-task-key="${taskKey(task.task_name, task.storage_name)}">
          <div class="task-card-title">${task.task_name}</div>
          <div class="task-card-subtitle">${task.storage_name}</div>
          <div class="task-card-stats">
            ${chips.map((chip) => `<span class="chip">${chip}</span>`).join("")}
          </div>
        </article>
      `;
    })
    .join("");

  for (const card of elements.taskList.querySelectorAll(".task-card")) {
    card.addEventListener("click", () => {
      const key = card.dataset.taskKey;
      if (!key || key === state.selectedTaskKey) return;
      state.selectedTaskKey = key;
      state.selectedSampleIdx = null;
      state.sampleEpisode = "all";
      state.sampleStage = "all";
      state.sampleSubtask = "all";
      state.sampleSearch = "";
      elements.sampleSearchInput.value = "";
      syncUrlState();
      renderTaskList();
      renderHeader();
      renderTypeTabs();
      loadCurrentSampleList();
    });
  }
}

function renderHeader() {
  const task = selectedTask();
  if (!task) {
    elements.selectionTitle.textContent = "未选择任务";
    elements.selectionSubtitle.textContent = "";
    return;
  }
  elements.selectionTitle.textContent = task.task_name;
  elements.selectionSubtitle.textContent = `${task.storage_name} · ${task.episode_count} episodes · object ${task.sample_counts.object_search || 0} · angle ${task.sample_counts.angle_delta || 0} · compression ${task.sample_counts.memory_compression_vqa || 0}`;
}

function renderTypeTabs() {
  const task = selectedTask();
  if (!task) {
    elements.taskTypeTabs.innerHTML = "";
    return;
  }
  const available = state.index.task_types;
  elements.taskTypeTabs.innerHTML = available
    .map((type) => {
      const count = task.sample_counts[type] || 0;
      const active = type === state.selectedType ? "active" : "";
      const labelMap = {
        object_search: "Object Search",
        angle_delta: "Angle Delta",
        memory_compression_vqa: "Memory Compression",
      };
      return `
        <button class="type-tab ${active}" data-type="${type}" type="button">
          ${labelMap[type] || type}
          <small>${count} samples</small>
        </button>
      `;
    })
    .join("");
  for (const button of elements.taskTypeTabs.querySelectorAll(".type-tab")) {
    button.addEventListener("click", () => {
      const nextType = button.dataset.type;
      if (!nextType || nextType === state.selectedType) return;
      state.selectedType = nextType;
      state.selectedSampleIdx = null;
      syncUrlState();
      renderTypeTabs();
      loadCurrentSampleList();
    });
  }
}

async function loadIndex(forceRefresh = false) {
  setStatus("Syncing…", "neutral");
  const payload = await fetchJson("/api/index", forceRefresh ? { refresh: 1 } : {});
  state.index = payload;
  if (!state.selectedTaskKey) {
    const firstTask = payload.tasks[0];
    if (firstTask) {
      state.selectedTaskKey = taskKey(firstTask.task_name, firstTask.storage_name);
    }
  }
  if (state.selectedTaskKey) {
    const taskExists = payload.tasks.some((task) => taskKey(task.task_name, task.storage_name) === state.selectedTaskKey);
    if (!taskExists && payload.tasks[0]) {
      state.selectedTaskKey = taskKey(payload.tasks[0].task_name, payload.tasks[0].storage_name);
    }
  }
  renderGlobalStats();
  renderStorageOptions();
  renderTaskList();
  renderHeader();
  renderTypeTabs();
  setStatus("Ready", "ok");
  syncUrlState();
}

async function loadCurrentSampleList() {
  const task = selectedTask();
  if (!task) return;
  const cacheKey = sampleListCacheKey();
  if (!cacheKey) return;

  let payload = caches.sampleLists.get(cacheKey);
  if (!payload) {
    elements.sampleList.innerHTML = '<div class="empty-state">正在加载样本…</div>';
    payload = await fetchJson("/api/samples", {
      task: task.task_name,
      storage: task.storage_name,
      type: state.selectedType,
    });
    caches.sampleLists.set(cacheKey, payload);
  }
  renderSampleFilters(payload);
  renderSampleList(payload);
}

function renderSelectOptions(select, values, labelFn = (value) => value) {
  const options = ['<option value="all">全部</option>'];
  for (const value of values) {
    options.push(`<option value="${value}">${labelFn(value)}</option>`);
  }
  select.innerHTML = options.join("");
}

function renderSampleFilters(payload) {
  const samples = payload.samples || [];
  const episodes = [...new Set(samples.map((item) => item.episode_idx).filter((item) => item !== null))].sort((a, b) => a - b);
  const stages = [...new Set(samples.map((item) => item.stage).filter((item) => item !== null))].sort((a, b) => a - b);
  const subtasks = [...new Set(samples.map((item) => item.subtask_id).filter((item) => item !== null))].sort((a, b) => a - b);
  renderSelectOptions(elements.episodeFilter, episodes, (value) => `Episode ${value}`);
  renderSelectOptions(elements.stageFilter, stages, (value) => `Stage ${value}`);
  renderSelectOptions(elements.subtaskFilter, subtasks, (value) => `Subtask ${value}`);
  elements.episodeFilter.value = state.sampleEpisode;
  elements.stageFilter.value = state.sampleStage;
  elements.subtaskFilter.value = state.sampleSubtask;
  elements.historyOnlyCheckbox.checked = state.historyOnly;
}

function currentFilteredSamples(payload) {
  const needle = state.sampleSearch.trim().toLowerCase();
  return (payload.samples || []).filter((sample) => {
    if (state.sampleEpisode !== "all" && String(sample.episode_idx) !== state.sampleEpisode) return false;
    if (state.sampleStage !== "all" && String(sample.stage) !== state.sampleStage) return false;
    if (state.sampleSubtask !== "all" && String(sample.subtask_id) !== state.sampleSubtask) return false;
    if (state.historyOnly && !sample.evidence_from_history) return false;
    if (!needle) return true;
    const haystack = `${sample.user_preview} ${sample.assistant_preview} ${sample.task_type}`.toLowerCase();
    return haystack.includes(needle);
  });
}

function ensureSelectedSample(filteredSamples) {
  if (!filteredSamples.length) {
    state.selectedSampleIdx = null;
    return;
  }
  const exists = filteredSamples.some((sample) => sample.sample_idx === state.selectedSampleIdx);
  if (!exists) {
    state.selectedSampleIdx = filteredSamples[0].sample_idx;
  }
}

function renderSampleList(payload) {
  const filteredSamples = currentFilteredSamples(payload);
  ensureSelectedSample(filteredSamples);
  elements.sampleSummary.textContent = `${filteredSamples.length} / ${payload.sample_count} samples`;

  if (!filteredSamples.length) {
    elements.sampleList.innerHTML = '<div class="empty-state">当前筛选条件下没有样本</div>';
    clearDetail();
    return;
  }

  elements.sampleList.innerHTML = filteredSamples
    .map((sample) => {
      const active = sample.sample_idx === state.selectedSampleIdx ? "active" : "";
      const chips = [
        sample.episode_idx !== null ? `ep ${sample.episode_idx}` : null,
        sample.current_frame_idx !== null ? `frame ${sample.current_frame_idx}` : null,
        sample.stage !== null ? `stage ${sample.stage}` : null,
        sample.subtask_id !== null ? `subtask ${sample.subtask_id}` : null,
        sample.prompt_image_count ? `${sample.prompt_image_count} imgs` : null,
      ].filter(Boolean);
      if (sample.evidence_from_history) {
        chips.push("history evidence");
      }
      return `
        <article class="sample-card ${active}" data-sample-idx="${sample.sample_idx}">
          <div class="sample-card-title">#${sample.sample_idx} · ${sample.task_type}</div>
          <div class="sample-card-meta">${chips.map((chip) => `<span class="chip">${chip}</span>`).join("")}</div>
          <p class="sample-preview">${sample.assistant_preview || sample.user_preview || "No preview"}</p>
        </article>
      `;
    })
    .join("");

  for (const card of elements.sampleList.querySelectorAll(".sample-card")) {
    card.addEventListener("click", () => {
      const idx = Number(card.dataset.sampleIdx);
      if (Number.isNaN(idx)) return;
      state.selectedSampleIdx = idx;
      syncUrlState();
      renderSampleList(payload);
      loadSelectedDetail();
    });
  }

  syncUrlState();
  loadSelectedDetail();
}

function clearDetail() {
  elements.detailTitle.textContent = "选择一个样本";
  elements.detailMetaLine.textContent = "";
  elements.detailStatChips.innerHTML = "";
  elements.imagePanelHint.textContent = "";
  elements.imageStage.innerHTML = "选择样本后预览图片";
  elements.thumbnailRail.innerHTML = "";
  elements.parsedResponse.innerHTML = "暂无内容";
  elements.videoGrid.innerHTML = "当前样本暂无视频可预览";
  elements.userMessage.textContent = "";
  elements.assistantMessage.textContent = "";
  elements.metadataGrid.innerHTML = "";
  elements.rawSampleJson.textContent = "";
}

async function loadSelectedDetail() {
  const task = selectedTask();
  if (!task || !Number.isInteger(state.selectedSampleIdx)) {
    clearDetail();
    return;
  }
  const cacheKey = detailCacheKey(state.selectedSampleIdx);
  if (!cacheKey) return;
  let payload = caches.sampleDetails.get(cacheKey);
  if (!payload) {
    payload = await fetchJson("/api/sample-detail", {
      task: task.task_name,
      storage: task.storage_name,
      type: state.selectedType,
      index: state.selectedSampleIdx,
    });
    caches.sampleDetails.set(cacheKey, payload);
  }
  renderDetail(payload);
}

function chip(label, variant = "") {
  return `<span class="chip ${variant}">${label}</span>`;
}

function formatMetadataLine(detail) {
  const metadata = detail.metadata || {};
  const parts = [
    metadata.episode_idx !== undefined ? `Episode ${metadata.episode_idx}` : null,
    metadata.current_frame_idx !== undefined ? `Current frame ${metadata.current_frame_idx}` : null,
    metadata.stage !== undefined ? `Stage ${metadata.stage}` : null,
    metadata.subtask_id !== undefined ? `Subtask ${metadata.subtask_id}` : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

function buildImageLabel(detail, imageIdx) {
  const metadata = detail.metadata || {};
  const frameIndices = metadata.prompt_frame_indices || metadata.frame_indices || [];
  const frameIdx = frameIndices[imageIdx];
  const labelChips = [];
  if (imageIdx === detail.images.length - 1) {
    labelChips.push(chip("Current", "current"));
  } else {
    labelChips.push(chip(`Memory ${imageIdx + 1}`));
  }
  if (metadata.evidence_prompt_index && Number(metadata.evidence_prompt_index) === imageIdx + 1) {
    labelChips.push(chip("Evidence", "evidence"));
  }
  return {
    title: `Image ${imageIdx + 1}${frameIdx !== undefined ? ` · frame ${frameIdx}` : ""}`,
    chips: labelChips.join(""),
  };
}

function renderImageStage(detail) {
  const images = detail.images || [];
  if (!images.length) {
    elements.imageStage.innerHTML = "当前样本没有图片";
    elements.thumbnailRail.innerHTML = "";
    return;
  }
  if (state.selectedImageIdx === null || state.selectedImageIdx >= images.length) {
    state.selectedImageIdx = images.length - 1;
  }
  const activeImage = images[state.selectedImageIdx];
  const label = buildImageLabel(detail, state.selectedImageIdx);
  elements.imagePanelHint.textContent = `${images.length} 张图，点击缩略图切换，点击大图放大`;
  elements.imageStage.innerHTML = `
    <div class="image-stage-label">${label.chips}</div>
    <img src="${activeImage}" alt="${label.title}" />
  `;
  elements.imageStage.querySelector("img").addEventListener("click", () => {
    elements.lightboxImage.src = activeImage;
    elements.lightboxCaption.textContent = label.title;
    elements.lightbox.classList.remove("hidden");
    elements.lightbox.setAttribute("aria-hidden", "false");
  });

  elements.thumbnailRail.innerHTML = images
    .map((imageUrl, index) => {
      const itemLabel = buildImageLabel(detail, index);
      const active = index === state.selectedImageIdx ? "active" : "";
      return `
        <button class="thumbnail ${active}" data-image-idx="${index}" type="button">
          <img src="${imageUrl}" alt="${itemLabel.title}" />
          <div class="thumbnail-meta">
            <div>${itemLabel.title}</div>
          </div>
        </button>
      `;
    })
    .join("");
  for (const button of elements.thumbnailRail.querySelectorAll(".thumbnail")) {
    button.addEventListener("click", () => {
      state.selectedImageIdx = Number(button.dataset.imageIdx);
      renderImageStage(detail);
    });
  }
}

function renderParsedResponse(detail) {
  const parsed = detail.parsed || {};
  const order = ["think", "answer", "info", "frame", "camera", "action"];
  const blocks = order
    .filter((key) => parsed[key])
    .map(
      (key) => `
        <div class="parsed-block">
          <h3>${key}</h3>
          <pre>${parsed[key]}</pre>
        </div>
      `
    );
  if (!blocks.length) {
    elements.parsedResponse.innerHTML = '<div class="empty-state">当前样本没有可解析标签</div>';
    return;
  }
  if (detail.action_stats && detail.action_stats.rows > 0) {
    blocks.push(`
      <div class="parsed-block">
        <h3>action stats</h3>
        <pre>${detail.action_stats.rows} rows × ${detail.action_stats.dims} dims</pre>
      </div>
    `);
  }
  elements.parsedResponse.innerHTML = blocks.join("");
}

function jumpVideo(video, metadata, frameIdx) {
  const fps = Number(metadata?.fps || 30);
  if (!video || !Number.isFinite(frameIdx)) return;
  const jump = () => {
    video.currentTime = Math.max(Number(frameIdx) / Math.max(fps, 1), 0);
  };
  if (video.readyState >= 1) {
    jump();
  } else {
    video.addEventListener("loadedmetadata", jump, { once: true });
  }
}

function renderVideoGrid(detail) {
  const episodeAssets = detail.episode_assets || {};
  const cards = [];
  const metadata = detail.metadata || {};
  const currentFrame = metadata.current_frame_idx;
  const videoEntries = [
    { key: "annotated", label: "Annotated Video" },
    { key: "qa_overlay", label: "QA Overlay Video" },
    { key: "main", label: "Raw Main Video" },
  ];

  for (const entry of videoEntries) {
    const asset = episodeAssets[entry.key];
    if (!asset || !asset.exists || !asset.url) continue;
    cards.push(`
      <div class="video-card" data-video-key="${entry.key}">
        <strong>${entry.label}</strong>
        <div class="muted small">fps ${Number(asset.fps || 30).toFixed(2)}${asset.frame_count ? ` · ${asset.frame_count} frames` : ""}</div>
        <video controls preload="metadata" src="${asset.url}"></video>
        <div class="video-actions">
          <button class="ghost-button jump-frame" type="button">跳到当前帧</button>
        </div>
      </div>
    `);
  }

  if (!cards.length) {
    elements.videoGrid.innerHTML = '<div class="empty-state">当前样本暂无视频可预览</div>';
    return;
  }

  elements.videoGrid.innerHTML = cards.join("");
  for (const card of elements.videoGrid.querySelectorAll(".video-card")) {
    const videoKey = card.dataset.videoKey;
    const video = card.querySelector("video");
    const button = card.querySelector(".jump-frame");
    const asset = episodeAssets[videoKey];
    button.addEventListener("click", () => jumpVideo(video, asset, currentFrame));
  }
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value) || typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

function renderMetadata(detail) {
  const metadata = detail.metadata || {};
  const fields = [
    ["task_type", detail.task_type],
    ["episode_idx", metadata.episode_idx],
    ["current_frame_idx", metadata.current_frame_idx],
    ["stage", metadata.stage],
    ["subtask_id", metadata.subtask_id],
    ["roles", metadata.roles],
    ["prompt_frame_indices", metadata.prompt_frame_indices || metadata.frame_indices],
    ["prompt_image_count", metadata.prompt_image_count],
    ["evidence_frame_idx", metadata.evidence_frame_idx],
    ["evidence_prompt_index", metadata.evidence_prompt_index],
    ["evidence_from_history", metadata.evidence_from_history],
    ["camera_delta_deg", metadata.camera_delta_deg],
  ];
  elements.metadataGrid.innerHTML = fields
    .map(
      ([label, value]) => `
        <div class="metadata-card">
          <strong>${label}</strong>
          <pre class="metadata-value">${formatValue(value)}</pre>
        </div>
      `
    )
    .join("");
  elements.rawSampleJson.textContent = JSON.stringify(detail.raw_json || {}, null, 2);
}

function renderDetail(detail) {
  state.selectedImageIdx = detail.images.length ? detail.images.length - 1 : null;
  elements.detailTitle.textContent = `#${detail.sample_idx} · ${detail.task_type}`;
  elements.detailMetaLine.textContent = formatMetadataLine(detail);
  const metadata = detail.metadata || {};
  const chips = [
    chip(`${detail.images.length} images`),
    metadata.prompt_image_count ? chip(`prompt ${metadata.prompt_image_count}`) : "",
    metadata.stage !== undefined ? chip(`stage ${metadata.stage}`) : "",
    metadata.subtask_id !== undefined ? chip(`subtask ${metadata.subtask_id}`) : "",
    metadata.evidence_from_history ? chip("history evidence", "history") : "",
    detail.action_stats?.rows ? chip(`action ${detail.action_stats.rows}×${detail.action_stats.dims}`) : "",
  ]
    .filter(Boolean)
    .join("");
  elements.detailStatChips.innerHTML = chips;
  renderImageStage(detail);
  renderParsedResponse(detail);
  renderVideoGrid(detail);
  elements.userMessage.textContent = detail.messages?.[0]?.content || "";
  elements.assistantMessage.textContent = detail.messages?.[1]?.content || "";
  renderMetadata(detail);

  elements.copySampleJsonButton.onclick = async () => {
    await navigator.clipboard.writeText(JSON.stringify(detail.raw_json || {}, null, 2));
    elements.copySampleJsonButton.textContent = "已复制";
    window.setTimeout(() => {
      elements.copySampleJsonButton.textContent = "复制 JSON";
    }, 1000);
  };
  elements.copyPromptButton.onclick = async () => {
    await navigator.clipboard.writeText(detail.messages?.[0]?.content || "");
    elements.copyPromptButton.textContent = "已复制";
    window.setTimeout(() => {
      elements.copyPromptButton.textContent = "复制 Prompt";
    }, 1000);
  };
}

function navigateSample(direction) {
  const cacheKey = sampleListCacheKey();
  if (!cacheKey) return;
  const payload = caches.sampleLists.get(cacheKey);
  if (!payload) return;
  const filtered = currentFilteredSamples(payload);
  if (!filtered.length) return;
  const currentIndex = filtered.findIndex((sample) => sample.sample_idx === state.selectedSampleIdx);
  const baseIndex = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = Math.min(Math.max(baseIndex + direction, 0), filtered.length - 1);
  state.selectedSampleIdx = filtered[nextIndex].sample_idx;
  syncUrlState();
  renderSampleList(payload);
}

function attachEvents() {
  elements.refreshIndexButton.addEventListener("click", async () => {
    caches.sampleLists.clear();
    caches.sampleDetails.clear();
    await loadIndex(true);
    await loadCurrentSampleList();
  });

  elements.taskSearchInput.addEventListener("input", () => {
    state.taskSearch = elements.taskSearchInput.value;
    syncUrlState();
    renderTaskList();
  });

  elements.storageSelect.addEventListener("change", () => {
    state.storageFilter = elements.storageSelect.value;
    syncUrlState();
    renderTaskList();
  });

  elements.sampleSearchInput.addEventListener("input", () => {
    state.sampleSearch = elements.sampleSearchInput.value;
    syncUrlState();
    const payload = caches.sampleLists.get(sampleListCacheKey());
    if (payload) renderSampleList(payload);
  });

  for (const [element, stateKey] of [
    [elements.episodeFilter, "sampleEpisode"],
    [elements.stageFilter, "sampleStage"],
    [elements.subtaskFilter, "sampleSubtask"],
  ]) {
    element.addEventListener("change", () => {
      state[stateKey] = element.value;
      syncUrlState();
      const payload = caches.sampleLists.get(sampleListCacheKey());
      if (payload) renderSampleList(payload);
    });
  }

  elements.historyOnlyCheckbox.addEventListener("change", () => {
    state.historyOnly = elements.historyOnlyCheckbox.checked;
    syncUrlState();
    const payload = caches.sampleLists.get(sampleListCacheKey());
    if (payload) renderSampleList(payload);
  });

  elements.prevSampleButton.addEventListener("click", () => navigateSample(-1));
  elements.nextSampleButton.addEventListener("click", () => navigateSample(1));

  document.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {
      return;
    }
    if (event.key === "j") {
      navigateSample(1);
    } else if (event.key === "k") {
      navigateSample(-1);
    } else if (event.key === "Escape") {
      elements.lightbox.classList.add("hidden");
      elements.lightbox.setAttribute("aria-hidden", "true");
    } else if (event.key === "/") {
      event.preventDefault();
      elements.sampleSearchInput.focus();
    }
  });

  elements.lightboxClose.addEventListener("click", () => {
    elements.lightbox.classList.add("hidden");
    elements.lightbox.setAttribute("aria-hidden", "true");
  });
  elements.lightbox.addEventListener("click", (event) => {
    if (event.target === elements.lightbox) {
      elements.lightbox.classList.add("hidden");
      elements.lightbox.setAttribute("aria-hidden", "true");
    }
  });
}

async function bootstrap() {
  parseUrlState();
  elements.taskSearchInput.value = state.taskSearch;
  elements.sampleSearchInput.value = state.sampleSearch;
  elements.historyOnlyCheckbox.checked = state.historyOnly;
  attachEvents();
  try {
    await loadIndex(false);
    await loadCurrentSampleList();
  } catch (error) {
    console.error(error);
    setStatus("Error", "error");
    elements.taskList.innerHTML = `<div class="empty-state">${String(error)}</div>`;
  }
}

bootstrap();
