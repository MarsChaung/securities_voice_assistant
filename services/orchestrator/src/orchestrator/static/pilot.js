const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const characterCount = document.querySelector("#character-count");
const clearButton = document.querySelector("#clear-chat");
const systemState = document.querySelector("#system-state");
const systemStateText = document.querySelector("#system-state-text");
const voiceButton = document.querySelector("#voice-button");
const voiceStatus = document.querySelector("#voice-status");
const asrModelControl = document.querySelector("#asr-model-control");
const asrModelSelect = document.querySelector("#asr-model");
const replyModeSelect = document.querySelector("#reply-mode");
const bargeInControl = document.querySelector("#barge-in-control");
const bargeInModeSelect = document.querySelector("#barge-in-mode");
const initialMessage = messages.firstElementChild.cloneNode(true);
const voiceTestMode = document.documentElement.dataset.voiceTest === "true";
const hangupButton = document.querySelector("#hangup-button");
const greetingInput = document.querySelector("#greeting");
const callState = document.querySelector("#call-state");
const callStateText = document.querySelector("#call-state-text");
const callDuration = document.querySelector("#call-duration");
const runtimeStatus = document.querySelector("#runtime-status");
const voiceServiceStatus = document.querySelector("#voice-service-status");
const asrStatus = document.querySelector("#asr-status");
const ttsStatus = document.querySelector("#tts-status");
const knowledgeCount = document.querySelector("#knowledge-count");
const phaseStatus = document.querySelector("#phase-status");
const asrLiveText = document.querySelector("#asr-live-text");
const sessionIdOutput = document.querySelector("#session-id");
const copySessionIdButton = document.querySelector("#copy-session-id");
const diagnosticLogNote = document.querySelector("#diagnostic-log-note");
const conversationPanel = document.querySelector(".conversation-panel");
const controlColumn = document.querySelector(".control-column");
const desktopVoiceTestLayout = window.matchMedia("(min-width: 981px)");
const VOICE_TEST_IDLE_TIMEOUT_MS = 8000;
const voiceState = {
  config: null,
  stream: null,
  audioContext: null,
  microphoneSource: null,
  captureNode: null,
  captureWorkletReady: false,
  socket: null,
  socketPromise: null,
  socketReady: false,
  activeAsrModel: null,
  listening: false,
  continuous: false,
  replying: false,
  replyController: null,
  activeSources: new Set(),
  playbackScheduler: null,
  acknowledgementAudio: new Map(),
  lastAcknowledgementAudioUrl: null,
  partialTranscript: "",
  replyDisplayed: false,
  bargeInDetector: null,
  bargeInStreaming: false,
  pendingBargeInMetrics: null,
  preRollFrames: [],
  preRollSamples: 0,
  endpointTimer: null,
  pendingEndpointTranscript: "",
  idleTimer: null,
  idlePromptStage: 0,
  subtitleNodes: [],
  callStartedAt: null,
  callTimer: null,
  layoutObserver: null,
  conversationId: null,
};

function selectedReplyMode() {
  return replyModeSelect?.value || "exact";
}

function selectedAsrModel() {
  return asrModelSelect.value || voiceState.config?.models?.asr || "";
}

function asrLanguage(model) {
  return model.includes("Qwen3-ASR") ? "Chinese" : "zh";
}

function asrUsesTokenStreaming(model) {
  return model.includes("Qwen3-ASR");
}

function asrModelLabel(model) {
  return model.split("/").pop() || model;
}

function selectedBargeInPreset() {
  const presets = voiceState.config?.barge_in?.presets || [];
  return presets.find((preset) => preset.id === bargeInModeSelect.value) || presets[0];
}

function asrEndpointGraceMs() {
  return Math.max(0, Number(voiceState.config?.asr_endpoint_grace_ms) || 0);
}

function clearPendingAsrEndpoint() {
  if (voiceState.endpointTimer !== null) clearTimeout(voiceState.endpointTimer);
  voiceState.endpointTimer = null;
  voiceState.pendingEndpointTranscript = "";
}

function mergeAsrTranscripts(first, second) {
  const leading = VoiceBargeIn.sanitizeAsrTranscript(first).trim();
  const trailing = VoiceBargeIn.sanitizeAsrTranscript(second).trim();
  if (!leading) return trailing;
  if (!trailing || leading.endsWith(trailing)) return leading;
  if (trailing.startsWith(leading)) return trailing;
  return `${leading} ${trailing}`;
}

function resumePendingAsrEndpoint(nextText = "") {
  if (voiceState.endpointTimer === null) return false;
  const pending = voiceState.pendingEndpointTranscript;
  clearPendingAsrEndpoint();
  voiceState.partialTranscript = mergeAsrTranscripts(pending, nextText);
  setVoiceStatus("偵測到你繼續說話，已取消送出並持續聆聽…");
  return true;
}

function clearVoiceIdleTimer() {
  if (voiceState.idleTimer !== null) clearTimeout(voiceState.idleTimer);
  voiceState.idleTimer = null;
}

function ensureVoiceTestSession() {
  if (!voiceTestMode) return null;
  if (!voiceState.conversationId) voiceState.conversationId = crypto.randomUUID();
  if (sessionIdOutput) sessionIdOutput.textContent = voiceState.conversationId;
  return voiceState.conversationId;
}

function clearVoiceConversation({ rotate = false } = {}) {
  const conversationId = voiceState.conversationId;
  if (conversationId) {
    void fetch(`/v1/voice/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {});
  }
  if (rotate) {
    voiceState.conversationId = null;
    ensureVoiceTestSession();
  }
}

function scheduleVoiceIdleTimer() {
  clearVoiceIdleTimer();
  if (
    !voiceTestMode
    || !voiceState.continuous
    || !voiceState.listening
    || voiceState.replying
  ) {
    return;
  }
  voiceState.idleTimer = setTimeout(() => {
    voiceState.idleTimer = null;
    void handleVoiceIdleTimeout();
  }, VOICE_TEST_IDLE_TIMEOUT_MS);
}

function closeRealtimeAsr() {
  clearPendingAsrEndpoint();
  const socket = voiceState.socket;
  voiceState.socket = null;
  voiceState.socketReady = false;
  voiceState.activeAsrModel = null;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: "stop" }));
  }
  socket?.close();
}

const decisionLabels = {
  answer: "核准知識回答",
  clarify: "請確認語音內容",
  refuse: "安全拒答",
  handoff: "轉人工處理",
};

function decisionLabel(result) {
  if (result.policy_rule_id === "SYS-INTENT-ROUTER-ERROR") return "意圖路由失敗";
  if (
    result.policy_rule_id === "POL-DEFAULT-DENY" ||
    result.policy_rule_id === "LLM-DEFAULT-DENY"
  ) {
    return "未涵蓋的問題";
  }
  return decisionLabels[result.decision] || "系統回覆";
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
  requestAnimationFrame(() => {
    messages.scrollTop = messages.scrollHeight;
  });
}

function syncVoiceTestPanelHeight() {
  if (!voiceTestMode || !conversationPanel || !controlColumn) return;
  if (!desktopVoiceTestLayout.matches) {
    conversationPanel.style.height = "";
    return;
  }
  conversationPanel.style.height = `${controlColumn.getBoundingClientRect().height}px`;
}

function initializeVoiceTestLayout() {
  if (!voiceTestMode || !conversationPanel || !controlColumn) return;
  voiceState.layoutObserver = new ResizeObserver(syncVoiceTestPanelHeight);
  voiceState.layoutObserver.observe(controlColumn);
  desktopVoiceTestLayout.addEventListener("change", syncVoiceTestPanelHeight);
  syncVoiceTestPanelHeight();
}

function removeConversationEmptyState() {
  messages.querySelector(".conversation-empty")?.remove();
}

function appendUserMessage(text) {
  removeConversationEmptyState();
  const article = element("article", "message user-message");
  const content = element("div", "message-content");
  content.append(element("p", "message-label", "你"));
  const copy = element("div", "answer-copy");
  copy.append(element("p", "", text));
  content.append(copy);
  article.append(content);
  messages.append(article);
  scrollToLatest();
}

function appendLoading(id = "loading-message") {
  const article = element("article", "message assistant-message");
  article.id = id;
  article.append(element("div", "avatar", "知"));
  const content = element("div", "message-content");
  content.append(element("p", "message-label", "知識助手正在查核來源"));
  const loading = element("div", "answer-copy loading-row");
  loading.append(element("span"), element("span"), element("span"));
  content.append(loading);
  article.append(content);
  messages.append(article);
  scrollToLatest();
}

function safeSourceReference(citation) {
  if (citation.source_uri) {
    let parsed;
    try {
      parsed = new URL(citation.source_uri);
    } catch {
      return null;
    }
    if (parsed.protocol !== "https:") return null;

    const link = element("a", "source-link");
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.append(document.createTextNode(`${citation.source_title} ↗`));
    link.append(element("span", "source-locator", citation.source_locator));
    return link;
  }

  const reference = element("div", "source-link source-reference");
  reference.append(document.createTextNode(citation.source_title));
  reference.append(element("span", "source-locator", citation.source_locator));
  return reference;
}

function feedbackPanel(turnId) {
  const panel = element("div", "feedback");
  panel.append(element("span", "", "這個結果有幫助嗎？"));
  const helpful = element("button", "", "有幫助");
  const notHelpful = element("button", "", "沒幫助");
  helpful.type = "button";
  notHelpful.type = "button";

  async function sendFeedback(rating, selected) {
    helpful.disabled = true;
    notHelpful.disabled = true;
    try {
      const response = await fetch(`/v1/turns/${encodeURIComponent(turnId)}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating }),
      });
      if (!response.ok) throw new Error("feedback request failed");
      panel.replaceChildren(element("span", "", selected));
    } catch {
      helpful.disabled = false;
      notHelpful.disabled = false;
      panel.prepend(element("span", "", "回饋未送出，請稍後再試。"));
    }
  }

  helpful.addEventListener("click", () => sendFeedback("helpful", "謝謝你的回饋。"));
  notHelpful.addEventListener("click", () =>
    sendFeedback("not_helpful", "已記錄，謝謝你的回饋。"),
  );
  panel.append(helpful, notHelpful);
  return panel;
}

function appendAssistantMessage(
  turn,
  allowFeedback = true,
  { progressiveSegments = [], label = "知識助手", showDecision = true } = {},
) {
  removeConversationEmptyState();
  const result = turn.result;
  const article = element("article", `message assistant-message ${result.decision}`);
  article.append(element("div", "avatar", "知"));
  const content = element("div", "message-content");
  content.append(element("p", "message-label", label));
  if (showDecision) {
    content.append(
      element(
        "div",
        `decision-badge decision-${result.decision}`,
        decisionLabel(result),
      ),
    );
  }
  const copy = element("div", "answer-copy");
  const answer = element("p");
  if (progressiveSegments.length) {
    voiceState.subtitleNodes = progressiveSegments.map((segment) => {
      const node = element("span", "subtitle-segment", segment);
      answer.append(node);
      return node;
    });
  } else {
    answer.textContent = result.answer;
    voiceState.subtitleNodes = [];
  }
  copy.append(answer);
  content.append(copy);

  const references = (result.citations || []).map(safeSourceReference).filter(Boolean);
  if (references.length) {
    const sources = element("div", "sources");
    sources.append(element("strong", "", "資料來源"), ...references);
    content.append(sources);
  }

  if (showDecision) {
    const details = element("details", "decision-details");
    details.append(element("summary", "", "查看決策資訊"));
    details.append(
      element(
        "p",
        "",
        `規則 ${result.policy_rule_id} · 意圖 ${result.intent} · 信心 ${Math.round(result.confidence * 100)}%`,
      ),
    );
    content.append(details);
  }
  if (allowFeedback) content.append(feedbackPanel(turn.turn_id));
  article.append(content);
  messages.append(article);
  scrollToLatest();
  return article;
}

function revealVoiceSubtitle(sentenceIndex, nodes = voiceState.subtitleNodes) {
  const node = nodes[sentenceIndex - 1];
  if (!node || node.classList.contains("visible")) return;
  node.classList.add("visible");
  scrollToLatest();
}

function revealAllVoiceSubtitles(nodes = voiceState.subtitleNodes) {
  for (const node of nodes) node.classList.add("visible");
}

function appendNetworkError() {
  appendAssistantMessage(
    {
      turn_id: "network-error",
      result: {
        decision: "refuse",
        answer: "服務目前無法連線，請稍後再試。若問題持續發生，請改用官方客服管道。",
        citations: [],
        policy_rule_id: "SYS-001",
        intent: "service_unavailable",
        confidence: 1,
      },
    },
    false,
  );
}

async function loadSystemState() {
  try {
    const response = await fetch("/healthz", { headers: { Accept: "application/json" } });
    const health = await response.json();
    const count = health.eligible_knowledge_count || 0;
    if (!response.ok || health.knowledge_database === "unavailable") {
      systemState.className = "system-state error";
      systemStateText.textContent = "知識服務無法連線";
      setStatusValue(runtimeStatus, "error", "Runtime 無法連線");
    } else if (count === 0) {
      systemState.className = "system-state warning";
      systemStateText.textContent = "0 筆可回答知識｜請檢查生效與複審到期時間";
      setStatusValue(runtimeStatus, "warning", "Runtime 已連線");
    } else {
      systemState.className = "system-state ready";
      systemStateText.textContent = `${count} 筆知識可回答`;
      setStatusValue(runtimeStatus, "ready", "Runtime 正常");
    }
    if (knowledgeCount) knowledgeCount.textContent = `${count} 筆`;
  } catch {
    systemState.className = "system-state error";
    systemStateText.textContent = "無法取得系統狀態";
    setStatusValue(runtimeStatus, "error", "無法取得狀態");
  }
}

function setStatusValue(target, state, text) {
  if (!target) return;
  const dot = element("span", `status-dot ${state}`);
  target.replaceChildren(dot, document.createTextNode(text));
}

function setVoiceStatus(text) {
  voiceStatus.textContent = text;
  if (phaseStatus) phaseStatus.textContent = text;
}

function setAsrLive(text) {
  if (asrLiveText) asrLiveText.textContent = text;
}

function formatCallDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function startCallClock() {
  if (!voiceTestMode || voiceState.callTimer) return;
  voiceState.callStartedAt = Date.now();
  callDuration.textContent = "00:00";
  voiceState.callTimer = setInterval(() => {
    callDuration.textContent = formatCallDuration(Date.now() - voiceState.callStartedAt);
  }, 1000);
}

function stopCallClock() {
  if (voiceState.callTimer) clearInterval(voiceState.callTimer);
  voiceState.callTimer = null;
}

function updateCallControls(state) {
  if (!voiceTestMode) return;
  const active = state !== "idle";
  voiceButton.disabled = active;
  hangupButton.disabled = !active;
  greetingInput.disabled = active;
  callState.dataset.state = state;
  callStateText.textContent = {
    idle: "通話已結束",
    connecting: "撥號中",
    active: "通話中",
    speaking: "AI 客服說話中",
  }[state] || "通話中";
  if (!active) stopCallClock();
}

function downsampleToInt16(inputSamples, sourceRate, targetRate = 16000) {
  const ratio = sourceRate / targetRate;
  const output = new Int16Array(Math.floor(inputSamples.length / ratio));
  for (let index = 0; index < output.length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.max(start + 1, Math.floor((index + 1) * ratio));
    let total = 0;
    for (let sourceIndex = start; sourceIndex < end && sourceIndex < inputSamples.length; sourceIndex += 1) {
      total += inputSamples[sourceIndex];
    }
    const sample = Math.max(-1, Math.min(1, total / (end - start)));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

async function ensureMicrophone() {
  if (!voiceState.stream) {
    voiceState.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
    });
  }
  if (!voiceState.audioContext) {
    voiceState.audioContext = new AudioContext();
    voiceState.microphoneSource = voiceState.audioContext.createMediaStreamSource(
      voiceState.stream,
    );
  }
  if (!voiceState.captureWorkletReady) {
    if (!voiceState.audioContext.audioWorklet) {
      throw new Error("目前瀏覽器不支援低延遲 AudioWorklet。");
    }
    const assetVersion = document.documentElement.dataset.assetVersion || "dev";
    await voiceState.audioContext.audioWorklet.addModule(
      `/pilot/static/voice-capture-worklet.js?v=${encodeURIComponent(assetVersion)}`,
    );
    voiceState.captureWorkletReady = true;
  }
  await voiceState.audioContext.resume();
}

function stopRealtimeCapture() {
  if (voiceState.captureNode) {
    try {
      voiceState.microphoneSource.disconnect(voiceState.captureNode);
    } catch {}
    voiceState.captureNode.port.onmessage = null;
    voiceState.captureNode.disconnect();
    voiceState.captureNode = null;
  }
}

function stopVoicePlayback(reason = "manual", bargeInMetrics = null) {
  if (voiceState.playbackScheduler) {
    voiceState.playbackScheduler.interrupt(reason, bargeInMetrics);
    voiceState.playbackScheduler = null;
    return;
  }
  for (const source of voiceState.activeSources) {
    try {
      source.stop();
    } catch {}
  }
  voiceState.activeSources.clear();
}

function resetPreRoll() {
  voiceState.preRollFrames = [];
  voiceState.preRollSamples = 0;
}

function rememberPreRoll(pcm) {
  const preset = selectedBargeInPreset();
  if (!preset) return;
  voiceState.preRollFrames.push(pcm);
  voiceState.preRollSamples += pcm.length;
  const maximumSamples = Math.ceil((preset.pre_roll_ms / 1000) * 16000);
  while (
    voiceState.preRollFrames.length > 1
    && voiceState.preRollSamples > maximumSamples
  ) {
    voiceState.preRollSamples -= voiceState.preRollFrames.shift().length;
  }
}

function sendAsrFrame(pcm) {
  if (pcm.length && voiceState.socket?.readyState === WebSocket.OPEN) {
    voiceState.socket.send(pcm.buffer);
  }
}

function flushPreRoll() {
  for (const frame of voiceState.preRollFrames) sendAsrFrame(frame);
  resetPreRoll();
}

function cancelBargeInCandidate() {
  if (voiceState.bargeInStreaming && voiceState.socket?.readyState === WebSocket.OPEN) {
    voiceState.socket.send(JSON.stringify({ action: "cancel" }));
  }
  voiceState.bargeInStreaming = false;
  resetPreRoll();
}

function configureBargeInDetector() {
  const preset = selectedBargeInPreset();
  if (!preset || !voiceState.config?.barge_in?.enabled) {
    voiceState.bargeInDetector = null;
    return;
  }
  voiceState.bargeInDetector = new VoiceBargeIn.BargeInDetector({
    preset,
    onDuck: () => {
      if (!voiceState.replying || !voiceState.playbackScheduler?.started) return;
      const activePreset = selectedBargeInPreset();
      voiceState.playbackScheduler.duck(
        activePreset.duck_volume,
        activePreset.fade_out_ms,
      );
      voiceState.bargeInStreaming = true;
      flushPreRoll();
      setVoiceStatus("偵測到你可能正在插話，正在確認…");
    },
    onRestore: () => {
      const activePreset = selectedBargeInPreset();
      voiceState.playbackScheduler?.restore(activePreset.fade_out_ms);
      voiceState.pendingBargeInMetrics = null;
      cancelBargeInCandidate();
      setVoiceStatus("已忽略短促聲音，繼續播放語音回答…");
    },
    onConfirm: (metrics) => {
      if (!voiceState.replying || !voiceState.playbackScheduler) return;
      voiceState.pendingBargeInMetrics = metrics;
      voiceState.listening = true;
      voiceState.bargeInStreaming = true;
      voiceButton.classList.remove("speaking");
      voiceButton.classList.add("listening");
      setVoiceStatus("已降低回答音量，正在確認插話內容…");
    },
  });
}

function armBargeIn() {
  if (!voiceState.bargeInDetector) return;
  voiceState.bargeInDetector.setPreset(selectedBargeInPreset());
  voiceState.bargeInStreaming = false;
  voiceState.pendingBargeInMetrics = null;
  resetPreRoll();
  voiceState.bargeInDetector.arm();
}

function disarmBargeIn({ restore = false } = {}) {
  voiceState.bargeInDetector?.disarm({ restore });
  voiceState.bargeInStreaming = false;
  voiceState.pendingBargeInMetrics = null;
  resetPreRoll();
}

function reportVoicePlayback(turnId, metrics) {
  if (!turnId || !metrics.chunk_count) return;
  void fetch(`/v1/voice/${encodeURIComponent(turnId)}/playback-metrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metrics),
    keepalive: true,
  }).catch(() => {});
}

function startRealtimeCapture() {
  stopRealtimeCapture();
  const captureNode = new AudioWorkletNode(
    voiceState.audioContext,
    "voice-capture-processor",
    {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
      channelCountMode: "explicit",
    },
  );
  captureNode.port.onmessage = (event) => {
    if (
      event.data?.type !== "audio-frame"
      || voiceState.socket?.readyState !== WebSocket.OPEN
    ) return;
    const pcm = downsampleToInt16(
      event.data.samples,
      voiceState.audioContext.sampleRate,
    );
    if (!pcm.length) return;

    if (voiceState.listening) {
      sendAsrFrame(pcm);
      return;
    }
    if (
      !voiceState.replying
      || !voiceState.playbackScheduler?.started
      || !voiceState.bargeInDetector
    ) return;

    rememberPreRoll(pcm);
    const wasStreaming = voiceState.bargeInStreaming;
    voiceState.bargeInDetector.processLevel(event.data.dbfs);
    if (wasStreaming && voiceState.bargeInStreaming) sendAsrFrame(pcm);
  };
  voiceState.microphoneSource.connect(captureNode);
  voiceState.captureNode = captureNode;
}

function restorePlaybackAfterRejectedBargeIn(status) {
  const scheduler = voiceState.playbackScheduler;
  if (!voiceState.replying || !scheduler || !voiceState.pendingBargeInMetrics) {
    return false;
  }
  const activePreset = selectedBargeInPreset();
  scheduler.recordBargeInMetrics?.(voiceState.pendingBargeInMetrics);
  scheduler.restore(activePreset.fade_out_ms);
  voiceState.pendingBargeInMetrics = null;
  voiceState.listening = false;
  cancelBargeInCandidate();
  armBargeIn();
  voiceButton.classList.remove("listening");
  voiceButton.classList.add("speaking");
  updateCallControls("speaking");
  setVoiceStatus(status);
  startRealtimeCapture();
  return true;
}

function commitPendingBargeIn() {
  if (!voiceState.pendingBargeInMetrics || !voiceState.playbackScheduler) return;
  voiceState.playbackScheduler.interrupt(
    "barge_in",
    voiceState.pendingBargeInMetrics,
  );
  voiceState.playbackScheduler = null;
  voiceState.pendingBargeInMetrics = null;
  voiceState.bargeInStreaming = false;
  voiceState.replyController?.abort();
}

function continueAfterIgnoredTranscript(status) {
  voiceState.partialTranscript = "";
  if (
    restorePlaybackAfterRejectedBargeIn(
      "已忽略短促插話，繼續播放原本的語音回答…",
    )
  ) {
    setAsrLive(status);
    return;
  }
  setVoiceStatus(status);
  if (voiceState.continuous) {
    void startVoiceListening({ preserveIdleStage: true });
  }
}

function finalizeRealtimeTranscript(transcript) {
  clearVoiceIdleTimer();
  voiceState.listening = false;
  stopRealtimeCapture();
  voiceButton.classList.remove("listening");
  setAsrLive(transcript || "沒有辨識到有效語音。");

  const invalidTranscript =
    !VoiceBargeIn.hasMeaningfulTranscript(transcript)
    || VoiceBargeIn.isLikelyContextEcho(
      transcript,
      voiceState.config?.asr_context || "",
    );
  if (invalidTranscript) {
    continueAfterIgnoredTranscript("已忽略無效辨識或環境雜音，持續聆聽中…");
    return;
  }
  if (
    voiceState.pendingBargeInMetrics
    && VoiceBargeIn.shouldResumePlaybackAfterBargeIn(transcript)
  ) {
    continueAfterIgnoredTranscript(`已忽略短插話「${transcript}」。`);
    return;
  }
  if (VoiceBargeIn.isNonActionableUtterance(transcript)) {
    continueAfterIgnoredTranscript("聽到了，請繼續說你的問題…");
    return;
  }

  if (voiceState.pendingBargeInMetrics) commitPendingBargeIn();
  voiceState.idlePromptStage = 0;
  void submitVoiceTurn(transcript);
}

function scheduleRealtimeTranscript(transcript) {
  clearPendingAsrEndpoint();
  voiceState.pendingEndpointTranscript = transcript;
  const delayMs = asrEndpointGraceMs();
  setVoiceStatus(
    delayMs
      ? `偵測到停頓，將再等待 ${(delayMs / 1000).toFixed(1)} 秒…`
      : "正在送出語音辨識結果…",
  );
  const finalize = () => {
    const pending = voiceState.pendingEndpointTranscript;
    voiceState.endpointTimer = null;
    voiceState.pendingEndpointTranscript = "";
    finalizeRealtimeTranscript(pending);
  };
  if (!delayMs) {
    finalize();
    return;
  }
  voiceState.endpointTimer = setTimeout(finalize, delayMs);
}

function handleRealtimeAsrEvent(data) {
  if (data.error) {
    stopVoiceSession("即時語音辨識發生錯誤，請稍後再試。");
    return;
  }
  if (data.type === "delta") {
    const delta = VoiceBargeIn.sanitizeAsrTranscript(data.delta);
    if (!resumePendingAsrEndpoint(delta)) {
      voiceState.partialTranscript += delta;
    }
    const partial = voiceState.partialTranscript.trim();
    if (partial) {
      clearVoiceIdleTimer();
      setVoiceStatus(`辨識中：${partial.slice(-36)}`);
      setAsrLive(partial);
    }
    return;
  }
  if (data.text && data.is_partial) {
    const partial = VoiceBargeIn.sanitizeAsrTranscript(data.text).trim();
    if (!resumePendingAsrEndpoint(partial)) {
      voiceState.partialTranscript = partial;
    }
    if (VoiceBargeIn.hasMeaningfulTranscript(voiceState.partialTranscript)) {
      clearVoiceIdleTimer();
    }
    setVoiceStatus(`辨識中：${voiceState.partialTranscript.slice(-36)}`);
    setAsrLive(voiceState.partialTranscript);
    return;
  }
  if (data.type === "candidate_end") {
    const candidate = VoiceBargeIn.sanitizeAsrTranscript(
      data.text || voiceState.partialTranscript,
    ).trim();
    if (!resumePendingAsrEndpoint(candidate)) {
      voiceState.partialTranscript = candidate;
    }
    if (VoiceBargeIn.hasMeaningfulTranscript(voiceState.partialTranscript)) {
      clearVoiceIdleTimer();
    }
    setVoiceStatus(
      data.semantic_complete ? "正在確認你是否說完…" : "句子似乎還沒說完，持續聆聽中…",
    );
    return;
  }
  if (data.type === "speech_start") {
    clearVoiceIdleTimer();
    resumePendingAsrEndpoint();
    if (voiceState.replying && voiceState.bargeInStreaming) {
      voiceState.bargeInDetector?.markSpeech();
    }
    return;
  }
  if (data.type === "speech_resumed") {
    clearVoiceIdleTimer();
    resumePendingAsrEndpoint();
    setVoiceStatus("偵測到你繼續說話，持續聆聽中…");
    return;
  }
  if (data.type === "noise_rejected") {
    if (voiceState.endpointTimer !== null) {
      setVoiceStatus("已忽略短暫環境音，繼續等待你是否說完…");
      return;
    }
    voiceState.partialTranscript = "";
    setVoiceStatus("已忽略短暫環境音，持續聆聽中…");
    setAsrLive("已忽略短暫環境音，等待有效語音…");
    scheduleVoiceIdleTimer();
    return;
  }
  if (data.type === "utterance_end" && voiceState.listening) {
    clearVoiceIdleTimer();
    const accumulated = mergeAsrTranscripts(
      voiceState.pendingEndpointTranscript,
      voiceState.partialTranscript,
    );
    const transcript = mergeAsrTranscripts(accumulated, data.text || "");
    voiceState.partialTranscript = transcript;
    scheduleRealtimeTranscript(transcript);
  }
}

async function ensureRealtimeAsr() {
  const requestedModel = selectedAsrModel();
  if (
    voiceState.socket?.readyState === WebSocket.OPEN
    && voiceState.socketReady
    && voiceState.activeAsrModel === requestedModel
  ) return;
  if (voiceState.socket?.readyState === WebSocket.OPEN) closeRealtimeAsr();
  if (voiceState.socketPromise) return voiceState.socketPromise;
  if (!voiceState.config?.realtime_asr_url || !requestedModel) {
    throw new Error("缺少即時 ASR 設定。");
  }

  voiceState.socketPromise = new Promise((resolve, reject) => {
    const socket = new WebSocket(voiceState.config.realtime_asr_url);
    voiceState.socket = socket;
    voiceState.socketReady = false;
    let settled = false;
    const timeout = setTimeout(() => {
      if (!settled) {
        voiceState.socketPromise = null;
        socket.close();
        reject(new Error("即時 ASR 模型載入逾時。"));
      }
    }, 120000);
    socket.onopen = () => socket.send(JSON.stringify({
      model: requestedModel,
      language: asrLanguage(requestedModel),
      sample_rate: 16000,
      streaming: asrUsesTokenStreaming(requestedModel),
      semantic_endpointing: true,
      output_script: "traditional",
      context: voiceState.config.asr_context || undefined,
    }));
    socket.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      if (data.status === "ready") {
        settled = true;
        clearTimeout(timeout);
        voiceState.socketReady = true;
        voiceState.activeAsrModel = requestedModel;
        voiceState.socketPromise = null;
        if (asrStatus) asrStatus.textContent = `${asrModelLabel(requestedModel)}｜已連線`;
        resolve();
        return;
      }
      handleRealtimeAsrEvent(data);
    };
    socket.onerror = () => {
      if (asrStatus) asrStatus.textContent = "連線失敗";
      if (!settled) {
        voiceState.socketPromise = null;
        reject(new Error("無法連線到即時 ASR。"));
      }
    };
    socket.onclose = () => {
      clearTimeout(timeout);
      voiceState.socketReady = false;
      voiceState.activeAsrModel = null;
      voiceState.socketPromise = null;
      if (asrStatus) asrStatus.textContent = "連線已關閉";
      if (!settled) reject(new Error("即時 ASR 連線已關閉。"));
      else if (voiceState.continuous) stopVoiceSession("即時 ASR 連線已中斷。");
    };
  });
  return voiceState.socketPromise;
}

async function startVoiceListening({ preserveIdleStage = false } = {}) {
  try {
    clearVoiceIdleTimer();
    clearPendingAsrEndpoint();
    asrModelSelect.disabled = true;
    bargeInModeSelect.disabled = true;
    if (replyModeSelect) replyModeSelect.disabled = true;
    await ensureMicrophone();
    setVoiceStatus("正在準備即時語音辨識…");
    await ensureRealtimeAsr();
    if (!voiceState.continuous) return;
    if (!preserveIdleStage) voiceState.idlePromptStage = 0;
    disarmBargeIn();
    voiceState.partialTranscript = "";
    voiceState.listening = true;
    voiceButton.classList.remove("speaking");
    voiceButton.classList.add("listening");
    voiceButton.setAttribute("aria-label", "停止即時語音對話");
    setVoiceStatus("正在聽…自然說完一句話即可");
    updateCallControls("active");
    startRealtimeCapture();
    scheduleVoiceIdleTimer();
  } catch (error) {
    stopVoiceSession(`無法啟動語音：${error.message}`);
  }
}

function stopVoiceSession(status = "語音對話已停止；按麥克風可重新開始") {
  clearVoiceIdleTimer();
  clearPendingAsrEndpoint();
  voiceState.idlePromptStage = 0;
  voiceState.continuous = false;
  voiceState.listening = false;
  voiceState.replying = false;
  voiceState.replyController?.abort();
  voiceState.replyController = null;
  disarmBargeIn();
  stopRealtimeCapture();
  stopVoicePlayback();
  if (voiceTestMode) {
    closeRealtimeAsr();
  } else if (voiceState.socket?.readyState === WebSocket.OPEN) {
    voiceState.socket.send(JSON.stringify({ action: "cancel" }));
  }
  voiceButton.classList.remove("listening", "speaking");
  voiceButton.setAttribute("aria-label", "開始即時語音對話");
  asrModelSelect.disabled = false;
  bargeInModeSelect.disabled = false;
  if (replyModeSelect) replyModeSelect.disabled = replyModeSelect.options.length < 2;
  clearVoiceConversation();
  updateCallControls("idle");
  if (asrStatus) asrStatus.textContent = "尚未連線";
  setVoiceStatus(status);
}

async function playVoiceResponse(response, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let scheduler = null;
  let pending = "";
  let turnId = null;
  let contentReceived = false;
  let audioReceived = false;
  let ttsFailed = false;
  let completed = false;
  let greetingResponse = false;
  let farewellResponse = false;
  let idlePromptResponse = false;
  let responseSubtitleNodes = [];

  const prepareScheduler = (characterCount) => {
    scheduler = new VoicePlayback.VoicePlaybackScheduler({
      audioContext: voiceState.audioContext,
      destination: voiceState.audioContext.destination,
      activeSources: voiceState.activeSources,
      initialBufferSeconds: VoicePlayback.recommendedInitialBufferSeconds(characterCount),
    });
    voiceState.playbackScheduler = scheduler;
  };

  const playAcknowledgement = async (audioUrl) => {
    if (!audioUrl) return;
    try {
      const cached = voiceState.acknowledgementAudio.get(audioUrl);
      const bytes = await (
        cached
          ? cached
          : fetch(audioUrl, { cache: "force-cache" }).then((audioResponse) => {
              if (!audioResponse.ok) throw new Error("無法載入等待提示音。");
              return audioResponse.arrayBuffer();
            })
      );
      if (signal.aborted) throw new DOMException("回覆已中斷。", "AbortError");
      const audioBuffer = await voiceState.audioContext.decodeAudioData(bytes.slice(0));
      await new Promise((resolve, reject) => {
        const source = voiceState.audioContext.createBufferSource();
        let settled = false;
        const finish = () => {
          if (settled) return;
          settled = true;
          signal.removeEventListener("abort", abort);
          voiceState.activeSources.delete(source);
          resolve();
        };
        const abort = () => {
          if (settled) return;
          settled = true;
          voiceState.activeSources.delete(source);
          try {
            source.stop();
          } catch {}
          reject(new DOMException("回覆已中斷。", "AbortError"));
        };
        source.buffer = audioBuffer;
        source.connect(voiceState.audioContext.destination);
        source.onended = finish;
        signal.addEventListener("abort", abort, { once: true });
        voiceState.activeSources.add(source);
        setVoiceStatus("正在為您查詢核准知識…");
        source.start();
      });
      setVoiceStatus("正在整理核准知識回答…");
    } catch (error) {
      if (error.name === "AbortError") throw error;
    }
  };

  const handleEvent = async (event) => {
    if (signal.aborted) throw new DOMException("回覆已中斷。", "AbortError");
    if (event.type === "acknowledgement") {
      const audioUrl = VoicePlayback.chooseNonRepeatingAudioUrl(
        event.audio_urls || [event.audio_url],
        voiceState.lastAcknowledgementAudioUrl,
      );
      voiceState.lastAcknowledgementAudioUrl = audioUrl;
      await playAcknowledgement(audioUrl);
      return;
    }
    if (event.type === "turn") {
      document.querySelector("#voice-loading-message")?.remove();
      appendAssistantMessage(event.turn, true, {
        progressiveSegments: voiceTestMode ? event.speech_segments || [] : [],
      });
      responseSubtitleNodes = [...voiceState.subtitleNodes];
      voiceState.replyDisplayed = true;
      turnId = event.turn.turn_id;
      prepareScheduler(event.turn.result.answer.length);
      if (voiceState.config?.barge_in?.enabled) {
        armBargeIn();
        startRealtimeCapture();
      }
      contentReceived = true;
      setVoiceStatus("文字回答完成，正在產生語音…可直接說話中斷");
      return;
    }
    if (
      event.type === "greeting"
      || event.type === "farewell"
      || event.type === "idle_prompt"
    ) {
      const segments = event.speech_segments || [];
      const isIdlePrompt = event.type === "idle_prompt";
      const isFarewell = event.type === "farewell" || Boolean(event.ends_call);
      document.querySelector("#voice-loading-message")?.remove();
      appendAssistantMessage(
        {
          turn_id: isFarewell
            ? "voice-call-farewell"
            : isIdlePrompt
              ? "voice-idle-check-in"
              : "voice-test-greeting",
          result: {
            decision: "answer",
            answer: segments.join(""),
            citations: [],
          },
        },
        false,
        {
          progressiveSegments: segments,
          label: isFarewell
            ? "AI 客服・結束語"
            : isIdlePrompt
              ? "AI 客服・確認提示"
              : "AI 客服・招呼語",
          showDecision: false,
        },
      );
      responseSubtitleNodes = [...voiceState.subtitleNodes];
      prepareScheduler(segments.join("").length);
      contentReceived = true;
      greetingResponse = !isFarewell && !isIdlePrompt;
      farewellResponse = isFarewell;
      idlePromptResponse = isIdlePrompt && !isFarewell;
      setVoiceStatus(
        isFarewell
          ? "正在產生結束語音…"
          : isIdlePrompt
            ? "正在產生確認提示語音…"
            : "正在產生招呼語音…",
      );
      return;
    }
    if (event.type === "error") {
      ttsFailed = true;
      revealAllVoiceSubtitles(responseSubtitleNodes);
      setVoiceStatus(event.detail || "語音合成暫時無法使用。");
      return;
    }
    if (event.type === "audio") {
      if (!scheduler) throw new Error("語音播放排程尚未就緒。");
      const arrivalTimeMs = performance.now();
      const bytes = Uint8Array.from(atob(event.audio), (char) => char.charCodeAt(0));
      const audioBuffer = await voiceState.audioContext.decodeAudioData(bytes.buffer.slice(0));
      scheduler.enqueue(audioBuffer, {
        arrivalTimeMs,
        onStart:
          event.sentence_chunk_index === 1
            ? () => revealVoiceSubtitle(event.sentence_index, responseSubtitleNodes)
            : null,
      });
      audioReceived = true;
      voiceButton.classList.add("speaking");
      updateCallControls("speaking");
      const playbackLabel = farewellResponse
        ? "結束語"
        : idlePromptResponse
          ? "確認提示"
        : greetingResponse
          ? "招呼語"
          : "語音回答";
      setVoiceStatus(
        scheduler.started
          ? `正在播放${playbackLabel}…`
          : `正在緩衝${playbackLabel}…`,
      );
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (line.trim()) await handleEvent(JSON.parse(line));
      }
      if (done) break;
    }
    if (pending.trim()) await handleEvent(JSON.parse(pending));
    if (!contentReceived) throw new Error("沒有收到知識助手回答。");
    if (!audioReceived && !ttsFailed) throw new Error("沒有收到可播放的語音。");
    if (audioReceived && scheduler) {
      const playbackLabel = farewellResponse
        ? "結束語"
        : idlePromptResponse
          ? "確認提示"
        : greetingResponse
          ? "招呼語"
          : "語音回答";
      setVoiceStatus(`正在播放${playbackLabel}…`);
      await scheduler.finish();
    }
    revealAllVoiceSubtitles(responseSubtitleNodes);
    completed = true;
  } finally {
    if (!completed) scheduler?.interrupt();
    if (scheduler && voiceState.playbackScheduler === scheduler) {
      voiceState.playbackScheduler = null;
    }
    if (scheduler && voiceState.bargeInDetector) {
      scheduler.recordBargeInMetrics?.(voiceState.bargeInDetector.metrics());
    }
    if (scheduler?.metrics) reportVoicePlayback(turnId, scheduler.metrics());
    if (!voiceState.listening) {
      disarmBargeIn();
      stopRealtimeCapture();
    }
  }
  return { endsCall: farewellResponse };
}

async function handleVoiceIdleTimeout() {
  if (
    !voiceTestMode
    || !voiceState.continuous
    || !voiceState.listening
    || voiceState.replying
  ) {
    return;
  }

  const stage = voiceState.idlePromptStage === 0 ? "check_in" : "farewell";
  if (stage === "check_in") voiceState.idlePromptStage = 1;
  clearPendingAsrEndpoint();
  voiceState.listening = false;
  stopRealtimeCapture();
  voiceButton.classList.remove("listening");
  voiceButton.classList.add("speaking");
  updateCallControls("speaking");
  setAsrLive(
    stage === "check_in"
      ? "8 秒內沒有辨識到有效問句。"
      : "再次等待 8 秒後仍沒有辨識到有效問句。",
  );
  setVoiceStatus(
    stage === "check_in" ? "準備播放確認提示…" : "準備播放通話結束語…",
  );

  const controller = new AbortController();
  voiceState.replyController = controller;
  voiceState.replying = true;
  voiceState.replyDisplayed = false;
  try {
    const response = await fetch("/v1/voice/idle-prompt-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify({ stage }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("等待提示語音服務目前無法使用。");
    const outcome = await playVoiceResponse(response, controller.signal);
    if (outcome.endsCall) {
      stopVoiceSession("等待逾時，本次通話已自動結束");
      setAsrLive("本次通話已自動結束。");
      return;
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      stopVoiceSession(`等待提示播放失敗：${error.message}`);
    }
    return;
  } finally {
    if (voiceState.replyController === controller) {
      voiceState.replyController = null;
      voiceState.replying = false;
      voiceButton.classList.remove("speaking");
    }
  }

  if (voiceState.continuous && !controller.signal.aborted) {
    await startVoiceListening({ preserveIdleStage: true });
  }
}

async function submitVoiceTurn(transcript) {
  clearVoiceIdleTimer();
  voiceState.idlePromptStage = 0;
  if (!transcript) {
    setVoiceStatus("沒有辨識到語音，請再說一次。");
    if (voiceState.continuous) await startVoiceListening();
    return;
  }
  appendUserMessage(transcript);
  appendLoading("voice-loading-message");
  setVoiceStatus("正在查核核准知識與安全邊界…");
  voiceState.replying = true;
  voiceState.replyDisplayed = false;
  voiceButton.classList.add("speaking");
  const controller = new AbortController();
  voiceState.replyController = controller;
  try {
    const requestBody = {
      transcript,
      reply_mode: selectedReplyMode(),
      conversation_id: ensureVoiceTestSession(),
    };
    const response = await fetch("/v1/voice/respond-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("語音回答服務目前無法使用。");
    const outcome = await playVoiceResponse(response, controller.signal);
    if (outcome.endsCall) {
      stopVoiceSession("已播放結束語，本次通話已結束");
      setAsrLive("客戶已結束本次通話。");
      return;
    }
    setVoiceStatus("本輪完成，正在重新開啟麥克風…");
  } catch (error) {
    const disposition = VoicePlayback.voiceFailureDisposition(
      error.name,
      voiceState.replyDisplayed,
    );
    if (disposition === "service_unavailable") {
      document.querySelector("#voice-loading-message")?.remove();
      appendNetworkError();
      stopVoiceSession(`語音處理失敗：${error.message}`);
      return;
    }
    if (disposition === "playback_degraded") {
      setVoiceStatus("文字回答已保留；語音播放未完整完成，正在重新開啟麥克風…");
    }
  } finally {
    if (voiceState.replyController === controller) {
      voiceState.replyController = null;
      voiceState.replying = false;
      voiceButton.classList.remove("speaking");
    }
  }
  if (voiceState.continuous && !controller.signal.aborted) await startVoiceListening();
}

async function startVoiceTestCall() {
  const greeting = greetingInput.value.trim();
  if (!greeting) {
    setVoiceStatus("請先設定招呼語。");
    greetingInput.focus();
    return;
  }

  ensureVoiceTestSession();
  voiceState.lastAcknowledgementAudioUrl = null;
  voiceState.continuous = true;
  updateCallControls("connecting");
  startCallClock();
  setAsrLive("正在建立通話，等待招呼語播放完成…");
  try {
    asrModelSelect.disabled = true;
    bargeInModeSelect.disabled = true;
    if (replyModeSelect) replyModeSelect.disabled = true;
    await ensureMicrophone();
    if (!voiceState.continuous) return;
    const controller = new AbortController();
    voiceState.replyController = controller;
    voiceState.replying = true;
    const response = await fetch("/v1/voice/test-greeting-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify({ greeting }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("招呼語音服務目前無法使用。");
    await playVoiceResponse(response, controller.signal);
    if (voiceState.continuous && !controller.signal.aborted) await startVoiceListening();
  } catch (error) {
    if (error.name !== "AbortError") {
      stopVoiceSession(`無法開始通話：${error.message}`);
    }
  } finally {
    voiceState.replying = false;
    voiceState.replyController = null;
    voiceButton.classList.remove("speaking");
  }
}

async function toggleVoiceSession() {
  if (voiceState.replying) {
    cancelBargeInCandidate();
    disarmBargeIn();
    voiceState.replyController?.abort();
    stopVoicePlayback();
    voiceState.replying = false;
    voiceState.continuous = true;
    await startVoiceListening();
    return;
  }
  if (voiceState.continuous) {
    stopVoiceSession();
    return;
  }
  if (voiceTestMode) {
    await startVoiceTestCall();
    return;
  }
  voiceState.continuous = true;
  await startVoiceListening();
}

async function loadVoiceConfig() {
  try {
    const response = await fetch("/v1/voice/config", { headers: { Accept: "application/json" } });
    const config = await response.json();
    voiceState.config = config;
    voiceState.acknowledgementAudio.clear();
    for (const acknowledgementAudioUrl of Object.values(
      config.acknowledgement?.audio_urls || {},
    )) {
      const bytesPromise = fetch(acknowledgementAudioUrl, { cache: "force-cache" }).then(
        (audioResponse) => {
          if (!audioResponse.ok) throw new Error("無法預載等待提示音。");
          return audioResponse.arrayBuffer();
        },
      );
      voiceState.acknowledgementAudio.set(acknowledgementAudioUrl, bytesPromise);
      bytesPromise.catch(() => {});
    }
    if (diagnosticLogNote) {
      diagnosticLogNote.hidden = !config.diagnostic_content_logging_enabled;
    }
    if (replyModeSelect) {
      const replyModes = config.reply_modes || [{ id: "exact", label: "核准原文" }];
      replyModeSelect.replaceChildren();
      for (const mode of replyModes) {
        const option = document.createElement("option");
        option.value = mode.id;
        option.textContent = mode.label;
        replyModeSelect.append(option);
      }
      replyModeSelect.value = "exact";
      replyModeSelect.disabled = replyModes.length < 2;
    }
    if (!config.enabled) {
      setStatusValue(voiceServiceStatus, "error", "未啟用");
      setVoiceStatus("語音功能尚未啟用；仍可使用文字代測");
      return;
    }
    const asrModels = [...new Set(config.asr_models || [config.models?.asr])].filter(Boolean);
    asrModelSelect.replaceChildren();
    asrModels.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = asrModelLabel(model);
      asrModelSelect.append(option);
    });
    asrModelSelect.value = config.models.asr;
    asrModelSelect.disabled = false;
    asrModelControl.hidden = !voiceTestMode && asrModels.length < 2;
    const bargeIn = config.barge_in;
    bargeInModeSelect.replaceChildren();
    for (const preset of bargeIn?.presets || []) {
      const option = document.createElement("option");
      option.value = preset.id;
      option.textContent = preset.label;
      bargeInModeSelect.append(option);
    }
    bargeInModeSelect.value = bargeIn?.default_mode || "standard";
    bargeInModeSelect.disabled = false;
    bargeInControl.hidden = !bargeIn?.enabled;
    configureBargeInDetector();
    voiceButton.hidden = false;
    voiceButton.disabled = !config.available;
    setStatusValue(
      voiceServiceStatus,
      config.available ? "ready" : "warning",
      config.available ? "服務正常" : "模型尚未就緒",
    );
    if (asrStatus) asrStatus.textContent = `${asrModelLabel(config.models.asr)}｜待機`;
    if (ttsStatus) {
      ttsStatus.textContent = `${asrModelLabel(config.models.tts)}｜${config.models.voice}`;
    }
    setVoiceStatus(
      config.available
        ? voiceTestMode
          ? "按「打電話」開始模擬客服來電"
          : "按麥克風開始即時語音對話"
        : "語音模型服務尚未就緒",
    );
  } catch {
    setStatusValue(voiceServiceStatus, "error", "無法取得狀態");
    setVoiceStatus("無法取得語音服務狀態");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const transcript = input.value.trim();
  if (!transcript || sendButton.disabled) return;

  appendUserMessage(transcript);
  input.value = "";
  input.style.height = "auto";
  characterCount.value = "0";
  sendButton.disabled = true;
  messages.setAttribute("aria-busy", "true");
  appendLoading();

  try {
    const voiceTestRequest = {
      transcript,
      session_id: ensureVoiceTestSession(),
      reply_mode: selectedReplyMode(),
    };
    const response = await fetch(
      voiceTestMode ? "/v1/voice/test-turns/evaluate" : "/v1/turns/evaluate",
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(
          voiceTestMode ? voiceTestRequest : { transcript, channel: "web" },
        ),
      },
    );
    if (!response.ok) throw new Error("turn request failed");
    const turn = await response.json();
    document.querySelector("#loading-message")?.remove();
    appendAssistantMessage(turn);
  } catch {
    document.querySelector("#loading-message")?.remove();
    appendNetworkError();
  } finally {
    sendButton.disabled = false;
    messages.setAttribute("aria-busy", "false");
    input.focus();
  }
});

input.addEventListener("input", () => {
  characterCount.value = String(input.value.length);
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
});

clearButton.addEventListener("click", () => {
  if (voiceTestMode) clearVoiceConversation({ rotate: true });
  messages.replaceChildren(initialMessage.cloneNode(true));
  voiceState.subtitleNodes = [];
  input.focus();
});

copySessionIdButton?.addEventListener("click", async () => {
  const sessionId = ensureVoiceTestSession();
  if (!sessionId) return;
  await navigator.clipboard.writeText(sessionId);
  copySessionIdButton.textContent = "已複製";
  setTimeout(() => {
    copySessionIdButton.textContent = "複製";
  }, 1200);
});

voiceButton.addEventListener("click", toggleVoiceSession);
hangupButton?.addEventListener("click", () => {
  stopVoiceSession("通話已結束；可調整參數後再次撥打");
  setAsrLive("本次通話已結束。");
});
asrModelSelect.addEventListener("change", () => {
  closeRealtimeAsr();
  setVoiceStatus(`已切換為 ${asrModelLabel(selectedAsrModel())}；按麥克風開始測試`);
});
bargeInModeSelect.addEventListener("change", () => {
  configureBargeInDetector();
  const preset = selectedBargeInPreset();
  setVoiceStatus(`插話偵測已切換為「${preset?.label || "標準"}」模式`);
});
loadSystemState();
loadVoiceConfig();
ensureVoiceTestSession();
initializeVoiceTestLayout();
