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
const initialMessage = messages.firstElementChild.cloneNode(true);
const voiceState = {
  config: null,
  stream: null,
  audioContext: null,
  microphoneSource: null,
  processor: null,
  silentGain: null,
  socket: null,
  socketPromise: null,
  socketReady: false,
  listening: false,
  continuous: false,
  replying: false,
  replyController: null,
  activeSources: new Set(),
  partialTranscript: "",
};

const decisionLabels = {
  answer: "核准知識回答",
  clarify: "請確認語音內容",
  refuse: "安全拒答",
  handoff: "轉人工處理",
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function appendUserMessage(text) {
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

function appendAssistantMessage(turn, allowFeedback = true) {
  const result = turn.result;
  const article = element("article", `message assistant-message ${result.decision}`);
  article.append(element("div", "avatar", "知"));
  const content = element("div", "message-content");
  content.append(element("p", "message-label", "知識助手"));
  content.append(
    element(
      "div",
      `decision-badge decision-${result.decision}`,
      decisionLabels[result.decision] || "系統回覆",
    ),
  );
  const copy = element("div", "answer-copy");
  copy.append(element("p", "", result.answer));
  content.append(copy);

  const references = (result.citations || []).map(safeSourceReference).filter(Boolean);
  if (references.length) {
    const sources = element("div", "sources");
    sources.append(element("strong", "", "資料來源"), ...references);
    content.append(sources);
  }

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
  if (allowFeedback) content.append(feedbackPanel(turn.turn_id));
  article.append(content);
  messages.append(article);
  scrollToLatest();
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
    } else if (count === 0) {
      systemState.className = "system-state warning";
      systemStateText.textContent = "0 筆可回答知識｜請檢查生效與複審到期時間";
    } else {
      systemState.className = "system-state ready";
      systemStateText.textContent = `${count} 筆知識可回答`;
    }
  } catch {
    systemState.className = "system-state error";
    systemStateText.textContent = "無法取得系統狀態";
  }
}

function setVoiceStatus(text) {
  voiceStatus.textContent = text;
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
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  }
  if (!voiceState.audioContext) {
    voiceState.audioContext = new AudioContext();
    voiceState.microphoneSource = voiceState.audioContext.createMediaStreamSource(
      voiceState.stream,
    );
  }
  await voiceState.audioContext.resume();
}

function stopRealtimeCapture() {
  if (voiceState.processor) {
    try {
      voiceState.microphoneSource.disconnect(voiceState.processor);
    } catch {}
    voiceState.processor.disconnect();
    voiceState.processor.onaudioprocess = null;
    voiceState.processor = null;
  }
  if (voiceState.silentGain) {
    voiceState.silentGain.disconnect();
    voiceState.silentGain = null;
  }
}

function stopVoicePlayback() {
  for (const source of voiceState.activeSources) {
    try {
      source.stop();
    } catch {}
  }
  voiceState.activeSources.clear();
}

function startRealtimeCapture() {
  stopRealtimeCapture();
  voiceState.partialTranscript = "";
  const processor = voiceState.audioContext.createScriptProcessor(2048, 1, 1);
  const gain = voiceState.audioContext.createGain();
  gain.gain.value = 0;
  processor.onaudioprocess = (event) => {
    if (!voiceState.listening || voiceState.socket?.readyState !== WebSocket.OPEN) return;
    const pcm = downsampleToInt16(
      event.inputBuffer.getChannelData(0),
      voiceState.audioContext.sampleRate,
    );
    if (pcm.length) voiceState.socket.send(pcm.buffer);
  };
  voiceState.microphoneSource.connect(processor);
  processor.connect(gain);
  gain.connect(voiceState.audioContext.destination);
  voiceState.processor = processor;
  voiceState.silentGain = gain;
}

function handleRealtimeAsrEvent(data) {
  if (data.error) {
    stopVoiceSession("即時語音辨識發生錯誤，請稍後再試。");
    return;
  }
  if (data.type === "delta") {
    voiceState.partialTranscript += data.delta || "";
    const partial = voiceState.partialTranscript.trim();
    if (partial) setVoiceStatus(`辨識中：${partial.slice(-36)}`);
    return;
  }
  if (data.text && data.is_partial) {
    voiceState.partialTranscript = data.text.trim();
    setVoiceStatus(`辨識中：${voiceState.partialTranscript.slice(-36)}`);
    return;
  }
  if (data.type === "candidate_end") {
    voiceState.partialTranscript = (data.text || voiceState.partialTranscript).trim();
    setVoiceStatus(
      data.semantic_complete ? "正在確認你是否說完…" : "句子似乎還沒說完，持續聆聽中…",
    );
    return;
  }
  if (data.type === "speech_resumed") {
    setVoiceStatus("偵測到你繼續說話，持續聆聽中…");
    return;
  }
  if (data.type === "noise_rejected") {
    voiceState.partialTranscript = "";
    setVoiceStatus("已忽略短暫環境音，持續聆聽中…");
    return;
  }
  if (data.type === "utterance_end" && voiceState.listening) {
    voiceState.listening = false;
    stopRealtimeCapture();
    voiceButton.classList.remove("listening");
    const transcript = (data.text || voiceState.partialTranscript).trim();
    void submitVoiceTurn(transcript);
  }
}

async function ensureRealtimeAsr() {
  if (voiceState.socket?.readyState === WebSocket.OPEN && voiceState.socketReady) return;
  if (voiceState.socketPromise) return voiceState.socketPromise;
  if (!voiceState.config?.realtime_asr_url || !voiceState.config?.models?.asr) {
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
      model: voiceState.config.models.asr,
      language: "zh",
      sample_rate: 16000,
      streaming: false,
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
        voiceState.socketPromise = null;
        resolve();
        return;
      }
      handleRealtimeAsrEvent(data);
    };
    socket.onerror = () => {
      if (!settled) {
        voiceState.socketPromise = null;
        reject(new Error("無法連線到即時 ASR。"));
      }
    };
    socket.onclose = () => {
      clearTimeout(timeout);
      voiceState.socketReady = false;
      voiceState.socketPromise = null;
      if (!settled) reject(new Error("即時 ASR 連線已關閉。"));
      else if (voiceState.continuous) stopVoiceSession("即時 ASR 連線已中斷。");
    };
  });
  return voiceState.socketPromise;
}

async function startVoiceListening() {
  try {
    await ensureMicrophone();
    setVoiceStatus("正在準備即時語音辨識…");
    await ensureRealtimeAsr();
    if (!voiceState.continuous) return;
    voiceState.listening = true;
    voiceButton.classList.remove("speaking");
    voiceButton.classList.add("listening");
    voiceButton.setAttribute("aria-label", "停止即時語音對話");
    setVoiceStatus("正在聽…自然說完一句話即可");
    startRealtimeCapture();
  } catch (error) {
    stopVoiceSession(`無法啟動語音：${error.message}`);
  }
}

function stopVoiceSession(status = "語音對話已停止；按麥克風可重新開始") {
  voiceState.continuous = false;
  voiceState.listening = false;
  voiceState.replying = false;
  voiceState.replyController?.abort();
  voiceState.replyController = null;
  stopRealtimeCapture();
  stopVoicePlayback();
  if (voiceState.socket?.readyState === WebSocket.OPEN) {
    voiceState.socket.send(JSON.stringify({ action: "cancel" }));
  }
  voiceButton.classList.remove("listening", "speaking");
  voiceButton.setAttribute("aria-label", "開始即時語音對話");
  setVoiceStatus(status);
}

async function playVoiceResponse(response, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let nextStartAt = 0;
  let lastPlayback = Promise.resolve();
  let turnReceived = false;
  let audioReceived = false;
  let ttsFailed = false;

  const handleEvent = async (event) => {
    if (signal.aborted) throw new DOMException("回覆已中斷。", "AbortError");
    if (event.type === "turn") {
      document.querySelector("#voice-loading-message")?.remove();
      appendAssistantMessage(event.turn);
      turnReceived = true;
      setVoiceStatus("文字回答完成，正在產生語音…按麥克風可中斷");
      return;
    }
    if (event.type === "error") {
      ttsFailed = true;
      setVoiceStatus(event.detail || "語音合成暫時無法使用。");
      return;
    }
    if (event.type === "audio") {
      const bytes = Uint8Array.from(atob(event.audio), (char) => char.charCodeAt(0));
      const audioBuffer = await voiceState.audioContext.decodeAudioData(bytes.buffer.slice(0));
      const source = voiceState.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(voiceState.audioContext.destination);
      const now = voiceState.audioContext.currentTime;
      const startAt = Math.max(now + 0.12, nextStartAt);
      nextStartAt = startAt + audioBuffer.duration;
      lastPlayback = new Promise((resolve) => {
        source.onended = () => {
          voiceState.activeSources.delete(source);
          resolve();
        };
      });
      voiceState.activeSources.add(source);
      source.start(startAt);
      audioReceived = true;
      voiceButton.classList.add("speaking");
      setVoiceStatus("正在播放語音回答…按麥克風可中斷並繼續說話");
    }
  };

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
  if (!turnReceived) throw new Error("沒有收到知識助手回答。");
  if (!audioReceived && !ttsFailed) throw new Error("沒有收到可播放的語音。");
  await lastPlayback;
}

async function submitVoiceTurn(transcript) {
  if (!transcript) {
    setVoiceStatus("沒有辨識到語音，請再說一次。");
    if (voiceState.continuous) await startVoiceListening();
    return;
  }
  appendUserMessage(transcript);
  appendLoading("voice-loading-message");
  setVoiceStatus("正在查核核准知識與安全邊界…");
  voiceState.replying = true;
  voiceButton.classList.add("speaking");
  const controller = new AbortController();
  voiceState.replyController = controller;
  try {
    const response = await fetch("/v1/voice/respond-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify({ transcript }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("語音回答服務目前無法使用。");
    await playVoiceResponse(response, controller.signal);
    setVoiceStatus("本輪完成，正在重新開啟麥克風…");
  } catch (error) {
    if (error.name !== "AbortError") {
      document.querySelector("#voice-loading-message")?.remove();
      appendNetworkError();
      stopVoiceSession(`語音處理失敗：${error.message}`);
      return;
    }
  } finally {
    if (voiceState.replyController === controller) voiceState.replyController = null;
    voiceState.replying = false;
    voiceButton.classList.remove("speaking");
  }
  if (voiceState.continuous && !controller.signal.aborted) await startVoiceListening();
}

async function toggleVoiceSession() {
  if (voiceState.replying) {
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
  voiceState.continuous = true;
  await startVoiceListening();
}

async function loadVoiceConfig() {
  try {
    const response = await fetch("/v1/voice/config", { headers: { Accept: "application/json" } });
    const config = await response.json();
    if (!config.enabled) return;
    voiceState.config = config;
    voiceButton.hidden = false;
    setVoiceStatus(
      config.available ? "按麥克風開始即時語音對話" : "語音模型服務尚未就緒",
    );
  } catch {
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
    const response = await fetch("/v1/turns/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ transcript, channel: "web" }),
    });
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
  messages.replaceChildren(initialMessage.cloneNode(true));
  input.focus();
});

voiceButton.addEventListener("click", toggleVoiceSession);
loadSystemState();
loadVoiceConfig();
