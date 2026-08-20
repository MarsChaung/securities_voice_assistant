const runButton = document.querySelector("#run-diagnostics");
const copyButton = document.querySelector("#copy-report");
const progressCard = document.querySelector("#progress-card");
const results = document.querySelector("#diagnostic-results");
const summaryCard = document.querySelector("#summary-card");
const overallLabel = document.querySelector("#overall-label");
const generatedAt = document.querySelector("#generated-at");
const countOutputs = {
  pass: document.querySelector("#pass-count"),
  warning: document.querySelector("#warning-count"),
  fail: document.querySelector("#fail-count"),
  skipped: document.querySelector("#skipped-count"),
};

const statusLabels = {
  pass: "通過",
  warning: "警告",
  fail: "失敗",
  skipped: "略過",
};

let latestReport = null;

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function appendDetail(card, title, items, className = "") {
  const block = makeElement("section", `detail-block ${className}`.trim());
  block.append(makeElement("h3", "", title));
  if (!items?.length) {
    block.append(makeElement("p", "detail-empty", "無"));
    card.append(block);
    return;
  }
  const list = document.createElement("ul");
  for (const item of items) list.append(makeElement("li", "", item));
  block.append(list);
  card.append(block);
}

function renderCheck(check) {
  const card = makeElement("article", "check-card");
  card.dataset.status = check.status;

  const heading = makeElement("header", "check-heading");
  heading.append(makeElement("span", "status-badge", statusLabels[check.status] || check.status));
  const title = makeElement("div", "check-title");
  title.append(makeElement("small", "", check.category));
  title.append(makeElement("strong", "", check.title));
  heading.append(title);
  heading.append(makeElement("span", "duration", `${Math.round(check.duration_ms)} ms`));
  card.append(heading);
  card.append(makeElement("p", "check-summary", check.summary));

  const details = makeElement("div", "check-details");
  appendDetail(details, "安全化證據", check.evidence);
  appendDetail(details, "建議處置", check.remediation, "remediation");
  card.append(details);
  results.append(card);
}

function overallStatus(checks) {
  if (checks.some((check) => check.status === "fail")) return "fail";
  if (checks.some((check) => check.status === "warning")) return "warning";
  return "pass";
}

function renderSummary(checks, timestamp) {
  const counts = { pass: 0, warning: 0, fail: 0, skipped: 0 };
  for (const check of checks) counts[check.status] += 1;
  for (const [status, output] of Object.entries(countOutputs)) {
    output.textContent = String(counts[status]);
  }
  const status = overallStatus(checks);
  summaryCard.dataset.status = status;
  overallLabel.textContent = {
    pass: "所有必要檢查通過",
    warning: "可運作，但有項目需要確認",
    fail: "發現會阻擋功能的問題",
  }[status];
  generatedAt.textContent = new Intl.DateTimeFormat("zh-TW", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(timestamp));
}

function browserASRFailure(summary, evidence = []) {
  return {
    check_id: "asr_browser",
    category: "瀏覽器路徑",
    title: "ASR WebSocket 實際連線",
    status: "fail",
    summary,
    evidence: [
      `頁面：${window.location.origin}`,
      `Secure Context：${window.isSecureContext ? "是" : "否"}`,
      ...evidence,
    ],
    remediation: [
      "確認瀏覽器可解析 ASR DNS，且 HTTPS/WSS 憑證由公司瀏覽器信任。",
      "Reverse proxy 必須支援 WebSocket Upgrade，並允許目前頁面的 Origin。",
      "目前前端無法附加 Bearer header；若 ASR 強制驗證，需由同源 proxy 注入或調整驗證方式。",
      "確認端點為 /v1/audio/transcriptions/realtime，且支援本系統的 ready／delta／utterance_end 事件。",
    ],
    duration_ms: 0,
  };
}

function probeBrowserASR(probe) {
  const startedAt = performance.now();
  if (window.location.protocol === "https:" && probe.url.startsWith("ws:")) {
    const check = browserASRFailure("HTTPS 頁面會阻擋未加密的 ws:// ASR 連線。", [
      `WebSocket：${probe.url}`,
    ]);
    check.duration_ms = performance.now() - startedAt;
    return Promise.resolve(check);
  }

  return new Promise((resolve) => {
    let socket;
    let settled = false;
    const finish = (check) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      check.duration_ms = performance.now() - startedAt;
      try {
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ action: "stop" }));
          socket.close(1000, "diagnostic complete");
        }
      } catch {}
      resolve(check);
    };
    const timeout = setTimeout(() => {
      finish(browserASRFailure("ASR WebSocket 在期限內沒有回傳 ready。", [
        `WebSocket：${probe.url}`,
        `模型：${probe.model}`,
      ]));
    }, probe.timeout_ms);

    try {
      socket = new WebSocket(probe.url);
    } catch {
      finish(browserASRFailure("瀏覽器拒絕建立 ASR WebSocket。", [
        `WebSocket：${probe.url}`,
      ]));
      return;
    }
    socket.onopen = () => socket.send(JSON.stringify(probe.init_message));
    socket.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        finish(browserASRFailure("ASR WebSocket 回傳非 JSON 訊息。"));
        return;
      }
      if (payload.error) {
        finish(browserASRFailure("ASR WebSocket 回傳錯誤事件。", [
          `WebSocket：${probe.url}`,
          `模型：${probe.model}`,
        ]));
        return;
      }
      if (payload.status === "ready") {
        finish({
          check_id: "asr_browser",
          category: "瀏覽器路徑",
          title: "ASR WebSocket 實際連線",
          status: "pass",
          summary: "本瀏覽器成功連線 ASR，模型完成初始化並回傳 ready。",
          evidence: [
            `WebSocket：${probe.url}`,
            `模型：${probe.model}`,
            `Secure Context：${window.isSecureContext ? "是" : "否"}`,
          ],
          remediation: [],
          duration_ms: 0,
        });
      }
    };
    socket.onerror = () => finish(browserASRFailure("ASR WebSocket 連線失敗。", [
      `WebSocket：${probe.url}`,
      `模型：${probe.model}`,
    ]));
    socket.onclose = () => {
      if (!settled) finish(browserASRFailure("ASR WebSocket 在 ready 前關閉。", [
        `WebSocket：${probe.url}`,
      ]));
    };
  });
}

async function runDiagnostics() {
  runButton.disabled = true;
  copyButton.disabled = true;
  progressCard.hidden = false;
  results.replaceChildren();
  summaryCard.dataset.status = "running";
  overallLabel.textContent = "正在執行診斷…";
  generatedAt.textContent = "等待報告";
  for (const output of Object.values(countOutputs)) output.textContent = "—";

  try {
    const response = await fetch("/v1/system-diagnostics/run", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        page_origin: window.location.origin,
        secure_context: window.isSecureContext,
      }),
    });
    if (!response.ok) throw new Error("diagnostic request failed");
    const report = await response.json();
    const checks = [...report.checks];
    for (const check of checks) renderCheck(check);

    if (report.browser_asr_probe) {
      const pending = {
        check_id: "asr_browser_pending",
        category: "瀏覽器路徑",
        title: "ASR WebSocket 實際連線",
        status: "skipped",
        summary: "正在從本瀏覽器連線 ASR 並等待模型 ready…",
        evidence: [`WebSocket：${report.browser_asr_probe.url}`],
        remediation: [],
        duration_ms: 0,
      };
      renderCheck(pending);
      const browserCheck = await probeBrowserASR(report.browser_asr_probe);
      results.lastElementChild?.remove();
      checks.push(browserCheck);
      renderCheck(browserCheck);
    }

    latestReport = { ...report, checks, overall_status: overallStatus(checks) };
    renderSummary(checks, report.generated_at);
    copyButton.disabled = false;
  } catch {
    results.replaceChildren(makeElement(
      "div",
      "fatal-message",
      "無法啟動診斷程序。請確認 orchestrator 仍在執行、診斷功能已啟用，並檢查瀏覽器 Network 與容器 log。",
    ));
    summaryCard.dataset.status = "fail";
    overallLabel.textContent = "診斷程序無法執行";
  } finally {
    progressCard.hidden = true;
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runDiagnostics);
copyButton.addEventListener("click", async () => {
  if (!latestReport) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(latestReport, null, 2));
    copyButton.textContent = "已複製";
    setTimeout(() => { copyButton.textContent = "複製報告"; }, 1500);
  } catch {
    copyButton.textContent = "複製失敗";
  }
});

void runDiagnostics();
