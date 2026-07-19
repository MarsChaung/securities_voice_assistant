const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const characterCount = document.querySelector("#character-count");
const clearButton = document.querySelector("#clear-chat");
const systemState = document.querySelector("#system-state");
const systemStateText = document.querySelector("#system-state-text");
const initialMessage = messages.firstElementChild.cloneNode(true);

const decisionLabels = {
  answer: "核准知識回答",
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

function appendLoading() {
  const article = element("article", "message assistant-message");
  article.id = "loading-message";
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

function safeSourceLink(citation) {
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

  const links = (result.citations || []).map(safeSourceLink).filter(Boolean);
  if (links.length) {
    const sources = element("div", "sources");
    sources.append(element("strong", "", "官方來源"), ...links);
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

loadSystemState();
