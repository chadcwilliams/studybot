// Reads its config from window.STUDYBOT_CONFIG, set by a small inline
// script in each course's index.html — this file itself is shared across
// every course (see docs/<course>/index.html), so a fix here reaches every
// class without needing to copy this file into each course's folder.
const config = window.STUDYBOT_CONFIG || {};
const API_BASE_URL = config.apiBaseUrl || "http://localhost:8000";
const MAX_HISTORY_MESSAGES = 12; // ~6 turns — mirrors the backend's own cap

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");

// Tracks the conversation so follow-ups ("lay it out in steps") are
// understood in relation to what was already asked, instead of each
// question being answered in isolation.
let conversationHistory = [];

function addMessage(html, className) {
  const div = document.createElement("div");
  div.className = `msg ${className}`;
  div.innerHTML = html;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

// Renders LLM output as Markdown (tables, bold, lists, etc.) and sanitizes
// the result before it touches the DOM — the model's output isn't fully
// trusted input, so this matters even though prompts constrain it.
function renderMarkdown(str) {
  const rawHtml = marked.parse(str, { breaks: true });
  return DOMPurify.sanitize(rawHtml);
}

// Renders LaTeX math (KaTeX) inside an already-inserted DOM element. Run
// this AFTER the element's HTML is set, not as part of the Markdown/sanitize
// step — it walks real text nodes looking for delimiters, so it needs the
// final DOM to exist first. No-ops safely if KaTeX's script failed to load.
function renderMathIn(el) {
  if (window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "typing";
  div.id = "typing-indicator";
  div.textContent = "Thinking…";
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

async function askQuestion(question) {
  addMessage(`<p>${escapeHtml(question)}</p>`, "msg-user");
  showTyping();
  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: conversationHistory }),
    });

    hideTyping();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const message = err.detail || "Something went wrong. Please try again.";
      addMessage(`<p>${escapeHtml(message)}</p>`, "msg-error");
      return;
    }

    const data = await res.json();
    let html = renderMarkdown(data.answer);
    if (data.sources && data.sources.length) {
      html += `<div class="sources">Source: ${data.sources.map(escapeHtml).join(", ")}</div>`;
    }
    const botDiv = addMessage(html, "msg-bot");
    renderMathIn(botDiv);

    conversationHistory.push({ role: "user", content: question });
    conversationHistory.push({ role: "assistant", content: data.answer });
    if (conversationHistory.length > MAX_HISTORY_MESSAGES) {
      conversationHistory = conversationHistory.slice(-MAX_HISTORY_MESSAGES);
    }
  } catch (e) {
    hideTyping();
    addMessage(
      `<p>Couldn't reach the course assistant. Check your connection and try again.</p>`,
      "msg-error"
    );
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  askQuestion(question);
});

// Auto-grow the textarea, submit on Enter (Shift+Enter for newline)
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});
