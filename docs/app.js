// Point this at your deployed backend (Render URL) before publishing.
const API_BASE_URL = "https://studybot-api-6m2e.onrender.com";

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");

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
      body: JSON.stringify({ question }),
    });

    hideTyping();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const message = err.detail || "Something went wrong. Please try again.";
      addMessage(`<p>${escapeHtml(message)}</p>`, "msg-error");
      return;
    }

    const data = await res.json();
    let html = `<p>${escapeHtml(data.answer)}</p>`;
    if (data.sources && data.sources.length) {
      html += `<div class="sources">Source: ${data.sources.map(escapeHtml).join(", ")}</div>`;
    }
    addMessage(html, "msg-bot");
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
