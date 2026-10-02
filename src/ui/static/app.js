/* ─────────────────────────────────────────────────────────────────────────────
   app.js  —  Nexus RAG Chatbot UI
   Talks to:
     POST /chat          → ChatRequest  → ChatResponse
     POST /chat/resume   → ResumeRequest → ChatResponse
     GET  /health        → HealthResponse
   ───────────────────────────────────────────────────────────────────────────── */

const API = '';           // same origin — FastAPI serves this file
let threadId    = null;   // current session thread
let interrupted = false;  // waiting for approval?

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatWrap        = document.getElementById('chat-wrap');
const emptyState      = document.getElementById('empty-state');
const questionEl      = document.getElementById('question');
const sendBtn         = document.getElementById('btn-send');
const sfUsernameEl    = document.getElementById('sf-username');
const threadPill      = document.getElementById('thread-pill');
const healthDot       = document.getElementById('health-dot');
const approvalPanel   = document.getElementById('approval-panel');
const approvalDetails = document.getElementById('approval-details');
const btnApprove      = document.getElementById('btn-approve');
const btnReject       = document.getElementById('btn-reject');
const btnNew          = document.getElementById('btn-new');

// Configure marked if available
if (window.marked) {
  marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: function(code, lang) {
      if (window.hljs && lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (e) {}
      }
      return code;
    }
  });
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    healthDot.className = r.ok ? 'status-indicator ok' : 'status-indicator error';
  } catch {
    healthDot.className = 'status-indicator error';
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Thread management ─────────────────────────────────────────────────────────
function setThread(id) {
  threadId = id;
  const span = threadPill.querySelector('span');
  if (span) {
    span.textContent = id ? `thread: ${id.slice(0, 16)}…` : 'no session';
  } else {
    threadPill.textContent = id ? `thread: ${id.slice(0, 16)}…` : 'no session';
  }
}

function newChat() {
  setThread(null);
  interrupted = false;
  hideApproval();
  chatWrap.innerHTML = '';
  chatWrap.appendChild(emptyState);
  emptyState.style.display = 'flex';
}
btnNew.addEventListener('click', newChat);

// Prompt card helper
window.fillPrompt = function(text) {
  questionEl.value = text;
  autoResize();
  questionEl.focus();
};

// ── Render helpers ────────────────────────────────────────────────────────────
function hideEmpty() {
  if (emptyState) {
    emptyState.style.display = 'none';
  }
}

function addMessage(role, text) {
  hideEmpty();
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = role === 'user' 
    ? '<span>U</span>' 
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
         <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
       </svg>`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'assistant') {
    // Rich render via marked if available
    if (window.marked && typeof text === 'string') {
      try {
        bubble.innerHTML = marked.parse(text);
      } catch (err) {
        bubble.textContent = text;
      }
    } else {
      bubble.textContent = text;
    }
  } else {
    bubble.textContent = text;
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWrap.appendChild(row);

  // Smooth scroll into view
  const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
  viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });

  return bubble;
}

function addTypingIndicator() {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'typing-row';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
  </svg>`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWrap.appendChild(row);

  const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
  viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
}

function removeTypingIndicator() {
  document.getElementById('typing-row')?.remove();
}

// ── Approval panel ────────────────────────────────────────────────────────────
function showApproval(payload) {
  interrupted = true;
  const displayName = payload.record_name || payload.opportunity_name || payload.record_id || payload.opportunity_id || '—';
  const changeDetail = payload.new_status
    ? `<div><strong>New Stage:</strong> <span style="background:rgba(245,158,11,0.15); color:#fbbf24; padding:2px 8px; border-radius:4px; font-weight:600;">${payload.new_status}</span></div>`
    : '';

  approvalDetails.innerHTML = `
    <div style="margin-bottom: 6px;"><strong>Request:</strong> <span>${payload.message ?? 'Update requested'}</span></div>
    <div style="margin-bottom: 6px;"><strong>Record:</strong> <span style="color:#fff; font-weight:600;">${displayName}</span></div>
    ${changeDetail}
  `;

  approvalPanel.classList.add('visible');
  approvalPanel.scrollIntoView({ behavior: 'smooth' });
}

function hideApproval() {
  interrupted = false;
  approvalPanel.classList.remove('visible');
}

// ── Core API calls ────────────────────────────────────────────────────────────
async function sendChat(question) {
  setLoading(true);
  addMessage('user', question);
  addTypingIndicator();

  try {
    const payload = {
      question,
      thread_id:   threadId   || undefined,
      sf_username: sfUsernameEl.value.trim() || undefined,
    };

    const r = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeTypingIndicator();

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      addMessage('assistant', `⚠️ **Error ${r.status}**: ${err.detail ?? r.statusText}`);
      return;
    }

    const data = await r.json();
    console.debug('[RAG] /chat response:', data);
    setThread(data.thread_id ?? threadId);

    if (data.interrupted === true) {
      addMessage('assistant', `⏸ **Approval Required**: The agent needs confirmation before proceeding with this write operation.`);
      showApproval(data.approval_request ?? {});
    } else {
      addMessage('assistant', data.answer || '(empty response)');
    }
  } catch (e) {
    removeTypingIndicator();
    addMessage('assistant', `⚠️ **Network Error**: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

async function sendResume(decision) {
  if (!threadId) return;
  setApprovalLoading(true);
  addTypingIndicator();

  try {
    const r = await fetch(`${API}/chat/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, decision }),
    });

    removeTypingIndicator();
    hideApproval();

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      addMessage('assistant', `⚠️ **Resume Error ${r.status}**: ${err.detail ?? r.statusText}`);
      return;
    }

    const data = await r.json();
    addMessage('assistant', `**[${decision}]** — ${data.answer || '(action completed)'}`);

    // Edge case: another approval in chain
    if (data.interrupted) {
      showApproval(data.approval_request ?? {});
    }

  } catch (e) {
    removeTypingIndicator();
    addMessage('assistant', `⚠️ **Network Error**: ${e.message}`);
  } finally {
    setApprovalLoading(false);
  }
}

// ── Loading states ────────────────────────────────────────────────────────────
function setLoading(on) {
  sendBtn.disabled    = on;
  questionEl.disabled = on;
  if (!on) {
    questionEl.focus();
  }
}

function setApprovalLoading(on) {
  btnApprove.disabled = on;
  btnReject.disabled  = on;
}

// ── Input events ──────────────────────────────────────────────────────────────
function submit() {
  if (interrupted) return;   // must resolve approval first
  const q = questionEl.value.trim();
  if (!q) return;
  questionEl.value = '';
  autoResize();
  sendChat(q);
}

sendBtn.addEventListener('click', submit);

questionEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
});

// Auto-resize textarea
function autoResize() {
  questionEl.style.height = 'auto';
  questionEl.style.height = Math.min(questionEl.scrollHeight, 160) + 'px';
}
questionEl.addEventListener('input', autoResize);

// ── Approval buttons ──────────────────────────────────────────────────────────
btnApprove.addEventListener('click', () => sendResume('Approve'));
btnReject.addEventListener ('click', () => sendResume('Reject'));

// ── Init ──────────────────────────────────────────────────────────────────────
setThread(null);
