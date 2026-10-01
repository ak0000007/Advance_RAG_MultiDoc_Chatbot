/* ─────────────────────────────────────────────────────────────────────────────
   app.js  —  RAG Chatbot UI
   Talks to:
     POST /chat          → ChatRequest  → ChatResponse
     POST /chat/resume   → ResumeRequest → ChatResponse
     GET  /health        → HealthResponse
   ───────────────────────────────────────────────────────────────────────────── */

const API = '';           // same origin — FastAPI serves this file
let threadId    = null;   // current session thread
let interrupted = false;  // waiting for approval?

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatWrap      = document.getElementById('chat-wrap');
const emptyState    = document.getElementById('empty-state');
const questionEl    = document.getElementById('question');
const sendBtn       = document.getElementById('btn-send');
const sfUsernameEl  = document.getElementById('sf-username');
const threadPill    = document.getElementById('thread-pill');
const healthDot     = document.getElementById('health-dot');
const approvalPanel = document.getElementById('approval-panel');
const approvalDetails = document.getElementById('approval-details');
const btnApprove    = document.getElementById('btn-approve');
const btnReject     = document.getElementById('btn-reject');
const btnNew        = document.getElementById('btn-new');

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    healthDot.className = r.ok ? 'ok' : 'error';
  } catch {
    healthDot.className = 'error';
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Thread management ─────────────────────────────────────────────────────────
function setThread(id) {
  threadId = id;
  threadPill.textContent = id ? `thread: ${id.slice(0, 20)}…` : 'no session';
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

// ── Render helpers ────────────────────────────────────────────────────────────
function hideEmpty() {
  emptyState.style.display = 'none';
}

function addMessage(role, text) {
  hideEmpty();
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWrap.appendChild(row);
  chatWrap.scrollTop = chatWrap.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.id = 'typing-row';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWrap.appendChild(row);
  chatWrap.scrollTop = chatWrap.scrollHeight;
}

function removeTypingIndicator() {
  document.getElementById('typing-row')?.remove();
}

// ── Approval panel ────────────────────────────────────────────────────────────
function showApproval(payload) {
  interrupted = true;
  const displayName = payload.opportunity_name || payload.opportunity_id || '—';
  approvalDetails.innerHTML = `
    <strong>Message:</strong> ${payload.message ?? '—'}<br>
    ${displayName           ? `<strong>Opportunity:</strong> ${displayName}<br>`         : ''}
    ${payload.new_status    ? `<strong>New Status:</strong> ${payload.new_status}<br>`   : ''}
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
      addMessage('assistant', `⚠️ Error ${r.status}: ${err.detail ?? r.statusText}`);
      return;
    }

    const data = await r.json();
    console.debug('[RAG] /chat response:', data);   // ← inspect in DevTools
    setThread(data.thread_id ?? threadId);

    if (data.interrupted === true) {
      addMessage('assistant', `⏸ Approval required — see panel below.`);
      showApproval(data.approval_request ?? {});
    } else {
      addMessage('assistant', data.answer || '(empty response)');
    }
  } catch (e) {
    removeTypingIndicator();
    addMessage('assistant', `⚠️ Network error: ${e.message}`);
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
      addMessage('assistant', `⚠️ Resume error ${r.status}: ${err.detail ?? r.statusText}`);
      return;
    }

    const data = await r.json();
    addMessage('assistant', `[${decision}] — ${data.answer || '(done)'}`);

    // Edge case: another approval in chain
    if (data.interrupted) showApproval(data.approval_request ?? {});

  } catch (e) {
    removeTypingIndicator();
    addMessage('assistant', `⚠️ Network error: ${e.message}`);
  } finally {
    setApprovalLoading(false);
  }
}

// ── Loading states ────────────────────────────────────────────────────────────
function setLoading(on) {
  sendBtn.disabled    = on;
  questionEl.disabled = on;
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
