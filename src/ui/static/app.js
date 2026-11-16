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

// ── Sidebar & History DOM refs ─────────────────────────────────────────────────
const historySidebar   = document.getElementById('history-sidebar');
const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
const btnRefreshHistory= document.getElementById('btn-refresh-history');
const btnSidebarNew    = document.getElementById('btn-sidebar-new');
const threadList       = document.getElementById('thread-list');

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

// ── Thread & History Management ──────────────────────────────────────────────
function highlightActiveThread(id) {
  if (!threadList) return;
  const items = threadList.querySelectorAll('.thread-item');
  items.forEach(el => {
    if (id && el.dataset.id === id) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });
}

function setThread(id) {
  threadId = id;
  if (id) {
    localStorage.setItem('activeThread', id);
    try {
      window.history.replaceState(null, '', `#${id}`);
    } catch (e) {}
  } else {
    localStorage.removeItem('activeThread');
    try {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    } catch (e) {}
  }

  const span = threadPill.querySelector('span');
  if (span) {
    span.textContent = id ? `thread: ${id.slice(0, 16)}…` : 'no session';
  } else {
    threadPill.textContent = id ? `thread: ${id.slice(0, 16)}…` : 'no session';
  }
  highlightActiveThread(id);
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
if (btnSidebarNew) {
  btnSidebarNew.addEventListener('click', newChat);
}

function formatRelativeTime(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);
    if (diffSec < 60) return 'just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

async function loadHistory() {
  if (!threadList) return;
  try {
    const r = await fetch(`${API}/threads`);
    if (!r.ok) return;
    const threads = await r.json();
    if (!Array.isArray(threads) || threads.length === 0) {
      threadList.innerHTML = '<div class="thread-empty-notice">No past conversations</div>';
      return;
    }
    threadList.innerHTML = '';
    threads.forEach(t => {
      const item = document.createElement('div');
      item.className = `thread-item ${t.thread_id === threadId ? 'active' : ''}`;
      item.dataset.id = t.thread_id;

      const titleEl = document.createElement('div');
      titleEl.className = 'thread-title';
      titleEl.textContent = t.title || 'Untitled Chat';

      const metaEl = document.createElement('div');
      metaEl.className = 'thread-meta';

      const timeSpan = document.createElement('span');
      timeSpan.textContent = formatRelativeTime(t.last_activity);

      const costSpan = document.createElement('span');
      costSpan.className = 'thread-cost';
      const costVal = Number(t.cost_usd || 0);
      costSpan.textContent = `${t.message_count} msg${t.message_count === 1 ? '' : 's'}` + (costVal > 0 ? ` · $${costVal.toFixed(4)}` : '');

      metaEl.appendChild(timeSpan);
      metaEl.appendChild(costSpan);
      item.appendChild(titleEl);
      item.appendChild(metaEl);

      item.addEventListener('click', () => {
        if (t.thread_id !== threadId) {
          loadThread(t.thread_id);
        }
      });
      threadList.appendChild(item);
    });
  } catch (err) {
    console.warn('Failed to load history:', err);
  }
}

async function loadThread(id) {
  if (!id) return;
  setLoading(true);
  addTypingIndicator();
  try {
    const r = await fetch(`${API}/threads/${id}`);
    removeTypingIndicator();
    if (!r.ok) {
      if (r.status === 404) {
        console.warn(`Thread ${id} not found or unauthorized.`);
        newChat();
      }
      return;
    }
    const data = await r.json();
    setThread(data.thread_id);

    // Clear viewport
    hideEmpty();
    chatWrap.innerHTML = '';

    // Replay messages
    if (Array.isArray(data.messages) && data.messages.length > 0) {
      data.messages.forEach(msg => {
        if (msg.role === 'human' || msg.role === 'user') {
          addMessage('user', msg.content);
        } else if (msg.role === 'ai' || msg.role === 'assistant') {
          addMessage('assistant', msg.content);
        }
      });
    } else {
      chatWrap.appendChild(emptyState);
      emptyState.style.display = 'flex';
    }

    // Check interrupted approval state
    if (data.interrupted && data.approval_request) {
      interrupted = true;
      showApproval(data.approval_request);
      addMessage('assistant', `⏸ **Approval Required**: The agent needs confirmation before proceeding with this write operation.`);
    } else {
      hideApproval();
    }

    const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
    viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
  } catch (err) {
    removeTypingIndicator();
    console.error(`Error loading thread ${id}:`, err);
  } finally {
    setLoading(false);
  }
}

// Sidebar toggle & refresh listeners
if (btnToggleSidebar && historySidebar) {
  btnToggleSidebar.addEventListener('click', () => {
    historySidebar.classList.toggle('collapsed');
  });
}
if (btnRefreshHistory) {
  btnRefreshHistory.addEventListener('click', () => {
    loadHistory();
  });
}

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

// ── Citations renderer ────────────────────────────────────────────────────────
function renderCitations(bubble, citations) {
  if (!bubble || !citations || !citations.length) return;
  if (bubble.querySelector('.citation-list')) return;

  const container = document.createElement('div');
  container.className = 'citation-list';

  const header = document.createElement('div');
  header.className = 'citation-header';
  header.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>
      <path d="M6 6h10"/><path d="M6 10h10"/>
    </svg>
    <span>Retrieved Sources (${citations.length})</span>
  `;
  container.appendChild(header);

  const chips = document.createElement('div');
  chips.className = 'citation-chips';

  citations.forEach((c, idx) => {
    const chip = document.createElement('div');
    chip.className = 'citation-chip';
    const title = c.title || c.source || `Document ${idx + 1}`;
    const pageText = (c.page !== null && c.page !== undefined) ? ` • p. ${c.page}` : '';
    const chunkTag = c.chunk_id ? `<span class="chunk-badge">${c.chunk_id.slice(0, 8)}</span>` : '';

    if (c.snippet) {
      chip.setAttribute('title', c.snippet);
    }
    chip.innerHTML = `<span class="citation-num">[${idx + 1}]</span> <span class="citation-title">${title}${pageText}</span> ${chunkTag}`;
    chips.appendChild(chip);
  });

  container.appendChild(chips);
  bubble.appendChild(container);
  const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
  viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
}

// ── Trace ID renderer ────────────────────────────────────────────────────────
function renderTraceId(bubble, traceId) {
  if (!bubble || !traceId) return;
  if (bubble.querySelector('.trace-badge')) return;
  const traceEl = document.createElement('div');
  traceEl.className = 'trace-badge';
  traceEl.style.fontSize = '0.72rem';
  traceEl.style.color = '#718096';
  traceEl.style.marginTop = '6px';
  traceEl.textContent = `Trace: ${traceId.slice(0, 8)}...`;
  traceEl.title = `Distributed Trace ID: ${traceId}`;
  bubble.appendChild(traceEl);
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

// ── SSE Stream Helper ─────────────────────────────────────────────────────────
async function consumeSSEStream(response, onStatus, onToken, onInterrupt, onDone, onError) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop(); // keep trailing partial block

    for (const block of blocks) {
      if (!block.trim()) continue;
      let eventType = 'message';
      let dataStr = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          dataStr = line.slice(6).trim();
        }
      }
      if (!dataStr) continue;
      try {
        const payload = JSON.parse(dataStr);
        const ev = eventType || payload.event;
        if (ev === 'token') {
          onToken(payload.chunk || '');
        } else if (ev === 'status') {
          onStatus(payload.message || '');
        } else if (ev === 'interrupt') {
          onInterrupt(payload);
        } else if (ev === 'done') {
          onDone(payload);
        } else if (ev === 'error') {
          onError(payload.detail || 'Streaming execution failed');
        }
      } catch (err) {
        console.warn('SSE block parse error:', err, dataStr);
      }
    }
  }
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

    const r = await fetch(`${API}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!r.ok) {
      removeTypingIndicator();
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      addMessage('assistant', `⚠️ **Error ${r.status}**: ${err.detail ?? r.statusText}`);
      return;
    }

    let bubble = null;
    let streamedText = '';

    await consumeSSEStream(
      r,
      // onStatus
      (statusMsg) => {
        const typingBubble = document.querySelector('#typing-row .bubble');
        if (typingBubble) {
          typingBubble.innerHTML = `<span style="font-size:0.8rem; opacity:0.85; margin-right:8px;">⚡ ${statusMsg}</span><div class="typing-dots" style="display:inline-flex;"><span></span><span></span><span></span></div>`;
        }
      },
      // onToken
      (chunk) => {
        if (!bubble) {
          removeTypingIndicator();
          bubble = addMessage('assistant', '');
        }
        streamedText += chunk;
        if (window.marked) {
          try {
            bubble.innerHTML = marked.parse(streamedText);
          } catch {
            bubble.textContent = streamedText;
          }
        } else {
          bubble.textContent = streamedText;
        }
        const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
      },
      // onInterrupt
      (data) => {
        removeTypingIndicator();
        setThread(data.thread_id ?? threadId);
        addMessage('assistant', `⏸ **Approval Required**: The agent needs confirmation before proceeding with this write operation.`);
        showApproval(data.approval_request ?? {});
      },
      // onDone
      (data) => {
        removeTypingIndicator();
        setThread(data.thread_id ?? threadId);
        if (!bubble && data.answer) {
          bubble = addMessage('assistant', data.answer);
        } else if (bubble && data.answer) {
          if (window.marked) {
            try {
              bubble.innerHTML = marked.parse(data.answer);
            } catch {
              bubble.textContent = data.answer;
            }
          } else {
            bubble.textContent = data.answer;
          }
        }
        if (data.citations && data.citations.length > 0) {
          renderCitations(bubble, data.citations);
        }
        if (data.trace_id) {
          renderTraceId(bubble, data.trace_id);
        }
        loadHistory();
      },
      // onError
      (detail) => {
        removeTypingIndicator();
        addMessage('assistant', `⚠️ **Error**: ${detail}`);
      }
    );
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
    const r = await fetch(`${API}/chat/resume/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: threadId, decision }),
    });

    hideApproval();

    if (!r.ok) {
      removeTypingIndicator();
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      addMessage('assistant', `⚠️ **Resume Error ${r.status}**: ${err.detail ?? r.statusText}`);
      return;
    }

    let bubble = null;
    let streamedText = '';

    await consumeSSEStream(
      r,
      // onStatus
      (statusMsg) => {
        const typingBubble = document.querySelector('#typing-row .bubble');
        if (typingBubble) {
          typingBubble.innerHTML = `<span style="font-size:0.8rem; opacity:0.85; margin-right:8px;">⚡ ${statusMsg}</span><div class="typing-dots" style="display:inline-flex;"><span></span><span></span><span></span></div>`;
        }
      },
      // onToken
      (chunk) => {
        if (!bubble) {
          removeTypingIndicator();
          bubble = addMessage('assistant', `**[${decision}]** — `);
          streamedText = `**[${decision}]** — `;
        }
        streamedText += chunk;
        if (window.marked) {
          try {
            bubble.innerHTML = marked.parse(streamedText);
          } catch {
            bubble.textContent = streamedText;
          }
        } else {
          bubble.textContent = streamedText;
        }
        const viewport = chatWrap.closest('.chat-viewport') || chatWrap;
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
      },
      // onInterrupt
      (data) => {
        removeTypingIndicator();
        showApproval(data.approval_request ?? {});
      },
      // onDone
      (data) => {
        removeTypingIndicator();
        if (!bubble && data.answer) {
          bubble = addMessage('assistant', `**[${decision}]** — ${data.answer}`);
        } else if (bubble && data.answer) {
          const finalMarkup = `**[${decision}]** — ${data.answer}`;
          if (window.marked) {
            try {
              bubble.innerHTML = marked.parse(finalMarkup);
            } catch {
              bubble.textContent = finalMarkup;
            }
          } else {
            bubble.textContent = finalMarkup;
          }
        }
        if (data.citations && data.citations.length > 0) {
          renderCitations(bubble, data.citations);
        }
        if (data.trace_id) {
          renderTraceId(bubble, data.trace_id);
        }
        loadHistory();
      },
      // onError
      (detail) => {
        removeTypingIndicator();
        addMessage('assistant', `⚠️ **Resume Error**: ${detail}`);
      }
    );
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
loadHistory();
const hashThread = window.location.hash ? window.location.hash.replace('#', '').trim() : null;
const storedThread = localStorage.getItem('activeThread');
const startThread = hashThread || storedThread;
if (startThread) {
  loadThread(startThread);
} else {
  setThread(null);
}
