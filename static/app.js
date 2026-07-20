const askBtn = document.getElementById('askBtn');
const q = document.getElementById('q');
const engineStatus = document.getElementById('engineStatus');
const allowWeb = document.getElementById('allowWeb');
const moduleStatusList = document.getElementById('moduleStatusList');
const newChatBtn = document.getElementById('newChatBtn');
const sidebarListContainer = document.getElementById('sidebarListContainer');
const historyLenInput = document.getElementById('historyLen');
const chatHistory = document.getElementById('chatHistory');
const toggleStatusBtn = document.getElementById('toggleStatusBtn');
const closeStatusBtn = document.getElementById('closeStatusBtn');
const statusModal = document.getElementById('statusModal');
const toggleRuntimeBtn = document.getElementById('toggleRuntimeBtn');
const closeRuntimeBtn = document.getElementById('closeRuntimeBtn');
const runtimeModal = document.getElementById('runtimeModal');
const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
const runtimeMode = document.getElementById('runtimeMode');
const runtimeIndex = document.getElementById('runtimeIndex');
const runtimeStage = document.getElementById('runtimeStage');
const runtimeProvider = document.getElementById('runtimeProvider');

let healthTimer = null;
let engineReady = false;
let sessionId = localStorage.getItem('currentSessionId') || btoa(Math.random().toString()).substr(10, 10);
let sessions = JSON.parse(localStorage.getItem('chatSessions') || '[]');
let sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === '1';
const pendingRequests = new Map();
const SEND_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4 20-7Z"/></svg>';
const PENDING_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>';

const STAGE_LABELS = {
  idle: '空闲',
  booting: '启动中',
  loading_chunks: '加载切块',
  loading_bm25: '加载 BM25',
  loading_chroma: '加载 Chroma',
  initializing_client: '初始化模型',
  retry_wait: '等待重试',
  ready: '已就绪',
};

const MODULE_LABELS = {
  chunks: '切块数据',
  bm25: 'BM25 检索',
  dense: '向量检索',
  rerank: '重排序',
  llm_primary: '主生成通道',
  llm_fallback: '备用生成通道',
  web_fallback: '联网补充',
};

function formatCount(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) return '-';
  return num.toLocaleString('zh-CN');
}

function providerLabel(value) {
  const raw = String(value || '').toLowerCase();
  if (raw === 'bailian') return '百炼';
  if (raw === 'ollama') return 'Ollama';
  if (raw === 'extractive') return '抽取式';
  return value || '-';
}

function statusLabel(value) {
  const raw = String(value || 'unknown');
  const labels = {
    ready: '就绪',
    enabled: '已启用',
    disabled: '已禁用',
    failed: '失败',
    skipped: '已跳过',
    loading: '加载中',
    pending: '等待中',
    unknown: '未知',
  };
  return labels[raw] || raw;
}

function formatTime(ts) {
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return '';
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mi = String(d.getMinutes()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
  } catch (e) {
    return '';
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderInlineMarkdown(value) {
  return String(value)
    .split(/(`[^`]*`)/g)
    .map((part) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
      }
      return escapeHtml(part)
        .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
        .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
        .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    })
    .join('');
}

function renderMarkdown(value) {
  const lines = String(value || '').replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let paragraph = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };

  const flushParagraph = () => {
    if (!paragraph.length) return;
    closeList();
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join('<br>')}</p>`);
    paragraph = [];
  };

  const pushListItem = (type, text) => {
    flushParagraph();
    if (listType && listType !== type) closeList();
    if (!listType) {
      listType = type;
      html.push(`<${type}>`);
    }
    html.push(`<li>${renderInlineMarkdown(text)}</li>`);
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      closeList();
      return;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const ordered = line.match(/^(\d+)[\.\)、]\s+(.+)$/);
    if (ordered) {
      pushListItem('ol', ordered[2]);
      return;
    }

    const unordered = line.match(/^[-*+]\s+(.+)$/);
    if (unordered) {
      pushListItem('ul', unordered[1]);
      return;
    }

    paragraph.push(line);
  });

  flushParagraph();
  closeList();
  return html.join('');
}

function setText(el, text) {
  if (el) el.textContent = text;
}

function setEngineStatus(text, tone = 'neutral') {
  if (!engineStatus) return;
  engineStatus.textContent = text;
  if (toggleRuntimeBtn) {
    toggleRuntimeBtn.className = `status-chip ${tone}`;
  }
}

function updateAskButtonState() {
  if (!askBtn) return;
  if (!engineReady) {
    askBtn.disabled = true;
    askBtn.innerHTML = '...';
    askBtn.title = '知识库加载中';
    return;
  }
  if (pendingRequests.has(sessionId)) {
    askBtn.disabled = true;
    askBtn.innerHTML = PENDING_ICON;
    askBtn.title = '当前会话正在回答';
    return;
  }
  askBtn.disabled = false;
  askBtn.innerHTML = SEND_ICON;
  askBtn.title = '发送';
}

function removePendingMessage(targetSessionId) {
  if (!chatHistory || sessionId !== targetSessionId) return;
  chatHistory.querySelectorAll('.message[data-pending-for]').forEach((el) => {
    if (el.dataset.pendingFor === targetSessionId) el.remove();
  });
}

function applySidebarState() {
  document.body.classList.toggle('sidebar-collapsed', sidebarCollapsed);
  if (toggleSidebarBtn) {
    toggleSidebarBtn.title = sidebarCollapsed ? '展开侧边栏' : '收起侧边栏';
    toggleSidebarBtn.setAttribute('aria-label', toggleSidebarBtn.title);
  }
}

function renderRuntimeSummary(data) {
  const stage = STAGE_LABELS[data.engine_stage] || data.engine_stage || '-';
  const indexText = `${formatCount(data.index_ntotal)} / ${formatCount(data.total_chunks)}`;
  const provider = data.llm_provider_last && data.llm_provider_last !== 'extractive'
    ? providerLabel(data.llm_provider_last)
    : providerLabel(data.llm_primary_provider || 'extractive');

  let mode = 'BM25';
  if (data.vector_enabled) mode = data.last_vec_used ? 'Hybrid' : 'Hybrid 待触发';
  if (data.force_bm25_only) mode = 'BM25 only';

  setText(runtimeMode, mode);
  setText(runtimeIndex, indexText);
  setText(runtimeStage, stage);
  setText(runtimeProvider, provider);
}

function renderModuleStatuses(data) {
  if (!moduleStatusList) return;

  const statuses = data.module_statuses || {};
  const keys = ['chunks', 'bm25', 'dense', 'rerank', 'llm_primary', 'llm_fallback', 'web_fallback'];
  moduleStatusList.innerHTML = keys.map((key) => {
    const raw = String(statuses[key] || 'unknown');
    const rawClass = raw.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase();
    const extra = [];
    if (key === 'bm25') extra.push(`最近命中 ${formatCount(data.last_bm25_hits)} 条`);
    if (key === 'dense') {
      extra.push(`最近命中 ${formatCount(data.last_vec_hits)} 条`);
      extra.push(data.last_vec_used ? '已参与最近一次回答' : '最近未参与');
    }
    if (key === 'rerank') extra.push(data.last_rerank_used ? '最近已生效' : '最近未触发');
    if (key === 'llm_primary') extra.push(`主通道 ${providerLabel(data.llm_primary_provider)}`);
    if (key === 'llm_fallback') extra.push(`最近来源 ${providerLabel(data.llm_provider_last)}`);
    if (key === 'web_fallback') extra.push(data.web_fallback_enabled ? '已开启' : '已关闭');

    return `<li>
      <div class="module-status-left">
        <span class="status-icon ${rawClass}"></span>
        <div>
          <div class="module-label">${MODULE_LABELS[key] || key}</div>
          <div style="font-size:12px;color:#6f6253">${extra.join(' · ')}</div>
        </div>
      </div>
      <span class="module-pill status-${rawClass}">${statusLabel(raw)}</span>
    </li>`;
  }).join('');
}

async function refreshHealth() {
  let nextPollMs = 3000;
  try {
    const resp = await fetch('/api/health?format=json', { headers: { accept: 'application/json' } });
    if (!resp.ok) throw new Error('health request failed');
    const data = await resp.json();

    renderRuntimeSummary(data);
    renderModuleStatuses(data);

    const stage = STAGE_LABELS[data.engine_stage] || data.engine_stage || '未知阶段';
    if (data.engine_ready) {
      engineReady = true;
      updateAskButtonState();
      const tone = data.vector_enabled && data.llm_enabled ? 'ok' : 'degraded';
      setEngineStatus(`已就绪 · 索引 ${formatCount(data.index_ntotal)}`, tone);
      nextPollMs = 60000;
    } else {
      engineReady = false;
      updateAskButtonState();
      if (data.engine_phase === 'retry_wait') {
        setEngineStatus(`${stage} · ${Number(data.engine_retry_in_sec || 0).toFixed(1)}s 后重试`, 'degraded');
      } else {
        const left = Number(data.engine_stage_remaining_sec_p50 || data.engine_stage_remaining_sec || 0).toFixed(1);
        setEngineStatus(`${stage} · 预计剩余 ${left}s`, 'neutral');
      }
      if (data.engine_error) console.error(`后台加载提示: ${data.engine_error}`);
      nextPollMs = 3000;
    }
  } catch (e) {
    engineReady = false;
    updateAskButtonState();
    setEngineStatus('无法获取健康状态', 'failed');
    nextPollMs = 5000;
  } finally {
    if (healthTimer) clearTimeout(healthTimer);
    healthTimer = setTimeout(refreshHealth, nextPollMs);
  }
}

function appendMessage(role, content, refsData, ts, options = {}) {
  if (!chatHistory) return null;
  const wasNearBottom = chatHistory.scrollHeight - chatHistory.scrollTop - chatHistory.clientHeight < 120;

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  if (options.trace) {
    msgDiv._traceData = options.trace;
  }

  const refsHtml = role === 'assistant' && !options.hideRefs
    ? renderRefsHtml(refsData, options.trace || null)
    : '';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  if (options.trustedHtml) {
    contentDiv.innerHTML = `${content}${refsHtml}`;
  } else if (role === 'assistant') {
    contentDiv.classList.add('markdown-body');
    contentDiv.innerHTML = `${renderMarkdown(content)}${refsHtml}`;
  } else {
    contentDiv.innerHTML = `${escapeHtml(content)}${refsHtml}`;
  }

  msgDiv.appendChild(contentDiv);
  chatHistory.appendChild(msgDiv);
  requestAnimationFrame(() => msgDiv.classList.add('show'));
  if (wasNearBottom) {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
  return msgDiv;
}

function explainPublicRefReason(reason) {
  const map = {
    no_refs: '检索阶段没有找到可用候选证据。',
    no_anchor_terms: '当前问题没有提取出稳定锚点，无法可靠展示证据。',
    'primary_anchor+answer_consistent': '已按主实体和答案内容校验后展示证据。',
    'secondary_anchor+answer_consistent': '已按候选锚点和答案内容校验后展示证据。',
    primary_anchor_only: '已按主实体过滤展示证据，但未通过更强的一致性校验。',
    secondary_anchor_only: '已按候选锚点过滤展示证据，但未通过更强的一致性校验。',
    filtered_out: '候选证据与问题主实体或答案内容不够一致，已自动隐藏。',
  };
  return map[String(reason || '')] || '当前没有足够可靠的可展示证据。';
}

function renderRefsHtml(refsData, traceData) {
  if (!refsData || refsData.length === 0) {
    const reason = traceData && traceData.public_ref_reason ? explainPublicRefReason(traceData.public_ref_reason) : '本次答案未找到足够一致的可展示证据。';
    return `<div class="refs-empty" style="margin-top:10px;font-size:13px;opacity:.72;">${escapeHtml(reason)}</div>`;
  }
  return '<ul class="refs-list">' + refsData.map((r) => {
    const u = escapeHtml(r.url || '#');
    const t = escapeHtml(r.title || '无标题');
    return `<li><a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a></li>`;
  }).join('') + '</ul>';
}

function updateAssistantMessage(msgEl, content, refsData, options = {}) {
  if (!msgEl) return;
  if (!options.hideRefs) delete msgEl.dataset.pendingFor;
  const contentDiv = msgEl.querySelector('.message-content');
  if (!contentDiv) return;
  contentDiv.classList.add('markdown-body');
  const traceData = msgEl._traceData || null;
  const refsHtml = options.hideRefs ? '' : renderRefsHtml(refsData, traceData);
  contentDiv.innerHTML = `${renderMarkdown(content || '')}${refsHtml}`;
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updatePendingMessage(msgEl, text) {
  if (!msgEl) return;
  const contentDiv = msgEl.querySelector('.message-content');
  if (!contentDiv) return;
  contentDiv.classList.remove('markdown-body');
  contentDiv.innerHTML = `<span style="opacity:.68;">${escapeHtml(text)}</span>`;
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function loadChatHistory() {
  if (!chatHistory) return;
  const session = sessions.find((s) => s.id === sessionId);
  const hasMessages = Boolean(session && session.messages && session.messages.length);

  chatHistory.innerHTML = hasMessages ? '' : `
    <div class="message system-msg">
      <div class="message-content">
        <span class="welcome-title">基于 Wiki-CN 的 RAG 问答系统</span>
        <span class="welcome-copy">可以提问知识库中的人物、地点、事件和概念。按 Enter 发送，Shift+Enter 换行。</span>
      </div>
    </div>
  `;

  if (session && session.messages) {
    session.messages.forEach((msg) => appendMessage(msg.role, msg.content, msg.refs, msg.ts, { trace: msg.trace || null }));
  }
  if (pendingRequests.has(sessionId)) {
    const pendingMsg = appendMessage('assistant', '<span style="opacity:.68;">正在检索与生成，请稍候...</span>', null, null, { trustedHtml: true, hideRefs: true });
    if (pendingMsg) pendingMsg.dataset.pendingFor = sessionId;
  }
  updateAskButtonState();
}

function updateSessionSidebar() {
  if (!sidebarListContainer) return;

  const todayMs = new Date(new Date().setHours(0, 0, 0, 0)).getTime();
  const yesterdayStart = todayMs - 86400000;
  const weekStart = todayMs - 7 * 86400000;
  const groups = { today: [], yesterday: [], week: [], older: [] };

  sessions.slice().reverse().forEach((s) => {
    const ts = s.timestamp || (s.messages && s.messages.length ? s.messages[s.messages.length - 1].ts : 0) || 0;
    const item = Object.assign({}, s, { last_ts: ts });
    if (ts >= todayMs) groups.today.push(item);
    else if (ts >= yesterdayStart) groups.yesterday.push(item);
    else if (ts >= weekStart) groups.week.push(item);
    else groups.older.push(item);
  });

  let html = '';
  const renderGroup = (title, list) => {
    if (!list.length) return;
    html += `<div class="session-group">
      <div class="session-group-title">${title}</div>
      <ul class="session-list">
        ${list.map((s) => {
          const latest = s.messages && s.messages.length ? s.messages[s.messages.length - 1] : null;
          const snippet = escapeHtml(latest ? String(latest.content || '').slice(0, 48) : (s.title || '新话题'));
          const titleText = escapeHtml(s.title || '新话题');
          const time = s.last_ts ? formatTime(s.last_ts) : '';
          const pending = pendingRequests.has(s.id);
          return `<li class="session-item ${s.id === sessionId ? 'active' : ''} ${pending ? 'pending' : ''}" data-id="${escapeHtml(s.id)}">
            <div class="session-entry">
              <div class="session-title">${titleText}</div>
              <div class="session-snippet">${pending ? '正在回答...' : snippet}</div>
            </div>
            <div class="session-meta">
              <span class="session-time">${time}</span>
              <button class="session-delete" data-id="${escapeHtml(s.id)}" title="删除会话" type="button">×</button>
            </div>
          </li>`;
        }).join('')}
      </ul>
    </div>`;
  };

  renderGroup('今天', groups.today);
  renderGroup('昨天', groups.yesterday);
  renderGroup('7 天内', groups.week);
  renderGroup('更早', groups.older);

  sidebarListContainer.innerHTML = html || '<div class="empty-history">还没有历史会话。开始提问后，这里会自动归档。</div>';

  sidebarListContainer.querySelectorAll('.session-item').forEach((el) => {
    el.addEventListener('click', () => {
      const id = el.getAttribute('data-id');
      if (!id) return;
      sessionId = id;
      localStorage.setItem('currentSessionId', sessionId);
      updateSessionSidebar();
      loadChatHistory();
      updateAskButtonState();
    });
  });

  sidebarListContainer.querySelectorAll('.session-delete').forEach((btn) => {
    btn.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
    });
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      const id = btn.getAttribute('data-id');
      if (!id) return;
      const idx = sessions.findIndex((s) => s.id === id);
      if (idx >= 0) {
        sessions.splice(idx, 1);
        pendingRequests.delete(id);
        if (sessionId === id) {
          sessionId = btoa(Math.random().toString()).substr(10, 10);
          localStorage.setItem('currentSessionId', sessionId);
        }
        localStorage.setItem('chatSessions', JSON.stringify(sessions));
        updateSessionSidebar();
        loadChatHistory();
        updateAskButtonState();
      }
    });
  });
}

async function ask() {
  if (!engineReady) {
    appendMessage('system-msg', '知识库尚未就绪，请稍后再试。');
    return;
  }

  const question = q.value.trim();
  if (!question) return;

  const requestSessionId = sessionId;
  if (pendingRequests.has(requestSessionId)) {
    appendMessage('system-msg', '当前会话正在回答，请稍后，或切换到其他会话继续提问。');
    return;
  }

  if (chatHistory && chatHistory.querySelector('.message.system-msg')) {
    chatHistory.innerHTML = '';
  }

  const nowTs = Date.now();
  appendMessage('user', question, null, nowTs);
  q.value = '';
  q.style.height = 'auto';

  const loadingMsg = appendMessage('assistant', '<span style="opacity:.68;">正在检索证据，请稍候...</span>', null, null, { trustedHtml: true, hideRefs: true });
  if (loadingMsg) {
    loadingMsg.dataset.pendingFor = requestSessionId;
  }
  pendingRequests.set(requestSessionId, true);
  updateAskButtonState();

  try {
    const useWeb = Boolean(allowWeb && allowWeb.checked);
    const historyLen = parseInt(historyLenInput.value, 10) || 5;

    let session = sessions.find((s) => s.id === requestSessionId);
    if (!session) {
      session = { id: requestSessionId, title: question.slice(0, 20) || '新话题', timestamp: nowTs, messages: [] };
      sessions.push(session);
      localStorage.setItem('currentSessionId', requestSessionId);
    }

    session.messages = session.messages || [];
    session.messages.push({ role: 'user', content: question, ts: nowTs });
    session.timestamp = nowTs;
    updateSessionSidebar();

    const resp = await fetch('/api/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, allow_web: useWeb, session_id: requestSessionId, history_len: historyLen }),
    });
    if (!resp.ok) throw new Error('request failed');
    if (!resp.body) throw new Error('empty response body');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let ansText = '';
    let refs = [];
    let finalEvent = null;
    let trace = null;

    const streamingMsg = sessionId === requestSessionId ? loadingMsg : null;
    let startedStreaming = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const event = JSON.parse(trimmed);
        if (event.type === 'chunk') {
          if (!startedStreaming) {
            startedStreaming = true;
          }
          ansText += event.delta || '';
          if (streamingMsg && sessionId === requestSessionId) {
            updateAssistantMessage(streamingMsg, ansText, refs, { hideRefs: true });
          }
        } else if (event.type === 'done') {
          finalEvent = event;
          ansText = event.answer || ansText || '无结果';
          refs = event.references || [];
          trace = event.trace || null;
          if (streamingMsg && sessionId === requestSessionId) {
            streamingMsg._traceData = trace;
            updateAssistantMessage(streamingMsg, ansText, refs);
          }
        } else if (event.type === 'provider_error') {
          console.warn('provider_error', event.provider, event.message);
        } else if (event.type === 'error') {
          throw new Error(event.message || 'request failed');
        }
      }
      if (!startedStreaming && !finalEvent && streamingMsg && sessionId === requestSessionId) {
        updatePendingMessage(streamingMsg, '正在检索证据，请稍候...');
      }
    }

    buffer += decoder.decode();

    if (buffer.trim()) {
      const event = JSON.parse(buffer.trim());
      if (event.type === 'done') {
        finalEvent = event;
        ansText = event.answer || ansText || '无结果';
        refs = event.references || [];
        trace = event.trace || null;
        if (streamingMsg && sessionId === requestSessionId) {
          streamingMsg._traceData = trace;
          updateAssistantMessage(streamingMsg, ansText, refs);
        }
      } else if (event.type === 'chunk') {
        if (!startedStreaming) {
          startedStreaming = true;
        }
        ansText += event.delta || '';
        if (streamingMsg && sessionId === requestSessionId) {
          updateAssistantMessage(streamingMsg, ansText, refs, { hideRefs: true });
        }
      } else if (event.type === 'provider_error') {
        console.warn('provider_error', event.provider, event.message);
      } else if (event.type === 'error') {
        throw new Error(event.message || 'request failed');
      }
    }

    if (!finalEvent && !ansText) {
      throw new Error('empty result');
    }

    const ansTs = Date.now();

    session.messages.push({ role: 'assistant', content: ansText, refs, trace, ts: ansTs });
    session.timestamp = ansTs;
    if (!session.title || session.title === '新话题') session.title = question.slice(0, 20) || '新话题';
    localStorage.setItem('chatSessions', JSON.stringify(sessions));

    if (sessionId === requestSessionId) {
      if (streamingMsg) {
        streamingMsg._traceData = trace;
        updateAssistantMessage(streamingMsg, ansText, refs);
      } else {
        appendMessage('assistant', ansText, refs, ansTs, { trace });
      }
    }

    updateSessionSidebar();
    refreshHealth();
  } catch (e) {
    const errTs = Date.now();
    let session = sessions.find((s) => s.id === requestSessionId);
    if (session) {
      session.messages = session.messages || [];
      session.messages.push({ role: 'system-msg', content: '请求失败，请检查服务日志。', ts: errTs });
      session.timestamp = errTs;
      localStorage.setItem('chatSessions', JSON.stringify(sessions));
    }
    if (sessionId === requestSessionId) {
      removePendingMessage(requestSessionId);
      appendMessage('system-msg', '请求失败，请检查服务日志。');
    }
  } finally {
    pendingRequests.delete(requestSessionId);
    updateSessionSidebar();
    updateAskButtonState();
  }
}

toggleStatusBtn?.addEventListener('click', () => {
  statusModal.style.display = 'flex';
});

closeStatusBtn?.addEventListener('click', () => {
  statusModal.style.display = 'none';
});

statusModal?.addEventListener('click', (e) => {
  if (e.target === statusModal) statusModal.style.display = 'none';
});

toggleRuntimeBtn?.addEventListener('click', () => {
  runtimeModal.style.display = 'flex';
});

closeRuntimeBtn?.addEventListener('click', () => {
  runtimeModal.style.display = 'none';
});

runtimeModal?.addEventListener('click', (e) => {
  if (e.target === runtimeModal) runtimeModal.style.display = 'none';
});

askBtn?.addEventListener('click', ask);

toggleSidebarBtn?.addEventListener('click', () => {
  sidebarCollapsed = !sidebarCollapsed;
  localStorage.setItem('sidebarCollapsed', sidebarCollapsed ? '1' : '0');
  applySidebarState();
});

newChatBtn?.addEventListener('click', () => {
  sessionId = btoa(Math.random().toString()).substr(10, 10);
  localStorage.setItem('currentSessionId', sessionId);
  loadChatHistory();
  updateSessionSidebar();
  updateAskButtonState();
});

q?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

q?.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = `${this.scrollHeight}px`;
});

applySidebarState();
refreshHealth();
updateSessionSidebar();
loadChatHistory();
