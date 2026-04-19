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

let healthTimer = null;
let engineReady = false;
let sessionId = localStorage.getItem('currentSessionId') || btoa(Math.random().toString()).substr(10, 10);
let sessions = JSON.parse(localStorage.getItem('chatSessions') || '[]');

if (toggleStatusBtn) {
  toggleStatusBtn.addEventListener('click', () => {
    statusModal.style.display = 'flex';
  });
}
if (closeStatusBtn) {
  closeStatusBtn.addEventListener('click', () => {
    statusModal.style.display = 'none';
  });
}

const STAGE_LABELS = {
  idle: '空闲',
  booting: '启动中',
  loading_chunks: '加载切块数据',
  loading_bm25: '加载 BM25 索引',
  loading_faiss: '加载向量索引',
  initializing_client: '初始化模型客户端',
  retry_wait: '等待自动重试',
  ready: '已就绪',
};

function setEngineStatusText(text) {
  if (engineStatus) {
    engineStatus.textContent = text;
  }
}

function renderModuleStatuses(data) {
  if (!moduleStatusList) {
    return;
  }

  const labels = {
    chunks: '切块数据',
    bm25: 'BM25 索引',
    dense: '稠密检索',
    rerank: '重排序',
    llm_primary: '主生成通道',
    llm_fallback: '回退生成通道',
    web_fallback: '联网补充',
  };

  const statuses = data.module_statuses || {};
  const keys = ['chunks', 'bm25', 'dense', 'rerank', 'llm_primary', 'llm_fallback', 'web_fallback'];
    moduleStatusList.innerHTML = keys.map((k) => {
    const raw = String(statuses[k] || 'unknown');
    const cls = `status-${raw.replace(/[^a-z-]/g, '')}`;
    const label = labels[k] || k;
    let extra = '';
    if (k === 'bm25' && data.last_bm25_hits !== undefined) {
      extra = ` <span style="font-size: 0.8em; color: #666;">命中了 ${data.last_bm25_hits}</span>`;
    }
    if (k === 'dense' && data.last_vec_hits !== undefined) {
      extra = ` <span style="font-size: 0.8em; color: #666;">命中 ${data.last_vec_hits} ${data.last_vec_used ? '(已使用)' : '(未启用)'}</span>`;
    }
    if (k === 'rerank' && data.last_rerank_used !== undefined) {
      extra = ` <span style="font-size: 0.8em; color: #666;">${data.last_rerank_used ? '(生效中)' : '(当前未触发)'}</span>`;
    }
    const rawClass = String(raw).replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase();
    return `<li>
      <div class="module-status-left">
        <span class="status-icon ${rawClass}"></span>
        <div>
          <div class="module-label">${label}</div>
          <div style="font-size:12px;color:#6b7280">${extra}</div>
        </div>
      </div>
      <span class="module-pill ${cls}">${raw}</span>
    </li>`;
  }).join('');
}

async function refreshHealth() {
  let nextPollMs = 3000;
  try {
    const resp = await fetch('/api/health');
    if (!resp.ok) {
      throw new Error('health 请求失败');
    }
    const data = await resp.json();
    renderModuleStatuses(data);
    const stage = STAGE_LABELS[data.engine_stage] || data.engine_stage || '未知阶段';

    if (data.engine_ready) {
      engineReady = true;
      askBtn.disabled = false;
      askBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
      const webMode = data.web_fallback_enabled ? '已开启（默认）' : '默认关闭（可用按钮按次开启）';
      setEngineStatusText(`已就绪（切块 ${data.total_chunks}，索引 ${data.index_ntotal}）`);
      nextPollMs = 60000;
    } else {
      engineReady = false;
      askBtn.disabled = true;

      if (data.engine_phase === 'retry_wait') {
        askBtn.innerHTML = '...';
        const retryIn = Number(data.engine_retry_in_sec || 0).toFixed(1);
        setEngineStatusText(`${stage}，重试倒计时: ${retryIn}s`);
      } else {
        askBtn.innerHTML = '...';
        const stageP50 = Number(data.engine_stage_remaining_sec_p50 || data.engine_stage_remaining_sec || 0).toFixed(1);
        setEngineStatusText(`${stage}，预计剩余: ${stageP50}s`);
      }

    if (data.engine_error) {
        // Just silent console log or keep it in status, don't overlay answer area
        console.error(`后台加载提示：${data.engine_error}`);
      }

      nextPollMs = 3000;
    }
  } catch (e) {
    engineReady = false;
    askBtn.disabled = true;
    askBtn.innerHTML = '...';
    setEngineStatusText('无法获取健康状态');
    nextPollMs = 5000;
  } finally {
    if (healthTimer) {
      clearTimeout(healthTimer);
    }
    healthTimer = setTimeout(refreshHealth, nextPollMs);
  }
}

function formatTime(ts) {
  try {
    const d = new Date(ts);
    return d.getHours().toString().padStart(2, '0') + ':' + d.getMinutes().toString().padStart(2, '0');
  } catch (e) {
    return '';
  }
}

function appendMessage(role, content, refsData, ts) {
  if (!chatHistory) return;

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  
  let refsHtml = '';
  if (refsData && refsData.length > 0) {
    refsHtml = `<ul class="refs-list">` + refsData.map((r) => {
      const u = r.url || '#';
      const t = r.title || '无标题';
      return `<li><a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a></li>`;
    }).join('') + `</ul>`;
  }
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  let metaHtml = '';
  if (ts) {
    metaHtml = `<div class="message-meta"><span class="message-timestamp">${formatTime(ts)}</span></div>`;
  }
  contentDiv.innerHTML = `${content}${refsHtml}${metaHtml}`;
  
  msgDiv.appendChild(contentDiv);
  chatHistory.appendChild(msgDiv);
  // small animation: add show class in next frame
  requestAnimationFrame(() => {
    msgDiv.classList.add('show');
    contentDiv.style.opacity = '1';
  });
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return msgDiv;
}

function loadChatHistory() {
  if (!chatHistory) return;
  chatHistory.innerHTML = `
    <div class="message system-msg">
      <div class="message-content" style="font-size: 24px; color: #1f2937; font-weight: 500;">基于Wiki-CN 的 RAG 问答系统</div>
      <div style="font-size: 14px; color: #6b7280; margin-top: 8px;">您可以问我关于知识库中的任何问题，按 Shift+Enter 换行。</div>
    </div>
  `;
  const session = sessions.find(s => s.id === sessionId);
  if (session && session.messages) {
    session.messages.forEach(msg => {
      appendMessage(msg.role, msg.content, msg.refs, msg.ts);
    });
  }
}

function updateSessionSidebar() {
  if (!sidebarListContainer) return;

  const now = Date.now();
  const todayStart = new Date(new Date().setHours(0,0,0,0));
  const todayMs = todayStart.getTime ? todayStart.getTime() : (new Date(new Date().setHours(0,0,0,0))).getTime();
  const yesterdayStart = todayMs - 86400000;
  const weekStart = todayMs - 7 * 86400000;

  const groups = { today: [], yesterday: [], week: [], older: [] };
  sessions.slice().reverse().forEach(s => {
    const ts = s.timestamp || (s.messages && s.messages.length ? s.messages[s.messages.length-1].ts : 0) || 0;
    if (ts >= todayMs) groups.today.push(Object.assign({}, s, { last_ts: ts }));
    else if (ts >= yesterdayStart) groups.yesterday.push(Object.assign({}, s, { last_ts: ts }));
    else if (ts >= weekStart) groups.week.push(Object.assign({}, s, { last_ts: ts }));
    else groups.older.push(Object.assign({}, s, { last_ts: ts }));
  });

  let html = '';
  const renderGroup = (title, list) => {
    if (!list.length) return;
    html += `<div class="session-group">
      <div class="session-group-title">${title}</div>
      <ul class="session-list">
        ${list.map(s => {
          const snippet = s.messages && s.messages.length ? (s.messages[s.messages.length-1].content || '').slice(0, 48) : (s.title || '新话题');
          const time = s.last_ts ? formatTime(s.last_ts) : '';
          return `<li class="session-item ${s.id === sessionId ? 'active' : ''}" data-id="${s.id}">
            <div class="session-entry">
              <div class="session-title">${s.title || '新话题'}</div>
              <div class="session-snippet">${snippet}</div>
            </div>
            <div class="session-meta">
              <span class="session-time">${time}</span>
              <button class="session-delete" data-id="${s.id}" title="删除会话">×</button>
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

  sidebarListContainer.innerHTML = html || '<div style="padding:12px;color:#6b7280">请点击“开启新对话”开始。</div>';

  // Attach events
  sidebarListContainer.querySelectorAll('.session-item').forEach(el => {
    el.addEventListener('click', (e) => {
      const id = el.getAttribute('data-id');
      if (!id) return;
      sessionId = id;
      localStorage.setItem('currentSessionId', sessionId);
      updateSessionSidebar();
      loadChatHistory();
    });
  });

  sidebarListContainer.querySelectorAll('.session-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.getAttribute('data-id');
      if (!id) return;
      const idx = sessions.findIndex(s => s.id === id);
      if (idx >= 0) {
        sessions.splice(idx, 1);
        if (sessionId === id) {
          sessionId = btoa(Math.random().toString()).substr(10, 10);
          localStorage.setItem('currentSessionId', sessionId);
        }
        localStorage.setItem('chatSessions', JSON.stringify(sessions));
        updateSessionSidebar();
        loadChatHistory();
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

  askBtn.disabled = true;
  askBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>';

  const nowTs = Date.now();
  appendMessage('user', question, null, nowTs);
  q.value = '';
  q.style.height = 'auto'; // reset textarea ht

  const loadingMsg = appendMessage('assistant', '<span style="opacity:0.5;">正在检索与生成，请稍候...</span>');

  try {
    const useWeb = Boolean(allowWeb && allowWeb.checked);
    const historyLen = parseInt(historyLenInput.value, 10) || 5;

    let session = sessions.find(s => s.id === sessionId);
    if (!session) {
      session = { id: sessionId, title: question.slice(0, 20) || '新话题', timestamp: nowTs, messages: [] };
      sessions.push(session);
      localStorage.setItem('chatSessions', JSON.stringify(sessions));
      localStorage.setItem('currentSessionId', sessionId);
      updateSessionSidebar();
    }
    
    session.messages = session.messages || [];
    session.messages.push({ role: 'user', content: question, ts: nowTs });

    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, allow_web: useWeb, session_id: sessionId, history_len: historyLen }),
    });
    const data = await resp.json();
    
    loadingMsg.remove();

    const ansText = data.answer || '无结果';
    const items = data.references || [];
    const ansTs = Date.now();
    appendMessage('assistant', ansText, items, ansTs);

    session.messages.push({ role: 'assistant', content: ansText, refs: items, ts: ansTs });
    // update session timestamp/title
    session.timestamp = ansTs;
    if (!session.title || session.title === '新话题') session.title = question.slice(0, 20) || '新话题';
    localStorage.setItem('chatSessions', JSON.stringify(sessions));

  } catch (e) {
    loadingMsg.remove();
    appendMessage('system-msg', '请求失败，请检查服务日志。');
  } finally {
    if (engineReady) {
      askBtn.disabled = false;
      askBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
    }
  }
}

askBtn.addEventListener('click', ask);
if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    sessionId = btoa(Math.random().toString()).substr(10, 10);
    localStorage.setItem('currentSessionId', sessionId);
    loadChatHistory();
    updateSessionSidebar();
  });
}
const newChatSmall = document.getElementById('newChatSmall');
if (newChatSmall) {
  newChatSmall.addEventListener('click', () => {
    sessionId = btoa(Math.random().toString()).substr(10, 10);
    localStorage.setItem('currentSessionId', sessionId);
    loadChatHistory();
    updateSessionSidebar();
  });
}
q.addEventListener('keydown', (e) => {
  // Enter to send, Shift+Enter for newline
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
q.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
});

refreshHealth();
updateSessionSidebar();
loadChatHistory();
