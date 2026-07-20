const statusBadge = document.getElementById('statusBadge');
const retrievalMode = document.getElementById('retrievalMode');
const retrievalModeHint = document.getElementById('retrievalModeHint');
const generationMode = document.getElementById('generationMode');
const generationModeHint = document.getElementById('generationModeHint');
const indexScale = document.getElementById('indexScale');
const indexScaleHint = document.getElementById('indexScaleHint');
const engineStage = document.getElementById('engineStage');
const engineStageHint = document.getElementById('engineStageHint');
const moduleMatrix = document.getElementById('moduleMatrix');
const backendFacts = document.getElementById('backendFacts');
const stageTimeline = document.getElementById('stageTimeline');
const diagnostics = document.getElementById('diagnostics');
const rawJson = document.getElementById('rawJson');
const refreshBtn = document.getElementById('refreshBtn');

const MODULE_META = {
  chunks: { name: '切块数据', desc: '知识库文本切块是否已经加载到内存，决定后续引用与片段映射是否可用。' },
  bm25: { name: 'BM25 关键词检索', desc: '稀疏检索通道，主要负责基于关键词、字面匹配和标题词命中召回内容。' },
  dense: { name: '稠密向量检索', desc: '向量检索通道，决定是否真的用到了 bge-m3 + Chroma 的语义召回能力。' },
  rerank: { name: '重排序模块', desc: '对初筛结果再排序，提升最终上下文质量。命中数不代表一定触发，是否生效要看 last_rerank_used。' },
  llm_primary: { name: '主生成通道', desc: '当前优先使用的大模型生成通道。你的配置里通常是百炼。' },
  llm_fallback: { name: '回退生成通道', desc: '主生成通道不可用时是否还能自动切到备用模型或抽取式回答。' },
  web_fallback: { name: '联网补充', desc: '默认在本地证据不足时使用；首页勾选后会按次主动尝试联网搜索。' },
};

const STAGE_LABELS = {
  idle: '空闲',
  booting: '启动中',
  loading_chunks: '加载切块数据',
  loading_bm25: '加载 BM25 索引',
  loading_chroma: '加载 Chroma 向量索引',
  initializing_client: '初始化模型客户端',
  retry_wait: '等待自动重试',
  ready: '已完成',
};

function safe(value, fallback = '-') {
  if (value === null || value === undefined || value === '') return fallback;
  return value;
}

function formatSeconds(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num)) return '-';
  return `${num.toFixed(num >= 10 ? 1 : 2)} 秒`;
}

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
  return safe(value);
}

function statusLabel(value) {
  const raw = String(value || 'unknown');
  const labels = {
    ok: '正常',
    degraded: '降级',
    failed: '失败',
    loading: '加载中',
    starting: '启动中',
    ready: '就绪',
    enabled: '已启用',
    disabled: '已禁用',
    skipped: '已跳过',
    pending: '等待中',
    unknown: '未知',
  };
  return labels[raw] || raw;
}

function setBadge(status) {
  const cls = status === 'ok' ? 'ok' : (status === 'degraded' ? 'degraded' : 'failed');
  statusBadge.className = `badge ${cls}`;
  statusBadge.textContent = statusLabel(status);
}

function getRetrievalMode(data) {
  if (data.force_bm25_only) {
    return ['BM25 only', '已强制禁用向量检索，只会走关键词召回。'];
  }
  if (data.vector_enabled && data.last_vec_used) {
    return ['Hybrid 检索', `BM25 + Chroma 向量同时参与，最近一次向量命中 ${safe(data.last_vec_hits, 0)} 条。`];
  }
  if (data.vector_enabled) {
    return ['Hybrid 待触发', '向量通道可用，但最近一次请求没有实际拿到向量结果。'];
  }
  return ['BM25 主导', data.dense_disabled_reason || '向量通道当前不可用，所以系统正在退化为稀疏检索模式。'];
}

function getGenerationMode(data) {
  if (data.llm_provider_last && data.llm_provider_last !== 'extractive') {
    return [`${providerLabel(data.llm_provider_last)} 生成`, `最近一次回答由 ${providerLabel(data.llm_provider_last)} 负责生成。主通道配置为 ${providerLabel(data.llm_primary_provider)}。`];
  }
  if (data.llm_enabled) {
    return ['抽取式回退', 'LLM 通道存在，但最近一次回答最终使用了 extractive 方式。'];
  }
  return ['抽取式回答', '生成通道当前不可用，只能依赖本地证据抽取答案。'];
}

function renderModules(data) {
  const statuses = data.module_statuses || {};
  const keys = ['chunks', 'bm25', 'dense', 'rerank', 'llm_primary', 'llm_fallback', 'web_fallback'];
  moduleMatrix.innerHTML = keys.map((key) => {
    const meta = MODULE_META[key];
    const raw = String(statuses[key] || 'unknown');
    const extra = [];
    if (key === 'bm25') extra.push(`最近命中: ${safe(data.last_bm25_hits, 0)} 条`);
    if (key === 'dense') extra.push(`最近命中: ${safe(data.last_vec_hits, 0)} 条`, `最近使用: ${data.last_vec_used ? '是' : '否'}`);
    if (key === 'rerank') extra.push(`最近生效: ${data.last_rerank_used ? '是' : '否'}`);
    if (key === 'llm_primary') extra.push(`当前主提供方: ${providerLabel(data.llm_primary_provider)}`);
    if (key === 'llm_fallback') extra.push(`最近回答来源: ${providerLabel(data.llm_provider_last)}`);
    if (key === 'web_fallback') extra.push(`是否启用: ${data.web_fallback_enabled ? '是' : '否'}`);
    return `
      <article class="module-card">
        <div class="module-row">
          <div class="module-name">${meta.name}</div>
          <div class="module-state state-${raw}">${statusLabel(raw)}</div>
        </div>
        <div class="module-desc">${meta.desc}</div>
        <div class="module-desc">${extra.join(' · ')}</div>
      </article>
    `;
  }).join('');
}

function renderFacts(data) {
  const facts = [
    ['稠密后端', safe(data.dense_backend)],
    ['Chroma 路径', `<span class="mono">${safe(data.chroma_persist_dir)}</span>`],
    ['Collection 名称', `<span class="mono">${safe(data.chroma_collection)}</span>`],
    ['主 LLM 提供方', providerLabel(data.llm_primary_provider)],
    ['最近回答来源', providerLabel(data.llm_provider_last)],
    ['向量通道可用', data.vector_enabled ? '是' : '否'],
    ['稠密索引已就绪', data.dense_ready ? '是' : '否'],
    ['索引计数校验', data.dense_index_count_ok ? '通过' : '未通过'],
    ['联网补充', data.web_fallback_enabled ? '开启' : '关闭'],
  ];
  backendFacts.innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function renderStages(data) {
  const durations = data.engine_stage_durations || {};
  const estimates = data.engine_stage_estimates_sec || {};
  const keys = ['loading_chunks', 'loading_bm25', 'loading_chroma', 'initializing_client'];
  stageTimeline.innerHTML = keys.map((key) => {
    const actual = Number(durations[key] || 0);
    const estimate = Number(estimates[key] || 0);
    const current = data.engine_stage === key && data.engine_loading;
    const ratio = estimate > 0 ? Math.min(100, Math.max(8, (actual / estimate) * 100)) : (actual > 0 ? 100 : 10);
    return `
      <div class="stage-item">
        <div class="stage-item-top">
          <div>
            <div class="stage-name">${STAGE_LABELS[key] || key}</div>
            <div class="stage-meta">${current ? '当前正在这个阶段' : '阶段耗时记录'}</div>
          </div>
          <div class="stage-meta">实际 ${formatSeconds(actual)} / 估计 ${formatSeconds(estimate)}</div>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width:${ratio}%"></div>
        </div>
      </div>
    `;
  }).join('');
}

function renderDiagnostics(data) {
  const items = [];
  if (!data.vector_enabled) {
    items.push({
      cls: 'bad',
      text: `当前没有使用向量检索。原因：${safe(data.dense_disabled_reason, '未给出具体原因')}。这通常意味着系统已经退化为 BM25 主导模式。`,
    });
  } else {
    items.push({
      cls: '',
      text: '向量通道当前可用，系统具备使用 bge-m3 + Chroma 的混合检索能力。',
    });
  }

  if (data.index_ntotal === 0) {
    items.push({
      cls: 'bad',
      text: '索引条数是 0。这通常说明 Chroma collection 能连接，但里面没有被当前服务识别到的可检索向量数据。',
    });
  } else {
    items.push({
      cls: '',
      text: `当前索引可见条数为 ${formatCount(data.index_ntotal)}，与切块规模 ${formatCount(data.total_chunks)} 对比可以判断索引是否完整。`,
    });
  }

  if (data.llm_provider_last === 'extractive') {
    items.push({
      cls: 'warn',
      text: '最近一次回答来源是 extractive，说明最近一次回答没有真正由大模型生成，或者主模型链路被回退了。',
    });
  } else {
    items.push({
      cls: '',
      text: `最近一次回答来源是 ${providerLabel(data.llm_provider_last)}，说明生成链路最近一次实际走到了这个通道。`,
    });
  }

  items.push({
    cls: data.web_fallback_enabled ? '' : 'warn',
    text: data.web_fallback_enabled
      ? '联网补充已开启。本地证据不足时可自动补充；首页勾选“联网补充”时也会主动尝试外部搜索。'
      : '联网补充已关闭，当前所有回答都严格依赖本地知识库与本地/配置好的模型链路。',
  });

  diagnostics.innerHTML = items.map((item) => `<div class="diagnostic ${item.cls}">${item.text}</div>`).join('');
}

function renderSummary(data) {
  setBadge(safe(data.status, 'unknown'));
  const [rMode, rHint] = getRetrievalMode(data);
  const [gMode, gHint] = getGenerationMode(data);
  retrievalMode.textContent = rMode;
  retrievalModeHint.textContent = rHint;
  generationMode.textContent = gMode;
  generationModeHint.textContent = gHint;
  indexScale.textContent = `${formatCount(data.index_ntotal)} / ${formatCount(data.total_chunks)}`;
  indexScaleHint.textContent = '前者是当前服务实际看到的索引条数，后者是知识库切块总数。';
  engineStage.textContent = STAGE_LABELS[data.engine_stage] || safe(data.engine_stage);
  engineStageHint.textContent = data.engine_ready
    ? `引擎已就绪，累计加载耗时 ${formatSeconds(data.engine_load_elapsed_sec)}`
    : `阶段剩余 ${formatSeconds(data.engine_stage_remaining_sec_p50 || data.engine_stage_remaining_sec)}`;
}

async function refreshHealth() {
  const resp = await fetch('/api/health?format=json', { headers: { accept: 'application/json' } });
  if (!resp.ok) {
    throw new Error('health request failed');
  }
  const data = await resp.json();
  renderSummary(data);
  renderModules(data);
  renderFacts(data);
  renderStages(data);
  renderDiagnostics(data);
  rawJson.textContent = JSON.stringify(data, null, 2);
}

refreshBtn?.addEventListener('click', () => {
  refreshHealth().catch((err) => {
    rawJson.textContent = `Health load failed: ${String(err)}`;
  });
});

refreshHealth().catch((err) => {
  rawJson.textContent = `Health load failed: ${String(err)}`;
});
