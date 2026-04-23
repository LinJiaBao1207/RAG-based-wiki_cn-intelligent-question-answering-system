# Wiki-CN 问答系统骨架（Python + Web）

目标约束：
- 问答范围：当前仅基于 wiki-cn 知识库回答。
- 回答风格：简洁事实型。
- 上线形态：Web（FastAPI + 简单前端）。
- 文本统一：离线处理中统一转简体。
- 数据集地址：https://opendatalab.com/ABear/Wiki_CN
- 前端页面如图所示
 <img width="1906" height="874" alt="3c072380628fad02d103b2888e8e2df6" src="https://github.com/user-attachments/assets/880efb46-662f-4d39-b5d8-9e108b7351dd" />


## 目录

- app: Web 服务与问答引擎
- scripts: 数据清洗、切块、建索引脚本
- templates / static: 前端页面
- build: 产物目录（语料与索引）

## 1. 安装依赖

在 qa_web 目录执行：

```bash
pip install -r requirements.txt
```

## 2. 生成简体语料

在 qa_web 目录执行：

```bash
python scripts/prepare_corpus.py --data-root ../wiki-cn --output ./build/corpus_simplified.jsonl
```

如需先小规模验证（例如 10000 条）：

```bash
python scripts/prepare_corpus.py --data-root ../wiki-cn --output ./build/corpus_simplified.jsonl --limit 10000
```

支持断点续跑（中断后继续）：

```bash
python scripts/prepare_corpus.py --data-root ../wiki-cn --output ./build/corpus_simplified.jsonl --resume
```

说明：
- 默认会在同目录生成 `corpus_simplified.state.json` 作为续跑状态文件。

## 3. 构建索引

```bash
python scripts/build_index.py --corpus ./build/corpus_simplified.jsonl --build-dir ./build --embed-model bge-m3 --base-url http://127.0.0.1:11434/v1
```

可选：启用更稳的切块参数（按句优先拼接 + 最小长度过滤）：

```bash
python scripts/build_index.py --corpus ./build/corpus_simplified.jsonl --build-dir ./build --embed-model bge-m3 --base-url http://127.0.0.1:11434/v1 --chunk-size 550 --chunk-overlap 80 --min-chunk-len 80
```

支持断点续跑（embedding 阶段中断后继续）：

```bash
python scripts/build_index.py --corpus ./build/corpus_simplified.jsonl --build-dir ./build --embed-model bge-m3 --base-url http://127.0.0.1:11434/v1 --resume
```

可选：保留 embedding 缓存，便于调试或复用。

```bash
python scripts/build_index.py --corpus ./build/corpus_simplified.jsonl --build-dir ./build --embed-model bge-m3 --base-url http://127.0.0.1:11434/v1 --resume --keep-emb-cache
```

会生成：
- build/chunks.jsonl
- build/bm25.pkl
- build/faiss.index
- build/meta.pkl

## 4. 启动 Web 服务

大索引建议先用非重载模式（更稳定，避免反复冷启动）：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

开发调试可用热重载：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

说明：
- 当前已启用“后台懒加载”引擎。服务会先启动，索引在后台加载。
- 页面可立即打开；加载期间提问会提示“知识库正在后台加载中”。
- 加载失败会自动重试（指数退避），无需手动重启服务。
- 可通过 `GET /api/health` 观察 `engine_phase`、`engine_stage`、`engine_ready`、`engine_load_elapsed_sec`。

低内存机器建议（跳过向量索引，直接 BM25-only，避免 `std::bad_alloc`）：

```powershell
$env:FORCE_BM25_ONLY="1"
uvicorn app.main:app --host 0.0.0.0 --port 8000

cmd 里，请用：
set FORCE_BM25_ONLY=1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

说明：
- 该模式会跳过 `faiss.index` 加载，启动更稳更快。
- `/api/health` 中 `vector_enabled` 将为 `false`，但系统仍可用（BM25 + 抽取/生成回答）。

浏览器访问：

- http://127.0.0.1:8000
- http://127.0.0.1:8000/api/health

## 5. 统一走 Ollama（Embedding + 生成）

先下载模型：

```bash
ollama pull bge-m3
ollama pull qwen3.5:4b
```

Windows PowerShell 设置环境变量（当前终端生效）：

```bash
$env:OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
$env:OPENAI_EMBED_MODEL="bge-m3"
$env:OPENAI_MODEL="qwen3.5:4b"
```

然后启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

说明：
- Embedding 与生成都通过 Ollama 的 OpenAI 兼容接口调用。
- 如果你不启用 Ollama，检索的向量部分将不可用。

### 5.1 生成模型：阿里云百炼优先，失败自动切本地 Ollama

当前版本已支持：
- 先调用阿里云百炼的生成模型（`BAILIAN_MODEL`）。
- 若百炼不可用（如 key 无效、网络失败、限流），自动回退到本地 Ollama（`OPENAI_MODEL`）。

PowerShell（示例）：

```powershell
$env:OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
$env:OPENAI_MODEL="qwen3.5:4b"
$env:OPENAI_EMBED_MODEL="bge-m3"

$env:LLM_PRIMARY_PROVIDER="bailian"
$env:ENABLE_LLM_FALLBACK="1"
$env:BAILIAN_API_KEY="<your-bailian-api-key>"
$env:BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:BAILIAN_MODEL="qwen-plus"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

cmd（示例）：

```cmd
set OPENAI_BASE_URL=http://127.0.0.1:11434/v1
set OPENAI_MODEL=qwen3.5:4b
set OPENAI_EMBED_MODEL=bge-m3

set LLM_PRIMARY_PROVIDER=bailian
set ENABLE_LLM_FALLBACK=1
set BAILIAN_API_KEY=<your-bailian-api-key>
set BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set BAILIAN_MODEL=qwen-plus

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

说明：
- 可选主通道：`LLM_PRIMARY_PROVIDER=bailian|ollama`。
- 若不希望回退，可设置 `ENABLE_LLM_FALLBACK=0`。
- `/api/health` 可观察 `llm_fallback_enabled` 与 `llm_provider_last`。

### 使用 Zilliz Cloud 承载稠密索引（推荐低内存本机）

为什么上传 `embeddings.f32`，而不是 `faiss.index`：
- `faiss.index` 是本地 FAISS 引擎的序列化格式，不能直接作为 Zilliz Collection 的可查询数据导入。
- Zilliz 需要的是「向量 + 主键」数据，因此应上传 `embeddings.f32`（或重新计算向量）并以 `id -> chunk` 对齐。

本地连接 Zilliz 的环境变量（PowerShell）：

```powershell
$env:DENSE_BACKEND="zilliz"
$env:ZILLIZ_URI="https://<your-endpoint>"
$env:ZILLIZ_TOKEN="<token>"
$env:ZILLIZ_COLLECTION="wiki_cn_dense"
$env:ZILLIZ_NPROBE="16"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

cmd：

```cmd
set DENSE_BACKEND=zilliz
set ZILLIZ_URI=https://<your-endpoint>
set ZILLIZ_TOKEN=<token>
set ZILLIZ_COLLECTION=wiki_cn_dense
set ZILLIZ_NPROBE=16
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

说明：
- 本地服务仍使用 BM25 与引用回填；稠密检索改为远程 Zilliz 查询。
- 建议在 Zilliz 中以 `id(INT64)` 存 chunk 序号，和本地 `chunks.jsonl` 行号一一对应。

稳定性参数（可选，PowerShell 当前终端生效）：

```bash
$env:EMBED_TIMEOUT_SECONDS="8"
$env:EMBED_FAILURE_THRESHOLD="2"
$env:EMBED_COOLDOWN_SECONDS="60"
$env:LLM_TIMEOUT_SECONDS="45"
$env:LLM_FAILURE_THRESHOLD="2"
$env:LLM_COOLDOWN_SECONDS="90"
```

说明：
- Embedding 或 LLM 连续失败达到阈值后，会短时间熔断并自动降级，避免接口长时间卡住。
- 熔断期间仍可返回结果（优先 BM25 + 抽取式答案）。

## 6. 开启本地重排序（Reranker）

系统已经集成了本地的 `bge-reranker-v2-m3` 模型作为重排序阶段。默认由于性能考虑未默认打开，你可以通过以下环境变量将其一键开启，且完全**不需要连网或依赖外部 API**。

开启方式（PowerShell，在启动 uvicorn 前执行）：

```powershell
$env:ENABLE_RERANKER="1"
$env:RERANK_TOP_N="40"
```

开启方式（cmd）：

```cmd
set ENABLE_RERANKER=1
set RERANK_TOP_N=40
```

说明：
- 开启后，检索链路会自动在“向量/BM25混合召回（Top 40）”之后，利用 `models/bge-reranker-v2-m3` 对这 40 个片段进行高精度语义打分。
- `/api/health` 以及前端的“模块状态”面板将亮起 `(生效中)` 的提示。
- Reranker 不会生成任何索引文件，它是在你提问期间实时运行打分。

## 6.1【完整版】当前最新系统启动命令总结

假设你的文件夹结构已经是这样，并且 `models/` 下放好了模型：
- `models/bge-m3`
- `models/bge-reranker-v2-m3`

请从上到下执行以下命令启动完整的最高性能问答系统：

**PowerShell 终端下执行：**

```powershell
# 1. 激活环境并进入目录
conda activate ai_course
cd d:\项目\Wiki_CN\qa_web

# 2. 开启重排序模块
$env:ENABLE_RERANKER="1"

# （可选）配置生成大模型，如果你想用本地的 Ollama qwen 请配置：
# $env:LLM_PRIMARY_PROVIDER="ollama"
# $env:OPENAI_MODEL="qwen3.5:4b"

# 3. 启动后台服务器
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**命令提示符 (CMD) 终端下执行：**

```cmd
:: 1. 激活环境并进入目录
call conda activate ai_course
cd /d d:\项目\Wiki_CN\qa_web

:: 2. 开启重排序模块
set ENABLE_RERANKER=1

:: （可选）配置生成大模型，如果你想用本地的 Ollama qwen 请配置：
:: set LLM_PRIMARY_PROVIDER=ollama
:: set OPENAI_MODEL=qwen3.5:4b

:: 3. 启动后台服务器
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

然后，直接打开浏览器访问 `http://127.0.0.1:8000` 即可开始体验带有智能重排功能的系统！

## 7. 预埋多跳检索（默认关闭）

当前代码已预留二跳检索，默认关闭，不影响现有结果。

开启方式（PowerShell 当前终端生效）：

```bash
$env:ENABLE_MULTI_HOP="1"
$env:MULTI_HOP_MAX_HOPS="2"
$env:MULTI_HOP_EXPAND_TITLES="2"
```

说明：
- 第 1 跳先做常规混合召回。
- 第 2 跳会基于第 1 跳的高分标题扩展查询，再召回并合并结果。
- 仅修改在线检索逻辑，不需要重跑清洗与构建。

## 8. 下一步建议

- 增加评测脚本（正确率、幻觉率、引用一致性）
- 增加缓存与日志（便于上线）
- 完善后再开放“联网补充”模式（默认关闭）

## 8.1 联网补充（默认关闭）

当本地知识库证据不足时，可自动走联网补充检索（DuckDuckGo API），用于兜底回答。

PowerShell：

```powershell
$env:ALLOW_WEB_FALLBACK="1"
$env:WEB_SEARCH_TIMEOUT_SECONDS="6"
$env:WEB_SEARCH_MAX_RESULTS="3"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

cmd：

```cmd
set ALLOW_WEB_FALLBACK=1
set WEB_SEARCH_TIMEOUT_SECONDS=6
set WEB_SEARCH_MAX_RESULTS=3
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

说明：
- 默认仍优先使用本地 wiki-cn 证据，不会替代本地检索链路。
- 只有在本地证据不足时才触发联网检索。
- 可在 `/api/health` 查看 `web_fallback_enabled` 是否已开启。
- 前端页面新增“联网补充（本次提问）”勾选框，可按次开启联网，无需重启服务。

## 9. 健康检查接口

新增接口：

```bash
GET /api/health
```

返回字段说明：
- status: `ok` | `degraded` | `bm25-only`
- client_ready: 是否可访问 Ollama OpenAI 兼容接口
- vector_enabled: 向量检索当前是否可用
- dense_backend: 稠密检索后端（`faiss` | `zilliz` | `none`）
- force_bm25_only: 是否命中 `FORCE_BM25_ONLY=1`
- dense_ready: 稠密索引当前是否可用（已加载/已连通）
- dense_disabled_reason: 稠密不可用原因（例如 `FORCE_BM25_ONLY=1`、`DENSE_BACKEND=none`、`std::bad_alloc`）
- llm_enabled: 生成当前是否可用
- llm_primary_provider: 当前配置的生成主通道（`bailian` | `ollama`）
- llm_fallback_enabled: 是否启用并配置了百炼回退通道
- llm_provider_last: 最近一次回答使用的通道（`ollama` | `bailian` | `extractive`）
- web_fallback_enabled: 是否启用联网补充检索
- vector_cooldown_left_sec: 向量熔断剩余秒数
- llm_cooldown_left_sec: 生成熔断剩余秒数
- total_chunks / index_ntotal: 当前加载的切块数与索引条数
- last_bm25_hits / last_vec_hits / last_vec_used: 最近一次检索中 BM25 命中数、向量命中数、向量是否参与
- engine_phase: 引擎主状态（`idle` | `loading` | `retry_wait` | `ready`）
- engine_stage: 引擎加载阶段（`loading_chunks` | `loading_bm25` | `loading_faiss` | `initializing_client` | `ready`）
- engine_retry_count / engine_retry_in_sec: 自动重试次数与下次重试倒计时（秒）
- engine_stage_elapsed_sec: 当前阶段已持续时长（秒）
- engine_stage_durations: 各阶段累计耗时（秒，字典）
- engine_stage_estimates_sec: 各阶段预计耗时（秒，字典）
- engine_stage_remaining_sec: 当前阶段预计剩余时长（秒）
- engine_total_estimated_sec / engine_total_remaining_sec: 启动总预计时长与总剩余时长（秒）
- engine_prediction_confidence: ETA 置信度（`low` | `medium` | `high`）
- engine_stage_remaining_sec_p50 / engine_stage_remaining_sec_p90: 当前阶段剩余时长区间预测（秒）
- engine_total_remaining_sec_p50 / engine_total_remaining_sec_p90: 总剩余时长区间预测（秒）
