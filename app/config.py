from pathlib import Path
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "wiki-cn"
BUILD_DIR = PROJECT_ROOT / "build"


def _resolve_env_path(value: str, *, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_env_path_str(value: str, *, base_dir: Path = PROJECT_ROOT) -> str:
    if not value:
        return ""
    return str(_resolve_env_path(value, base_dir=base_dir))

CORPUS_PATH = BUILD_DIR / "corpus_simplified.jsonl"
CHUNKS_PATH = BUILD_DIR / "chunks.jsonl"
BM25_PATH = BUILD_DIR / "bm25.pkl"
META_PATH = BUILD_DIR / "meta.pkl"
CHROMA_PERSIST_DIR = _resolve_env_path(os.getenv("CHROMA_PERSIST_DIR", str(BUILD_DIR / "chroma_db")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "wiki_cn_dense")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "20"))
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "20"))
TOP_K_MERGE = int(os.getenv("TOP_K_MERGE", "40"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "8"))
GEN_CONTEXT_TOP_N = int(os.getenv("GEN_CONTEXT_TOP_N", "4"))
GEN_CONTEXT_CHARS = int(os.getenv("GEN_CONTEXT_CHARS", "420"))

# dense backend: chroma | none
DENSE_BACKEND = os.getenv("DENSE_BACKEND", "chroma").strip().lower()

# 子词分词模型（默认指向 models/bge-m3 下的 sentencepiece 模型）
SENTENCEPIECE_MODEL_PATH = os.getenv(
	"SENTENCEPIECE_MODEL_PATH",
	str(PROJECT_ROOT.parent / "models" / "bge-m3" / "sentencepiece.bpe.model"),
)
SENTENCEPIECE_MODEL_PATH = _resolve_env_path_str(SENTENCEPIECE_MODEL_PATH, base_dir=PROJECT_ROOT)

# 多跳检索默认关闭。大索引下它会额外触发检索，适合复杂问题手动开启。
ENABLE_MULTI_HOP = os.getenv("ENABLE_MULTI_HOP", "0") == "1"
MULTI_HOP_MAX_HOPS = int(os.getenv("MULTI_HOP_MAX_HOPS", "2"))
MULTI_HOP_EXPAND_TITLES = int(os.getenv("MULTI_HOP_EXPAND_TITLES", "2"))

# Query Rewrite 默认关闭。它包含一次额外 LLM 调用，简单事实题通常不需要。
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "0") == "1"
WEB_FALLBACK_MIN_LOCAL_REFS = int(os.getenv("WEB_FALLBACK_MIN_LOCAL_REFS", "1"))

# 联网补充：默认关闭，开启后仅在本地证据不足时触发。
ALLOW_WEB_FALLBACK = os.getenv("ALLOW_WEB_FALLBACK", "0") == "1"
WEB_SEARCH_TIMEOUT_SECONDS = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "6"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
SERPAPI_ENGINE = os.getenv("SERPAPI_ENGINE", "google")
SERPAPI_GL = os.getenv("SERPAPI_GL", "cn")
SERPAPI_HL = os.getenv("SERPAPI_HL", "zh-cn")

# 简洁事实型回答
ANSWER_STYLE = "请用简洁事实型中文回答，优先给出结论，避免冗长解释。不要使用 Markdown 标题符号 #，不要用 ** 包裹加粗；需要分点时直接使用 1. 2. 3. 或短段落。"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.5:4b")

# 本地专属向量化接口（专门给 bge-m3 提问专用，解耦线上大模型）
# 自动检测外部 models/bge-m3 目录，如果有则自动加载本地离线模型，无需启动 Ollama
DEFAULT_LOCAL_MODEL = PROJECT_ROOT.parent / "models" / "bge-m3"
LOCAL_EMBED_MODEL_PATH = _resolve_env_path_str(
    os.getenv("LOCAL_EMBED_MODEL_PATH", str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else ""),
    base_dir=PROJECT_ROOT,
)
LOCAL_EMBED_BASE_URL = os.getenv("LOCAL_EMBED_BASE_URL", "http://127.0.0.1:11434/v1")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "bge-m3")

# 生成模型兜底：优先本地 Ollama；不可用时切换到阿里云百炼（OpenAI 兼容接口）。
ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "1") == "1"
LLM_PRIMARY_PROVIDER = os.getenv("LLM_PRIMARY_PROVIDER", "bailian").strip().lower()
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")

FORCE_BM25_ONLY = os.getenv("FORCE_BM25_ONLY", "0") == "1"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "220"))
EMBED_TIMEOUT_SECONDS = float(os.getenv("EMBED_TIMEOUT_SECONDS", "8"))
EMBED_FAILURE_THRESHOLD = int(os.getenv("EMBED_FAILURE_THRESHOLD", "2"))
EMBED_COOLDOWN_SECONDS = float(os.getenv("EMBED_COOLDOWN_SECONDS", "60"))
LLM_FAILURE_THRESHOLD = int(os.getenv("LLM_FAILURE_THRESHOLD", "2"))
LLM_COOLDOWN_SECONDS = float(os.getenv("LLM_COOLDOWN_SECONDS", "90"))

# Redis 对话历史存储
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# 最多保留几轮对话 (默认3-5轮, 这里配最大上限为5轮, 1轮包含user+assistant 2条)
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "5"))

# 预埋 reranker: 默认关闭，不影响现有链路
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "0") == "1"
DEFAULT_LOCAL_RERANKER = PROJECT_ROOT.parent / "models" / "bge-reranker-v2-m3"
LOCAL_RERANK_MODEL_PATH = _resolve_env_path_str(
    os.getenv("LOCAL_RERANK_MODEL_PATH", str(DEFAULT_LOCAL_RERANKER) if DEFAULT_LOCAL_RERANKER.exists() else ""),
    base_dir=PROJECT_ROOT,
)
OPENAI_RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "40"))
