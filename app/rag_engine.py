from __future__ import annotations

import json
import importlib
import pickle
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import sentencepiece as spm
import numpy as np
from openai import OpenAI

ChromaClient = None
try:
    _chroma = importlib.import_module("chromadb")
    ChromaClient = getattr(_chroma, "PersistentClient", None)
except Exception:
    ChromaClient = None

from .config import (
    ALLOW_WEB_FALLBACK,
    ANSWER_STYLE,
    BM25_PATH,
    CHUNKS_PATH,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBED_COOLDOWN_SECONDS,
    EMBED_FAILURE_THRESHOLD,
    EMBED_TIMEOUT_SECONDS,
    ENABLE_LLM_FALLBACK,
    ENABLE_QUERY_REWRITE,
    LLM_PRIMARY_PROVIDER,
    DENSE_BACKEND,
    ENABLE_MULTI_HOP,
    ENABLE_RERANKER,
    FORCE_BM25_ONLY,
    GEN_CONTEXT_CHARS,
    GEN_CONTEXT_TOP_N,
    LLM_COOLDOWN_SECONDS,
    LLM_FAILURE_THRESHOLD,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    BAILIAN_API_KEY,
    BAILIAN_BASE_URL,
    BAILIAN_MODEL,
    MULTI_HOP_EXPAND_TITLES,
    MULTI_HOP_MAX_HOPS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LOCAL_EMBED_BASE_URL,
    LOCAL_EMBED_MODEL_PATH,
    LOCAL_RERANK_MODEL_PATH,
    OPENAI_EMBED_MODEL,
    OPENAI_MODEL,
    OPENAI_RERANK_MODEL,
    RERANK_TOP_N,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_MERGE,
    TOP_K_VECTOR,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT_SECONDS,
    WEB_FALLBACK_MIN_LOCAL_REFS,
    SERPAPI_API_KEY,
    SERPAPI_ENGINE,
    SERPAPI_GL,
    SERPAPI_HL,
    SENTENCEPIECE_MODEL_PATH,
)

try:
    from serpapi import SerpApiClient as _SerpApiClient
except Exception:
    _SerpApiClient = None

try:
    from serpapi import GoogleSearch as _SerpApiGoogleSearch
except Exception:
    _SerpApiGoogleSearch = None

# 尝试在模块级加载 SentencePiece 模型（可用则用于子词分词）
SP: spm.SentencePieceProcessor | None = None
try:
    SP = spm.SentencePieceProcessor()
    try:
        SP.Load(SENTENCEPIECE_MODEL_PATH)
    except Exception:
        SP = None
except Exception:
    SP = None


@dataclass
class ScoredChunk:
    idx: int
    score: float


ZH_QUERY_STOPWORDS = {
    "什么",
    "什么是",
    "请问",
    "一下",
    "关于",
    "吗",
    "呢",
}

GENERIC_ANCHOR_TERMS = {
    "原理", "产业", "行业", "企业", "公司", "集团",
    "概念", "定义", "特点", "作用", "功能", "意义", "影响", "原因",
    "关系", "区别", "资料", "简介", "介绍", "说明", "历史", "数据",
    "位置", "地点", "所在地", "时间", "作者", "导演", "校长",
    "首都", "省会", "本名", "原名", "别名",
}


class RagEngine:
    def __init__(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._on_stage = on_stage
        self._index_error = ""
        self._dense_backend = DENSE_BACKEND
        self._dense_disabled_reason = ""
        self._chroma = None
        self._index_ntotal = 0
        self.index = None
        self._last_retrieval_stats = {
            "bm25_hits": 0,
            "vec_hits": 0,
            "vec_used": False,
            "rerank_used": False,
        }
        self._last_trace = {
            "rewrite_used": False,
            "rewrite_queries": [],
            "web_used": False,
            "web_provider": "none",
            "web_ref_count": 0,
            "public_ref_count": 0,
            "public_ref_reason": "",
            "timing_ms": {},
            "anchor_rescue_used": False,
            "anchor_rescue_count": 0,
            "anchor_terms": [],
            "gen_provider": "",
            "gen_model": "",
        }
        self._emit_stage("loading_chunks")
        self.chunks = self._load_chunks(CHUNKS_PATH)
        self._chunk_id_to_row: dict[str, int] = {}
        self._title_to_rows: dict[str, list[int]] = {}
        self._lower_titles: list[str] = []
        for i, row in enumerate(self.chunks):
            cid = str(row.get("chunk_id", "")).strip()
            if cid:
                self._chunk_id_to_row[cid] = i
            title = str(row.get("title", "")).strip()
            self._lower_titles.append(title.lower())
            if title:
                rows = self._title_to_rows.setdefault(title, [])
                if len(rows) < 8:
                    rows.append(i)
        self._anchor_rescue_cache: dict[str, list[int]] = {}

        self._emit_stage("loading_bm25")
        self.bm25 = self._load_pickle(BM25_PATH)

        if FORCE_BM25_ONLY:
            self._emit_stage("skipping_dense")
            self.index = None
            self._index_error = "dense retrieval disabled"
            self._dense_disabled_reason = "FORCE_BM25_ONLY=1"
        elif self._dense_backend == "none":
            self._emit_stage("skipping_dense")
            self.index = None
            self._index_error = "dense retrieval disabled"
            self._dense_disabled_reason = "DENSE_BACKEND=none"
        elif self._dense_backend == "chroma":
            self._emit_stage("loading_chroma")
            if ChromaClient is None:
                self._index_error = "chromadb not installed"
                self._dense_disabled_reason = "chromadb not installed"
            elif not CHROMA_PERSIST_DIR.exists():
                self._index_error = f"missing Chroma persist dir: {CHROMA_PERSIST_DIR}"
                self._dense_disabled_reason = self._index_error
            else:
                try:
                    self._chroma = ChromaClient(path=str(CHROMA_PERSIST_DIR))
                    self._chroma_collection = self._chroma.get_collection(name=CHROMA_COLLECTION)
                    self._index_ntotal = self._get_index_ntotal()
                    if self._index_ntotal <= 0:
                        self._index_error = (
                            f"empty Chroma collection: name={CHROMA_COLLECTION}, path={CHROMA_PERSIST_DIR}"
                        )
                        self._dense_disabled_reason = self._index_error
                except Exception as e:
                    self._chroma = None
                    self._index_error = str(e)
                    self._dense_disabled_reason = self._index_error
        else:
            self._emit_stage("skipping_dense")
            self.index = None
            self._index_error = "dense retrieval disabled"
            self._dense_disabled_reason = f"unsupported DENSE_BACKEND={self._dense_backend}"

        self._emit_stage("initializing_client")
        self.client = None
        self._llm_fallback_client = None
        self.embed_client = None
        self.local_embed_model = None
        self.local_rerank_model = None
        self._local_embed_device = ""
        self._local_rerank_device = ""
        self._chroma_collection: Any | None = getattr(self, "_chroma_collection", None)

        if ENABLE_RERANKER and LOCAL_RERANK_MODEL_PATH:
            try:
                from sentence_transformers import CrossEncoder
                self.local_rerank_model = CrossEncoder(LOCAL_RERANK_MODEL_PATH)
                self._local_rerank_device = self._detect_model_device(self.local_rerank_model)
            except ImportError:
                print("未安装 sentence-transformers，或者模型路径不存在，请检查配置")
                pass

        if LOCAL_EMBED_MODEL_PATH:
            # 如果配置了本地下载的 HuggingFace 模型路径，则使用 sentence-transformers 直接加载入内存
            try:
                from sentence_transformers import SentenceTransformer
                self.local_embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
                self._local_embed_device = self._detect_model_device(self.local_embed_model)
            except ImportError:
                print("未安装 sentence-transformers，请执行 pip install sentence-transformers")
                raise
        else:
            # 否则回退为原本的 OpenAI API 兼容形式 (如 Ollama)，与 OPENAI_API_KEY 对齐便于自定义网关
            self.embed_client = OpenAI(
                api_key=OPENAI_API_KEY or "ollama",
                base_url=LOCAL_EMBED_BASE_URL,
                timeout=EMBED_TIMEOUT_SECONDS,
                max_retries=0,
            )

        self._llm_primary_provider = "bailian" if LLM_PRIMARY_PROVIDER == "bailian" else "ollama"
        self._llm_provider_last = "extractive"
        if OPENAI_API_KEY or OPENAI_BASE_URL:
            # Ollama 的 OpenAI 兼容接口通常不要求真实 key，这里给默认占位值。
            self.client = OpenAI(
                api_key=OPENAI_API_KEY or "ollama",
                base_url=OPENAI_BASE_URL or None,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=0,
            )

            # 简单熔断器：上游连续失败后短期禁用该通道，优先保证接口稳定返回。
            self._embed_fail_count = 0
            self._embed_disabled_until = 0.0
            self._llm_fail_count = 0
            self._llm_disabled_until = 0.0

        if ENABLE_LLM_FALLBACK and BAILIAN_API_KEY:
            self._llm_fallback_client = OpenAI(
                api_key=BAILIAN_API_KEY,
                base_url=BAILIAN_BASE_URL,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=0,
            )

        if not hasattr(self, "_embed_fail_count"):
            self._embed_fail_count = 0
            self._embed_disabled_until = 0.0
            self._llm_fail_count = 0
            self._llm_disabled_until = 0.0

        self._emit_stage("ready")

    def _emit_stage(self, stage: str) -> None:
        if self._on_stage:
            self._on_stage(stage)

    def _record_timing(self, key: str, started_at: float) -> None:
        timings = self._last_trace.setdefault("timing_ms", {})
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        timings[key] = round(float(timings.get(key, 0.0)) + elapsed_ms, 2)

    @staticmethod
    def _detect_model_device(model: Any) -> str:
        candidates = [
            getattr(model, "device", None),
            getattr(model, "_target_device", None),
        ]
        inner_model = getattr(model, "model", None)
        if inner_model is not None:
            candidates.append(getattr(inner_model, "device", None))
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _load_chunks(path: Path):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _tokenize(text: str):
        if SP is not None:
            try:
                toks = SP.encode_as_pieces(text)
                return [t for t in toks if t.strip()]
            except Exception:
                pass
        # 回退：按空白切分
        return [w for w in text.split() if w.strip()]

    @staticmethod
    def _normalize_query(text: str) -> str:
        q = text.strip()
        q = re.sub(r"[?？!！。,.，:：;；]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        for bad in ["什么是", "请问", "一下", "关于"]:
            q = q.replace(bad, " ")
        q = re.sub(r"\s+", " ", q).strip()
        return q or text.strip()

    @staticmethod
    def _query_guard_terms(text: str) -> set[str]:
        """Terms that a rewritten query must keep anchored to."""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        terms: set[str] = set()

        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}", normalized):
            terms.add(token)

        for block in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            if len(block) <= 6:
                terms.add(block)
            for size in (2, 3, 4):
                if len(block) >= size:
                    for i in range(0, len(block) - size + 1):
                        terms.add(block[i : i + size])

        generic = {
            "什么", "什么是", "请问", "关于", "介绍", "说明", "区别", "关系",
            "如何", "为什么", "多少", "哪个", "哪些", "是谁", "是啥",
        }
        return {t for t in terms if t and t not in generic}

    @staticmethod
    def _query_anchor_terms(text: str) -> set[str]:
        """Higher precision entity-like anchors that must survive rewriting."""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        anchors: set[str] = set()
        generic = {
            "什么", "什么是", "请问", "关于", "介绍", "说明", "区别", "关系",
            "如何", "为什么", "多少", "哪个", "哪些", "是谁", "是啥",
            "历史", "数据", "生涯", "主要", "原因", "影响", "意义",
        }

        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", normalized):
            if token not in generic:
                anchors.add(token)

        for block in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            cleaned = block
            cleaned = re.sub(r"(是什么|是谁提出|谁提出|是谁|有哪些|如何|为什么|多少|哪个|哪些)$", "", cleaned)
            cleaned = re.sub(r"(生涯数据|数据|生涯|历史|介绍|说明|原因|影响|意义|关系|区别)$", "", cleaned)
            cleaned = re.sub(r"[的地得了和与及或是有在为对中里上下一些多少哪个哪些谁]+$", "", cleaned)
            if len(cleaned) >= 2 and cleaned not in generic:
                anchors.add(cleaned)
            if len(cleaned) > 4:
                first = cleaned[:2]
                if first not in generic:
                    anchors.add(first)
        return anchors

    @staticmethod
    def _is_generic_anchor_term(term: str) -> bool:
        cleaned = str(term or "").strip()
        if not cleaned:
            return True
        return cleaned in GENERIC_ANCHOR_TERMS

    @classmethod
    def _effective_anchor_terms(cls, question: str) -> list[str]:
        terms = cls._core_entity_terms(question)
        strong_terms = [term for term in terms if not cls._is_generic_anchor_term(term)]
        return strong_terms or terms

    @classmethod
    def _primary_anchor_terms(cls, question: str) -> list[str]:
        terms = cls._effective_anchor_terms(question)
        primary = [term for term in terms if "的" not in term and len(term.strip()) >= 2]
        return primary or terms

    @staticmethod
    def _normalize_entity_text(text: str) -> str:
        cleaned = re.sub(r"[\s\u00b7·•・'\"“”‘’`~!@#$%^&*()_+\-=\[\]{}\\|;:：,，。！？?/<>《》〈〉（）]", "", str(text or ""))
        return cleaned.lower().strip()

    @staticmethod
    def _is_entity_char(ch: str) -> bool:
        return bool(re.match(r"[a-z0-9\u4e00-\u9fff]", ch or "", re.IGNORECASE))

    @classmethod
    def _entity_match_strength(cls, text: str, term: str) -> int:
        raw_text = str(text or "").strip()
        raw_term = str(term or "").strip()
        if not raw_text or not raw_term:
            return 0

        text_norm = cls._normalize_entity_text(raw_text)
        term_norm = cls._normalize_entity_text(raw_term)
        if not text_norm or not term_norm or len(term_norm) < 2:
            return 0
        if text_norm == term_norm:
            return 3

        pos = text_norm.find(term_norm)
        if pos < 0:
            return 0

        end = pos + len(term_norm)
        before = text_norm[pos - 1] if pos > 0 else ""
        after = text_norm[end] if end < len(text_norm) else ""
        before_is_entity = cls._is_entity_char(before)
        after_is_entity = cls._is_entity_char(after)

        if not before_is_entity and not after_is_entity:
            return 3
        if end == len(text_norm) and pos <= 6:
            return 2
        if pos == 0 and not after_is_entity:
            return 2
        return 1

    @classmethod
    def _answer_focus_terms(cls, answer: str) -> list[str]:
        text = re.sub(r"\s+", " ", str(answer or "")).strip()
        if not text:
            return []

        terms: list[str] = []

        def add_term(value: str) -> None:
            cleaned = str(value or "").strip(" ：:;；，,。.!?、()（）[]{}<>《》〈〉\"'“”‘’ ")
            cleaned = re.sub(r"(共同)?(撰写|写作|创办|创立|创建|发明|发现|设计|建立|提出|担任|收购|投资|经营|从事|专注)[的者]?$", "", cleaned)
            cleaned = re.sub(r"(公司|组织|机构|人物|作品|著作|国家)$", "", cleaned)
            cleaned = cleaned.strip(" ：:;；，,。.!?、()（）[]{}<>《》〈〉\"'“”‘’ ")
            if len(cleaned) >= 2 and not cls._is_generic_anchor_term(cleaned) and cleaned not in terms:
                terms.append(cleaned)

        first_lines = [seg.strip() for seg in re.split(r"[\n。；!！?？]", text) if seg.strip()][:3]
        for line in first_lines:
            for quoted in re.findall(r"《([^》]{1,40})》", line):
                add_term(quoted)
            for match in re.findall(r"(?:是|为|包括|包含|指|即|有)[:：]?\s*([^。；!！?？\n]{1,60})", line):
                for part in re.split(r"[、/]|和|与|及|以及|并", match):
                    add_term(part)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9.&\- ]{1,40}", line):
                add_term(token)

        for line in first_lines:
            for term in cls._effective_anchor_terms(line):
                add_term(term)

        return terms[:6]

    @staticmethod
    def _core_entity_terms(text: str) -> list[str]:
        q = re.sub(r"[?？!！。,.，:：;；]", " ", str(text or "")).strip()
        q = re.sub(r"\s+", " ", q)
        candidates: list[str] = []

        def add_candidate(value: str) -> None:
            raw_value = value.strip(" 《》〈〉（）()")
            variants = [raw_value]
            if "的" in raw_value:
                head = raw_value.split("的", 1)[0]
                tail = raw_value.rsplit("的", 1)[-1]
                if head and head != raw_value:
                    variants.append(head)
                if tail and tail != raw_value:
                    variants.append(tail)
            variants.append(
                re.sub(
                    r"^(中国|我国|国内|中华|中国古代|古代中国|世界|全球|古代|近代|现代|当代)的",
                    "",
                    raw_value,
                )
            )
            variants.append(
                re.sub(
                    r"^(中国|我国|中华|世界|全球|古代|近代|现代|当代)(?=[\u4e00-\u9fff]{2,})",
                    "",
                    raw_value,
                )
            )

            for variant in variants:
                cleaned_value = variant.strip(" 《》〈〉（）()")
                cleaned_value = re.sub(r"^(请问|帮我查|介绍一下|介绍|关于|什么是)", "", cleaned_value)
                cleaned_value = re.sub(r"^(了|的|是|为|在|位于|属于)", "", cleaned_value)
                cleaned_value = re.sub(r"^[的地得了和与及或是有在为对里上下一些]+", "", cleaned_value)
                cleaned_value = re.sub(r"[的地得了和与及或是有在为对里上下一些]+$", "", cleaned_value)
                if len(cleaned_value) >= 2 and cleaned_value not in candidates:
                    candidates.append(cleaned_value)

        q_core = re.sub(r"^(请问|帮我查|介绍一下|介绍|关于|我想知道|什么是)", "", q)
        q_core = q_core.strip()

        subject_suffixes = [
            "是谁创办的", "是谁创立的", "是谁创建的", "谁创办的", "谁创立的", "谁创建的",
            "由谁创办", "由谁创立", "由谁创建", "是谁提出的", "谁提出的", "是谁提出", "谁提出",
            "是谁发明的", "谁发明的", "是谁发现的", "谁发现的", "是谁设计的", "谁设计的",
            "是谁建立的", "谁建立的",
            "创办者是谁", "创立者是谁", "创建者是谁", "创办人是谁", "创始人是谁",
            "原名是什么", "本名是什么", "别名是什么", "作者是谁", "导演是谁", "校长是谁",
            "首都是哪里", "省会是哪里", "位置在哪里", "地点在哪里", "所在地在哪里",
            "成立时间是什么", "创办时间是什么", "创立时间是什么", "创建时间是什么",
            "位于哪里", "在哪里", "在哪", "属于哪里", "属于哪个", "隶属于哪里",
            "分别是什么", "分别是哪些", "包括哪些", "包括什么", "包含哪些", "包含什么",
            "有哪几个", "是哪几个", "是哪几项", "指哪些", "指什么", "有哪些",
            "是谁", "是什么", "是啥",
        ]
        for suffix in subject_suffixes:
            if q_core.endswith(suffix):
                add_candidate(q_core[: -len(suffix)])

        object_prefixes = [
            "谁创办了", "谁创立了", "谁创建了", "谁提出了", "谁发明了", "谁发现了", "谁设计了", "谁建立了",
            "哪位创办了", "哪位创立了", "哪位创建了", "哪位提出了", "哪位发明了", "哪位发现了",
            "谁创办", "谁创立", "谁创建", "谁提出", "谁发明", "谁发现", "谁设计", "谁建立",
            "哪位创办", "哪位创立", "哪位创建", "哪位提出", "哪位发明", "哪位发现",
        ]
        for prefix_text in object_prefixes:
            if q_core.startswith(prefix_text):
                add_candidate(q_core[len(prefix_text):])

        attr_markers = [
            "的原名", "的本名", "的别名", "的作者", "的导演", "的校长", "的首都", "的省会",
            "的创办者", "的创立者", "的创建者", "的创办人", "的创始人",
            "的成立时间", "的创办时间", "的创立时间", "的创建时间",
        ]
        for marker in attr_markers:
            pos = q_core.find(marker)
            if pos > 0:
                add_candidate(q_core[:pos])

        for block in re.findall(r"[\u4e00-\u9fff]{2,}", q):
            cleaned = block
            cleaned = re.sub(r"^(请问|帮我查|介绍一下|介绍|关于|什么是)", "", cleaned)
            cleaned = re.sub(
                r"(是谁创办的|是谁创立的|是谁创建的|谁创办的|谁创立的|谁创建的|"
                r"创办者是谁|创立者是谁|创建者是谁|创办人是谁|创始人是谁|"
                r"由谁创办|由谁创立|由谁创建|谁提出的|谁提出|"
                r"是谁发明的|谁发明的|是谁发现的|谁发现的|是谁设计的|谁设计的|"
                r"是谁建立的|谁建立的|"
                r"位于哪里|在哪里|在哪|属于哪里|属于哪个|"
                r"原名是什么|本名是什么|别名是什么|作者是谁|导演是谁|校长是谁|"
                r"成立时间是什么|创办时间是什么|创立时间是什么|创建时间是什么|"
                r"分别是什么|分别是哪些|包括哪些|包括什么|包含哪些|包含什么|"
                r"有哪几个|是哪几个|是哪几项|指哪些|指什么|"
                r"是什么|是谁|是啥|有哪些|如何|为什么|多少|哪个|哪些|吗|呢)$",
                "",
                cleaned,
            )
            cleaned = re.sub(r"(历史|资料|简介|介绍|说明|原因|影响|意义|关系|区别)$", "", cleaned)
            cleaned = re.sub(r"^[的地得了和与及或是有在为对里上下一些]+", "", cleaned)
            cleaned = re.sub(r"[的地得了和与及或是有在为对里上下一些]+$", "", cleaned)
            add_candidate(cleaned)
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", q):
            token = token.strip()
            if token and token not in candidates:
                candidates.append(token)

        noisy_markers = [
            "谁", "什么", "哪里", "哪", "多少", "如何", "为什么", "怎么", "怎样",
            "创办", "创立", "创建", "提出", "设计", "建立",
            "原理", "产业", "行业", "企业", "公司", "集团",
            "原名", "本名", "别名", "作者", "导演", "校长", "首都", "省会",
            "位置", "地点", "所在地", "时间", "包括", "包含", "分别", "哪些", "几个",
        ]
        scope_prefixes = (
            "中国的", "我国的", "国内的", "中华的", "中国古代的", "古代中国的", "世界的", "全球的",
            "中国", "我国", "中华", "世界", "全球", "古代", "近代", "现代", "当代",
        )

        def rank_candidate(term: str) -> tuple[int, int, int, int]:
            generic = 1 if term in GENERIC_ANCHOR_TERMS else 0
            noisy = 1 if any(marker in term for marker in noisy_markers) else 0
            scoped = 1 if term.startswith(scope_prefixes) else 0
            return (generic, noisy, scoped, -len(term))

        candidates.sort(key=rank_candidate)
        return candidates[:3]

    def _anchor_rescue_hits(self, question: str, existing_refs: list[dict]) -> list[ScoredChunk]:
        terms = self._effective_anchor_terms(question)
        self._last_trace["anchor_terms"] = terms
        if not terms:
            return []

        existing_idxs: set[int] = set()
        for r in existing_refs:
            try:
                idx = int(r.get("_idx", -1))
            except Exception:
                continue
            if idx >= 0:
                existing_idxs.add(idx)
        found: list[ScoredChunk] = []
        seen: set[int] = set()

        for term in terms:
            title_rows = self._title_to_rows.get(term, [])
            for row_idx in title_rows:
                if row_idx not in existing_idxs and row_idx not in seen:
                    seen.add(row_idx)
                    found.append(ScoredChunk(row_idx, 2.0))
                    if len(found) >= 6:
                        break
            if len(found) >= 6:
                break

            cache_key = term
            rows = self._anchor_rescue_cache.get(cache_key)
            if rows is None:
                rows = []
                term_l = term.lower()
                for i, title_l in enumerate(self._lower_titles):
                    if term_l in title_l:
                        rows.append(i)
                        if len(rows) >= 12:
                            break
                self._anchor_rescue_cache[cache_key] = rows

            for row_idx in rows:
                if row_idx not in existing_idxs and row_idx not in seen:
                    seen.add(row_idx)
                    row = self.chunks[row_idx]
                    title = str(row.get("title", ""))
                    score = 1.8 if term == title else 1.4 if term in title else 1.0
                    found.append(ScoredChunk(row_idx, score))
                    if len(found) >= 6:
                        break
            if found:
                break

        self._last_trace["anchor_rescue_count"] = len(found)
        self._last_trace["anchor_rescue_used"] = bool(found)
        return found

    def _prioritize_anchor_refs(self, question: str, refs: list[dict]) -> list[dict]:
        terms = self._effective_anchor_terms(question)
        primary_terms = self._primary_anchor_terms(question)
        if not terms or not refs:
            return refs

        strongly_anchored: list[dict] = []
        anchored: list[dict] = []
        others: list[dict] = []
        for ref in refs:
            title = str(ref.get("title", ""))
            text = str(ref.get("full_text") or ref.get("snippet") or "")
            matched = False
            primary_matched = False
            best_score = 0.0
            for term in primary_terms:
                if not term:
                    continue
                title_strength = self._entity_match_strength(title, term)
                text_strength = self._entity_match_strength(text, term)
                if title_strength >= 3:
                    primary_matched = True
                    best_score = max(best_score, 3.2)
                elif title_strength >= 2:
                    primary_matched = True
                    best_score = max(best_score, 2.7)
                elif text_strength >= 2:
                    primary_matched = True
                    best_score = max(best_score, 2.1)
            for term in terms:
                if not term:
                    continue
                title_strength = self._entity_match_strength(title, term)
                text_strength = self._entity_match_strength(text, term)
                if title_strength >= 3:
                    matched = True
                    best_score = max(best_score, 3.0)
                elif title_strength >= 2:
                    matched = True
                    best_score = max(best_score, 2.5)
                elif text_strength >= 2:
                    matched = True
                    best_score = max(best_score, 2.0)
            if primary_matched:
                item = dict(ref)
                item["score"] = max(float(item.get("score", 0.0)), best_score)
                strongly_anchored.append(item)
            elif matched:
                item = dict(ref)
                item["score"] = max(float(item.get("score", 0.0)), best_score)
                anchored.append(item)
            else:
                others.append(ref)

        strongly_anchored.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        anchored.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return strongly_anchored + anchored + others

    def _filter_public_refs(self, question: str, refs: list[dict], answer: str = "") -> list[dict]:
        terms = self._effective_anchor_terms(question)
        primary_terms = self._primary_anchor_terms(question)
        if not refs:
            self._last_trace["public_ref_count"] = 0
            self._last_trace["public_ref_reason"] = "no_refs"
            return []
        if not terms:
            out = self._dedupe_public_refs(refs)
            self._last_trace["public_ref_count"] = len(out)
            self._last_trace["public_ref_reason"] = "no_anchor_terms"
            return out

        primary_anchored = []
        secondary_anchored = []
        for ref in refs:
            title = str(ref.get("title", ""))
            text = str(ref.get("full_text") or ref.get("snippet") or "")
            if any(term and (self._entity_match_strength(title, term) >= 2 or self._entity_match_strength(text, term) >= 2) for term in primary_terms):
                primary_anchored.append(ref)
            elif any(term and (self._entity_match_strength(title, term) >= 2 or self._entity_match_strength(text, term) >= 2) for term in terms):
                secondary_anchored.append(ref)

        focus_terms = self._answer_focus_terms(answer)
        if focus_terms:
            def answer_consistent(candidates: list[dict]) -> list[dict]:
                kept = []
                for ref in candidates:
                    title = str(ref.get("title", ""))
                    text = str(ref.get("full_text") or ref.get("snippet") or "")
                    if any(
                        self._entity_match_strength(title, term) >= 2 or self._entity_match_strength(text, term) >= 2
                        for term in focus_terms
                    ):
                        kept.append(ref)
                return kept

            primary_consistent = answer_consistent(primary_anchored)
            secondary_consistent = answer_consistent(secondary_anchored)
            if primary_consistent:
                out = self._dedupe_public_refs(primary_consistent)
                self._last_trace["public_ref_count"] = len(out)
                self._last_trace["public_ref_reason"] = "primary_anchor+answer_consistent"
                return out
            if secondary_consistent:
                out = self._dedupe_public_refs(secondary_consistent)
                self._last_trace["public_ref_count"] = len(out)
                self._last_trace["public_ref_reason"] = "secondary_anchor+answer_consistent"
                return out

        if primary_anchored:
            out = self._dedupe_public_refs(primary_anchored)
            self._last_trace["public_ref_count"] = len(out)
            self._last_trace["public_ref_reason"] = "primary_anchor_only"
            return out
        if secondary_anchored:
            out = self._dedupe_public_refs(secondary_anchored)
            self._last_trace["public_ref_count"] = len(out)
            self._last_trace["public_ref_reason"] = "secondary_anchor_only"
            return out
        self._last_trace["public_ref_count"] = 0
        self._last_trace["public_ref_reason"] = "filtered_out"
        return []

    @staticmethod
    def _dedupe_public_refs(refs: list[dict]) -> list[dict]:
        deduped: dict[tuple[str, str, str], dict] = {}
        for ref in refs:
            key = (
                str(ref.get("doc_id", "")),
                str(ref.get("title", "")),
                str(ref.get("url", "")),
            )
            prev = deduped.get(key)
            if prev is None or float(ref.get("score", 0.0)) > float(prev.get("score", 0.0)):
                deduped[key] = ref
        out = list(deduped.values())
        out.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return out

    @staticmethod
    def _clean_query_candidate(text: str) -> str:
        q = str(text or "").strip()
        q = re.sub(r"^\s*[\-\*\d一二三四五六七八九十]+[\.、:：\)]\s*", "", q)
        q = q.strip(" \t\r\n\"'“”‘’`")
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _is_safe_query_variant(self, original: str, candidate: str) -> bool:
        c = self._clean_query_candidate(candidate)
        if not c or c == original:
            return False
        if len(c) > 90 or len(c) > max(40, int(len(original) * 2.2)):
            return False
        if "\n" in c:
            return False
        if re.search(r"(答案|证据|片段|根据|假设|可能是|以下|无法确定|不足以|综上)", c):
            return False

        original_terms = self._query_guard_terms(original)
        anchor_terms = self._query_anchor_terms(original)
        chinese_anchors = {a for a in anchor_terms if re.search(r"[\u4e00-\u9fff]", a)}
        required_anchors = chinese_anchors or anchor_terms
        if required_anchors and not any(anchor in c.lower() for anchor in required_anchors):
            return False
        if not original_terms:
            return True
        candidate_l = c.lower()
        hits = sum(1 for term in original_terms if term in candidate_l)
        required = 1 if len(original_terms) <= 3 else 2
        return hits >= required

    def _safe_query_variants(self, original: str, candidates: list[str], *, limit: int = 2) -> list[str]:
        results = [original]
        for raw in candidates:
            cleaned = self._clean_query_candidate(raw)
            if self._is_safe_query_variant(original, cleaned) and cleaned not in results:
                results.append(cleaned)
            if len(results) >= max(1, limit):
                break
        return results

    @staticmethod
    def _title_overlap_boost(title: str, query_tokens: list[str]) -> float:
        if not title:
            return 0.0
        t = title.lower()
        hits = 0
        for tok in query_tokens:
            if len(tok) < 2:
                continue
            if tok.lower() in t:
                hits += 1
        return float(hits) * 0.35

    def _get_index_ntotal(self) -> int:
        cached = int(getattr(self, "_index_ntotal", 0) or 0)
        if cached > 0:
            return cached
        if self.index is not None:
            try:
                self._index_ntotal = int(self.index.ntotal)
                return self._index_ntotal
            except Exception:
                return 0
        if self._chroma_collection is not None:
            try:
                self._index_ntotal = int(self._chroma_collection.count())
                return self._index_ntotal
            except Exception:
                return 0
        return 0

    def _bm25_retrieve(self, q: str) -> list[ScoredChunk]:
        started_at = time.perf_counter()
        nq = self._normalize_query(q)
        tokens = [t for t in self._tokenize(nq) if t not in ZH_QUERY_STOPWORDS and t.strip()]
        if not tokens:
            tokens = self._tokenize(q)

        scores = self.bm25.get_scores(tokens)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            self._record_timing("bm25_ms", started_at)
            return []

        # 先取较小候选池，再做标题加权，避免对全量 290w 行逐条 Python 循环。
        candidate_k = max(TOP_K_BM25, min(len(scores), TOP_K_BM25 * 8))
        if candidate_k >= len(scores):
            idx = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -candidate_k)[-candidate_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]

        boosted: list[tuple[int, float]] = []
        for i in idx:
            score = float(scores[i]) + self._title_overlap_boost(self._lower_titles[int(i)], tokens)
            boosted.append((int(i), score))
        boosted.sort(key=lambda x: x[1], reverse=True)
        boosted = boosted[:TOP_K_BM25]

        out = [ScoredChunk(i, score) for i, score in boosted]
        self._record_timing("bm25_ms", started_at)
        return out

    def _vec_retrieve(self, q: str) -> list[ScoredChunk]:
        started_at = time.perf_counter()
        if not self.embed_client and not self.local_embed_model:
            return []
        if time.time() < self._embed_disabled_until:
            return []

        if self._dense_backend == "chroma":
            return self._vec_retrieve_chroma(q)
        if self.index is None:
            return []

        try:
            if self.local_embed_model:
                vec = self.local_embed_model.encode(q, normalize_embeddings=True)
            else:
                resp = self.embed_client.with_options(timeout=EMBED_TIMEOUT_SECONDS, max_retries=0).embeddings.create(
                    model=OPENAI_EMBED_MODEL,
                    input=[q],
                )
                vec = resp.data[0].embedding
                
            qv = np.asarray([vec], dtype=np.float32)
            # 与 build_index 保持一致：使用归一化向量 + 内积检索
            norm = np.linalg.norm(qv, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            qv = qv / norm
            scores, idx = self.index.search(qv, TOP_K_VECTOR)
            out = []
            for i, s in zip(idx[0], scores[0]):
                if i >= 0:
                    out.append(ScoredChunk(int(i), float(s)))
            self._embed_fail_count = 0
            self._dense_disabled_reason = ""
            self._record_timing("vector_ms", started_at)
            return out
        except Exception as e:
            self._dense_disabled_reason = f"Embedding failure: {e!r}"
            self._embed_fail_count += 1
            if self._embed_fail_count >= EMBED_FAILURE_THRESHOLD:
                self._embed_disabled_until = time.time() + EMBED_COOLDOWN_SECONDS
                self._embed_fail_count = 0
            self._record_timing("vector_ms", started_at)
            return []

    def _vec_retrieve_chroma(self, q: str) -> list[ScoredChunk]:
        started_at = time.perf_counter()
        if self._chroma_collection is None:
            return []

        try:
            if self.local_embed_model:
                vec = self.local_embed_model.encode(q, normalize_embeddings=True)
            else:
                resp = self.embed_client.with_options(timeout=EMBED_TIMEOUT_SECONDS, max_retries=0).embeddings.create(
                    model=OPENAI_EMBED_MODEL,
                    input=[q],
                )
                vec = resp.data[0].embedding

            vec = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                norm = 1.0
            qv = (vec / norm).tolist()

            res = self._chroma_collection.query(
                query_embeddings=[qv],
                n_results=TOP_K_VECTOR,
                include=["distances"],
            )

            ids = (res.get("ids") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]

            out: list[ScoredChunk] = []
            for chroma_id, distance in zip(ids, distances):
                sid = str(chroma_id)
                row_idx = self._chunk_id_to_row.get(sid)
                if row_idx is None:
                    # 兼容早期仅有顺序号、未写入 chunk_id 的假数据
                    try:
                        row_idx = int(sid)
                    except Exception:
                        continue
                    if row_idx < 0 or row_idx >= len(self.chunks):
                        continue
                score = 1.0 - float(distance)
                out.append(ScoredChunk(idx=row_idx, score=score))

            self._embed_fail_count = 0
            self._dense_disabled_reason = ""
            self._record_timing("vector_ms", started_at)
            return out
        except Exception as e:
            self._dense_disabled_reason = f"Chroma/Embedding failure: {e!r}"
            self._embed_fail_count += 1
            if self._embed_fail_count >= EMBED_FAILURE_THRESHOLD:
                self._embed_disabled_until = time.time() + EMBED_COOLDOWN_SECONDS
                self._embed_fail_count = 0
            self._record_timing("vector_ms", started_at)
            return []

    

    @staticmethod
    def _rrf_merge(a: list[ScoredChunk], b: list[ScoredChunk], k: int = 60) -> list[ScoredChunk]:
        rank_a = {x.idx: r for r, x in enumerate(a, start=1)}
        rank_b = {x.idx: r for r, x in enumerate(b, start=1)}
        keys = set(rank_a) | set(rank_b)
        merged = []
        for idx in keys:
            ra = rank_a.get(idx, 10_000)
            rb = rank_b.get(idx, 10_000)
            score = 1.0 / (k + ra) + 1.0 / (k + rb)
            merged.append(ScoredChunk(idx=idx, score=score))
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:TOP_K_MERGE]

    def _hybrid_retrieve(self, query: str) -> list[ScoredChunk]:
        bm25_hits = self._bm25_retrieve(query)
        vec_hits = self._vec_retrieve(query)
        self._last_retrieval_stats = {
            "bm25_hits": len(bm25_hits),
            "vec_hits": len(vec_hits),
            "vec_used": len(vec_hits) > 0,
            "rerank_used": False,
        }
        return self._rrf_merge(bm25_hits, vec_hits)

    def _hits_to_refs(self, hits: list[ScoredChunk]) -> list[dict]:
        refs = []
        for h in hits:
            c = self.chunks[h.idx]
            refs.append(
                {
                    "_idx": h.idx,
                    "doc_id": c.get("doc_id", ""),
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                    "score": round(h.score, 6),
                    "snippet": c.get("text", "")[:220],
                    "full_text": c.get("text", ""),
                }
            )
        return refs

    @staticmethod
    def _merge_refs(primary: list[dict], secondary: list[dict]) -> list[dict]:
        merged: dict[int, dict] = {}
        for r in primary + secondary:
            idx = int(r.get("_idx", -1))
            if idx < 0:
                continue
            prev = merged.get(idx)
            if prev is None or float(r.get("score", 0.0)) > float(prev.get("score", 0.0)):
                merged[idx] = r
        out = list(merged.values())
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    def _build_followup_query(self, question: str, refs: list[dict]) -> str:
        fallback_query = question
        original_terms = self._query_guard_terms(question)
        terms = []
        for r in refs:
            t = str(r.get("title", "")).strip()
            if not t or t in terms or len(t) > 40:
                continue
            title_l = t.lower()
            if original_terms and not any(term in title_l for term in original_terms):
                continue
            terms.append(t)
            if len(terms) >= max(1, MULTI_HOP_EXPAND_TITLES):
                break
        if terms:
            fallback_query = f"{question} {' '.join(terms)}"

        # Conservative multi-hop: do not ask an LLM to invent the next query.
        # Free-form follow-up generation can drift into unrelated entities.
        return fallback_query

        if not getattr(self, "client", None) and not getattr(self, "_llm_fallback_client", None):
            return fallback_query

        # 使用 LLM 总结已检索到的缺失信息并生成 Follow-up 查询
        context_blocks = []
        for i, r in enumerate(refs[:3], start=1):
            context_blocks.append(f"片段{i}: {r.get('snippet','')}")
        
        joined_context = "\n".join(context_blocks)
        system_prompt = (
            "你是一个知识库搜索专家。基于用户原始问题和初次检索到的片段，判断是否缺少某些关键信息来完整回答问题。\n"
            "如果缺少，请生成一个简短的查询语句(Query)来专门搜索缺失的信息；\n"
            "如果现有片段已经足够回答，或者无法推断缺失什么，请直接回复'<足够>'。\n"
            "请只输出查询语句本身，不要输出任何解释或格式。"
        )
        user_prompt_content = f"问题：{question}\n\n已检索片段：\n{joined_context}\n\n请输出下一跳查询："

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]

        def _try_generate(client: Any, model: str) -> str | None:
            resp = client.with_options(timeout=LLM_TIMEOUT_SECONDS, max_retries=0).chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=60,
            )
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            return None

        ollama_available = getattr(self, "client", None) is not None and time.time() >= getattr(self, "_llm_disabled_until", 0.0)
        bailian_available = getattr(self, "_llm_fallback_client", None) is not None

        providers = ["bailian", "ollama"] if getattr(self, "_llm_primary_provider", "bailian") == "bailian" else ["ollama", "bailian"]
        
        text = None
        for provider in providers:
            if provider == "ollama" and ollama_available:
                try:
                    text = _try_generate(self.client, OPENAI_MODEL)
                    if text:
                        break
                except Exception:
                    pass
            elif provider == "bailian" and bailian_available:
                try:
                    text = _try_generate(self._llm_fallback_client, BAILIAN_MODEL)
                    if text:
                        break
                except Exception:
                    pass
        
        if not text or text == "<足够>" or "足够" in text:
            return fallback_query
            
        return text

    def _rewrite_query(self, question: str) -> list[str]:
        if not ENABLE_QUERY_REWRITE:
            self._last_trace["rewrite_used"] = False
            self._last_trace["rewrite_queries"] = [question]
            return [question]

        if not self.client and not self._llm_fallback_client:
            self._last_trace["rewrite_used"] = False
            self._last_trace["rewrite_queries"] = [question]
            return [question]
        
        system_prompt = (
            "你是一个保守的中文搜索查询改写器，只能帮助本地百科知识库检索。\n"
            "必须遵守：\n"
            "1. 不要回答问题，不要生成 HyDE、假设答案、背景段落或推测事实。\n"
            "2. 每一行都必须保留原问题里的核心实体、专名、时间、地点或限定词。\n"
            "3. 只能做同义词、别名、繁简/中英文名、语序和检索关键词层面的轻微改写。\n"
            "4. 如果没有把握，输出原问题本身。\n"
            "5. 最多输出2行，每行一个短查询，不要编号，不要解释。"
        )
        user_prompt_content = f"原始查询：{question}\n\n请输出最多2行保守改写后的搜索查询："

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]

        def _try_generate(client: Any, model: str) -> str | None:
            resp = client.with_options(timeout=LLM_TIMEOUT_SECONDS, max_retries=0).chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
            )
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            return None

        ollama_available = self.client is not None and time.time() >= getattr(self, "_llm_disabled_until", 0.0)
        bailian_available = self._llm_fallback_client is not None

        providers = ["bailian", "ollama"] if self._llm_primary_provider == "bailian" else ["ollama", "bailian"]
        
        text = None
        for provider in providers:
            if provider == "ollama" and ollama_available:
                try:
                    text = _try_generate(self.client, OPENAI_MODEL)
                    if text:
                        break
                except Exception:
                    pass
            elif provider == "bailian" and bailian_available:
                try:
                    text = _try_generate(self._llm_fallback_client, BAILIAN_MODEL)
                    if text:
                        break
                except Exception:
                    pass
        
        if not text:
            self._last_trace["rewrite_used"] = False
            self._last_trace["rewrite_queries"] = [question]
            return [question]
            
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        results = self._safe_query_variants(question, lines, limit=3)
        self._last_trace["rewrite_used"] = len(results) > 1
        self._last_trace["rewrite_queries"] = list(results)
        return results

    def _rerank_refs(self, question: str, refs: list[dict]) -> list[dict]:
        started_at = time.perf_counter()
        if not refs:
            return refs
        if not ENABLE_RERANKER:
            return refs

        pool = refs[: max(1, min(RERANK_TOP_N, len(refs)))]
        docs = [str(r.get("full_text", ""))[:GEN_CONTEXT_CHARS] for r in pool]
        if not any(docs):
            return refs

        if self.local_rerank_model is not None:
            # 采用本地 CrossEncoder 模型
            scores = self.local_rerank_model.predict([(question, doc) for doc in docs])
            pairs = list(zip(pool, scores))
            pairs.sort(key=lambda x: x[1], reverse=True)
            reranked = [dict(r, score=round(float(s), 6)) for r, s in pairs]
            tail = refs[len(pool) :]
            self._last_retrieval_stats["rerank_used"] = True
            self._record_timing("rerank_ms", started_at)
            return reranked + tail

        # 兜底：如果本地模型没加载但用户非要用 embedding 计算相似度重排（一般效果较差）
        if not self.client or not OPENAI_RERANK_MODEL:
            return refs

        # 使用独立 rerank 模型计算 query-doc 相似度，作为可选重排阶段。
        try:
            q_resp = self.client.with_options(timeout=EMBED_TIMEOUT_SECONDS, max_retries=0).embeddings.create(
                model=OPENAI_RERANK_MODEL,
                input=[question],
            )
            d_resp = self.client.with_options(timeout=EMBED_TIMEOUT_SECONDS, max_retries=0).embeddings.create(
                model=OPENAI_RERANK_MODEL,
                input=docs,
            )
        except Exception:
            return refs

        qv = np.asarray(q_resp.data[0].embedding, dtype=np.float32)
        dv = np.asarray([x.embedding for x in sorted(d_resp.data, key=lambda x: x.index)], dtype=np.float32)
        if dv.size == 0:
            return refs

        qn = np.linalg.norm(qv)
        if qn == 0:
            qn = 1.0
        qv = qv / qn

        dvn = np.linalg.norm(dv, axis=1, keepdims=True)
        dvn[dvn == 0] = 1.0
        dv = dv / dvn

        scores = (dv @ qv).tolist()
        pairs = list(zip(pool, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        reranked = [dict(r, score=round(float(s), 6)) for r, s in pairs]
        tail = refs[len(pool) :]
        self._last_retrieval_stats["rerank_used"] = True
        self._record_timing("rerank_ms", started_at)
        return reranked + tail

    def retrieve(self, question: str):
        retrieve_started_at = time.perf_counter()
        rewrite_started_at = time.perf_counter()
        queries = self._rewrite_query(question)
        self._record_timing("rewrite_ms", rewrite_started_at)
        
        all_refs = []
        total_bm25 = 0
        total_vec = 0
        vec_used = False

        hybrid_started_at = time.perf_counter()
        for q in queries:
            hits = self._hybrid_retrieve(q)
            
            # 聚合多查询的统计信息以免仅保留最后一次的值
            total_bm25 += self._last_retrieval_stats.get("bm25_hits", 0)
            total_vec += self._last_retrieval_stats.get("vec_hits", 0)
            vec_used = vec_used or self._last_retrieval_stats.get("vec_used", False)
            
            q_refs = self._hits_to_refs(hits)
            all_refs = self._merge_refs(all_refs, q_refs)
        self._record_timing("hybrid_ms", hybrid_started_at)
            
        self._last_retrieval_stats["bm25_hits"] = total_bm25
        self._last_retrieval_stats["vec_hits"] = total_vec
        self._last_retrieval_stats["vec_used"] = vec_used
        self._last_retrieval_stats["rerank_used"] = False # 重置重排标志，稍后决定
        
        refs = all_refs
        anchor_started_at = time.perf_counter()
        rescue_hits = self._anchor_rescue_hits(question, refs)
        if rescue_hits:
            refs = self._merge_refs(self._hits_to_refs(rescue_hits), refs)
        else:
            self._last_trace["anchor_rescue_count"] = 0
            self._last_trace["anchor_rescue_used"] = False
        self._record_timing("anchor_rescue_ms", anchor_started_at)

        if ENABLE_MULTI_HOP and MULTI_HOP_MAX_HOPS > 1 and refs:
            query2 = self._build_followup_query(question, refs)
            if query2.strip() and query2.strip() != question.strip():
                prev_bm25 = self._last_retrieval_stats.get("bm25_hits", 0)
                prev_vec = self._last_retrieval_stats.get("vec_hits", 0)
                prev_vec_used = self._last_retrieval_stats.get("vec_used", False)
                
                hop2_hits = self._hybrid_retrieve(query2)
                
                self._last_retrieval_stats["bm25_hits"] += prev_bm25
                self._last_retrieval_stats["vec_hits"] += prev_vec
                self._last_retrieval_stats["vec_used"] = self._last_retrieval_stats["vec_used"] or prev_vec_used
                
                hop2_refs = self._hits_to_refs(hop2_hits)
                refs = self._merge_refs(refs, hop2_refs)

        refs = self._prioritize_anchor_refs(question, refs)
        refs = self._rerank_refs(question, refs)
        refs = self._prioritize_anchor_refs(question, refs)
        refs = refs[:TOP_K_FINAL]
        for r in refs:
            r.pop("_idx", None)
        self._record_timing("retrieve_total_ms", retrieve_started_at)
        return refs

    def _extractive_answer(self, refs: list[dict]) -> str:
        if not refs:
            return "未检索到可用证据，当前仅基于 wiki-cn 回答。"
        text = refs[0].get("full_text", "")
        if not text:
            return "未检索到可用证据，当前仅基于 wiki-cn 回答。"

        for sep in ["。", "！", "？", "\n"]:
            if sep in text:
                parts = [p.strip() for p in text.split(sep) if p.strip()]
                if parts:
                    if len(parts) >= 2:
                        return f"{parts[0]}。{parts[1]}。"
                    return f"{parts[0]}。"
        return text[:180]

    def _should_try_web_fallback(self, refs: list[dict], allow_web: bool | None = None) -> bool:
        if allow_web is True:
            return True
        if allow_web is False:
            return False
        if not ALLOW_WEB_FALLBACK:
            return False
        if not refs:
            return True
        usable_refs = [
            r for r in refs
            if str(r.get("full_text") or r.get("snippet") or "").strip()
        ]
        return len(usable_refs) < max(1, WEB_FALLBACK_MIN_LOCAL_REFS)

    def _web_search(self, query: str) -> list[dict]:
        started_at = time.perf_counter()
        data: dict[str, Any] | None = None

        if SERPAPI_API_KEY:
            self._last_trace["web_provider"] = "serpapi"
            params = {
                "engine": SERPAPI_ENGINE or "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "gl": SERPAPI_GL or "cn",
                "hl": SERPAPI_HL or "zh-cn",
            }

            try:
                if _SerpApiClient is not None:
                    client = _SerpApiClient(params)
                    data = client.get_dict()
                elif _SerpApiGoogleSearch is not None:
                    client = _SerpApiGoogleSearch(params)
                    data = client.get_dict()
                else:
                    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "WikiCN-QA/0.1 (+serpapi fallback)"},
                        method="GET",
                    )
                    with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
                        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            except Exception:
                data = None

        if not data:
            self._last_trace["web_provider"] = "duckduckgo"
            encoded_q = urllib.parse.quote(query)
            url = (
                "https://api.duckduckgo.com/?"
                f"q={encoded_q}&format=json&no_html=1&skip_disambig=1"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "WikiCN-QA/0.1 (+local web fallback)"},
                method="GET",
            )

            try:
                with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            except Exception:
                self._last_trace["web_ref_count"] = 0
                self._record_timing("web_ms", started_at)
                return []

        refs: list[dict] = []
        answer_box = data.get("answer_box") if isinstance(data, dict) else None
        knowledge_graph = data.get("knowledge_graph") if isinstance(data, dict) else None

        if isinstance(answer_box, dict):
            answer = str(answer_box.get("answer", "")).strip()
            if answer:
                refs.append(
                    {
                        "doc_id": "web:answer_box",
                        "title": str(answer_box.get("title", "Google")) or "Google",
                        "url": str(answer_box.get("link", "")).strip(),
                        "score": 0.12,
                        "snippet": answer[:220],
                        "full_text": answer,
                    }
                )

        if isinstance(answer_box, dict):
            answer_box_list = answer_box.get("answer_box_list")
            if isinstance(answer_box_list, list):
                joined = "\n".join(str(x).strip() for x in answer_box_list if str(x).strip())
                if joined:
                    refs.append(
                        {
                            "doc_id": "web:answer_box_list",
                            "title": str(answer_box.get("title", "Google")) or "Google",
                            "url": str(answer_box.get("link", "")).strip(),
                            "score": 0.11,
                            "snippet": joined[:220],
                            "full_text": joined,
                        }
                    )

        if isinstance(data, dict) and ("AbstractText" in data or "RelatedTopics" in data):
            abstract = str(data.get("AbstractText", "")).strip()
            abstract_url = str(data.get("AbstractURL", "")).strip()
            heading = str(data.get("Heading", "")).strip()
            if abstract:
                refs.append(
                    {
                        "doc_id": "web:abstract",
                        "title": heading or "Web",
                        "url": abstract_url,
                        "score": 0.08,
                        "snippet": abstract[:220],
                        "full_text": abstract,
                    }
                )

            related = data.get("RelatedTopics", [])
            for item in related:
                if len(refs) >= WEB_SEARCH_MAX_RESULTS:
                    break
                if not isinstance(item, dict):
                    continue

                rows = [item]
                if "Topics" in item and isinstance(item.get("Topics"), list):
                    rows = item.get("Topics", [])

                for row in rows:
                    if len(refs) >= WEB_SEARCH_MAX_RESULTS:
                        break
                    if not isinstance(row, dict):
                        continue
                    text = str(row.get("Text", "")).strip()
                    first_url = str(row.get("FirstURL", "")).strip()
                    if not text:
                        continue
                    refs.append(
                        {
                            "doc_id": f"web:{len(refs)+1}",
                            "title": "Web",
                            "url": first_url,
                            "score": 0.05,
                            "snippet": text[:220],
                            "full_text": text,
                        }
                    )

        if isinstance(knowledge_graph, dict):
            description = str(knowledge_graph.get("description", "")).strip()
            title = str(knowledge_graph.get("title", "Google")) or "Google"
            link = str(knowledge_graph.get("website", "")).strip() or str(knowledge_graph.get("link", "")).strip()
            if description:
                refs.append(
                    {
                        "doc_id": "web:knowledge_graph",
                        "title": title,
                        "url": link,
                        "score": 0.10,
                        "snippet": description[:220],
                        "full_text": description,
                    }
                )

        organic_results = data.get("organic_results", []) if isinstance(data, dict) else []
        for item in organic_results[:WEB_SEARCH_MAX_RESULTS]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "Web")).strip() or "Web"
            snippet = str(item.get("snippet", "")).strip()
            link = str(item.get("link", "")).strip()
            if not snippet and not title:
                continue
            refs.append(
                {
                    "doc_id": f"web:{len(refs)+1}",
                    "title": title,
                    "url": link,
                    "score": 0.05,
                    "snippet": snippet[:220] if snippet else title[:220],
                    "full_text": snippet or title,
                }
            )

        refs = refs[:WEB_SEARCH_MAX_RESULTS]
        self._last_trace["web_ref_count"] = len(refs)
        self._record_timing("web_ms", started_at)
        return refs

    def _build_answer_messages(self, question: str, refs: list[dict], history: list[dict] | None = None) -> list[dict[str, str]]:
        context_blocks = []
        use_refs = refs[: max(1, min(GEN_CONTEXT_TOP_N, len(refs)))]
        for i, r in enumerate(use_refs, start=1):
            context_blocks.append(
                f"[证据{i}] 标题：{r.get('title','')}\n"
                f"URL：{r.get('url','')}\n"
                f"片段：{r.get('full_text','')[:GEN_CONTEXT_CHARS]}"
            )

        system_prompt = (
            "你是一个仅基于给定知识库证据回答的助手。"
            "禁止编造知识库外信息。"
            "只要证据中包含与问题实体直接相关的信息，就应基于证据给出答案。"
            "只有在证据为空或所有证据都与问题实体无关时，才说明‘证据不足’。"
            + ANSWER_STYLE
        )
        joined_context = "\n\n".join(context_blocks)
        user_prompt_content = (
            f"问题：{question}\n\n"
            f"证据：\n{joined_context}\n\n"
            "请输出简洁事实型答案。不要输出 Markdown 标题符号 # 或 **加粗标记**。"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt_content})
        return messages

    def _llm_provider_candidates(self) -> list[tuple[str, OpenAI, str]]:
        providers: list[tuple[str, OpenAI, str]] = []
        ollama_available = self.client is not None and time.time() >= self._llm_disabled_until
        bailian_available = self._llm_fallback_client is not None
        provider_order = ["bailian", "ollama"] if self._llm_primary_provider == "bailian" else ["ollama", "bailian"]
        for provider in provider_order:
            if provider == "ollama" and ollama_available and self.client is not None:
                providers.append(("ollama", self.client, OPENAI_MODEL))
            elif provider == "bailian" and bailian_available and self._llm_fallback_client is not None:
                providers.append(("bailian", self._llm_fallback_client, BAILIAN_MODEL))
        return providers

    def _mark_llm_success(self, provider: str, model: str) -> None:
        if provider == "ollama":
            self._llm_fail_count = 0
        self._llm_provider_last = provider
        self._last_trace["gen_provider"] = provider
        self._last_trace["gen_model"] = model

    def _mark_llm_failure(self, provider: str) -> None:
        if provider != "ollama":
            return
        self._llm_fail_count += 1
        if self._llm_fail_count >= LLM_FAILURE_THRESHOLD:
            self._llm_disabled_until = time.time() + LLM_COOLDOWN_SECONDS
            self._llm_fail_count = 0

    def _try_generate_text(self, client: OpenAI, model: str, messages: list[dict[str, str]]) -> str | None:
        resp = client.with_options(timeout=LLM_TIMEOUT_SECONDS, max_retries=0).chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS,
        )
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
        return None

    def _stream_generate_text(self, client: OpenAI, model: str, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = client.with_options(timeout=LLM_TIMEOUT_SECONDS, max_retries=0).chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        yield text

    def _llm_answer(self, question: str, refs: list[dict], history: list[dict] = None) -> str:
        started_at = time.perf_counter()
        if not self.client and not self._llm_fallback_client:
            self._llm_provider_last = "extractive"
            self._last_trace["gen_provider"] = "extractive"
            self._last_trace["gen_model"] = ""
            self._record_timing("llm_ms", started_at)
            return self._extractive_answer(refs)

        messages = self._build_answer_messages(question, refs, history=history)
        for provider, client, model in self._llm_provider_candidates():
            try:
                text = self._try_generate_text(client, model, messages)
                if text:
                    self._mark_llm_success(provider, model)
                    self._record_timing("llm_ms", started_at)
                    return text
            except Exception:
                self._mark_llm_failure(provider)

        self._llm_provider_last = "extractive"
        self._last_trace["gen_provider"] = "extractive"
        self._last_trace["gen_model"] = ""
        self._record_timing("llm_ms", started_at)
        return self._extractive_answer(refs)

    def stream_answer(
        self,
        question: str,
        history: list[dict] | None = None,
        allow_web: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        ask_started_at = time.perf_counter()
        self._last_trace["web_used"] = False
        self._last_trace["web_ref_count"] = 0
        self._last_trace["timing_ms"] = {}

        refs = self.retrieve(question)
        if self._should_try_web_fallback(refs, allow_web=allow_web):
            web_refs = self._web_search(question)
            if web_refs:
                refs = web_refs
                self._last_trace["web_used"] = True

        llm_started_at = time.perf_counter()
        if not self.client and not self._llm_fallback_client:
            answer = self._extractive_answer(refs)
            display_refs = self._filter_public_refs(question, refs, answer=answer)
            public_refs = []
            for r in display_refs:
                x = dict(r)
                x.pop("full_text", None)
                public_refs.append(x)
            self._llm_provider_last = "extractive"
            self._last_trace["gen_provider"] = "extractive"
            self._last_trace["gen_model"] = ""
            self._record_timing("llm_ms", llm_started_at)
            self._record_timing("ask_total_ms", ask_started_at)
            yield {"type": "chunk", "delta": answer}
            yield {"type": "done", "answer": answer, "references": public_refs, "trace": self.get_last_trace()}
            return

        messages = self._build_answer_messages(question, refs, history=history)
        for provider, client, model in self._llm_provider_candidates():
            answer_parts: list[str] = []
            try:
                for text in self._stream_generate_text(client, model, messages):
                    answer_parts.append(text)
                    yield {"type": "chunk", "delta": text}
                answer = "".join(answer_parts).strip()
                if answer:
                    display_refs = self._filter_public_refs(question, refs, answer=answer)
                    public_refs = []
                    for r in display_refs:
                        x = dict(r)
                        x.pop("full_text", None)
                        public_refs.append(x)
                    self._mark_llm_success(provider, model)
                    self._record_timing("llm_ms", llm_started_at)
                    self._record_timing("ask_total_ms", ask_started_at)
                    yield {"type": "done", "answer": answer, "references": public_refs, "trace": self.get_last_trace()}
                    return
            except Exception as e:
                self._mark_llm_failure(provider)
                yield {"type": "provider_error", "provider": provider, "message": str(e)}

        answer = self._extractive_answer(refs)
        display_refs = self._filter_public_refs(question, refs, answer=answer)
        public_refs = []
        for r in display_refs:
            x = dict(r)
            x.pop("full_text", None)
            public_refs.append(x)
        self._llm_provider_last = "extractive"
        self._last_trace["gen_provider"] = "extractive"
        self._last_trace["gen_model"] = ""
        self._record_timing("llm_ms", llm_started_at)
        self._record_timing("ask_total_ms", ask_started_at)
        yield {"type": "chunk", "delta": answer}
        yield {"type": "done", "answer": answer, "references": public_refs, "trace": self.get_last_trace()}

    def ask(self, question: str, history: list[dict] = None, allow_web: bool | None = None):
        ask_started_at = time.perf_counter()
        self._last_trace["web_used"] = False
        self._last_trace["web_ref_count"] = 0
        self._last_trace["timing_ms"] = {}
        refs = self.retrieve(question)
        if self._should_try_web_fallback(refs, allow_web=allow_web):
            web_refs = self._web_search(question)
            if web_refs:
                refs = web_refs
                self._last_trace["web_used"] = True
        answer = self._llm_answer(question, refs, history=history)
        display_refs = self._filter_public_refs(question, refs, answer=answer)
        self._record_timing("ask_total_ms", ask_started_at)

        public_refs = []
        for r in display_refs:
            x = dict(r)
            x.pop("full_text", None)
            public_refs.append(x)
        return answer, public_refs

    def ask_with_full_refs(
        self,
        question: str,
        history: list[dict] | None = None,
        allow_web: bool | None = None,
    ) -> tuple[str, list[dict], list[dict]]:
        """Return answer, full refs (with full_text), and public refs."""
        ask_started_at = time.perf_counter()
        self._last_trace["web_used"] = False
        self._last_trace["web_ref_count"] = 0
        self._last_trace["timing_ms"] = {}
        refs = self.retrieve(question)
        if self._should_try_web_fallback(refs, allow_web=allow_web):
            web_refs = self._web_search(question)
            if web_refs:
                refs = web_refs
                self._last_trace["web_used"] = True

        answer = self._llm_answer(question, refs, history=history)
        display_refs = self._filter_public_refs(question, refs, answer=answer)
        self._record_timing("ask_total_ms", ask_started_at)

        public_refs = []
        for r in display_refs:
            x = dict(r)
            x.pop("full_text", None)
            public_refs.append(x)
        return answer, display_refs, public_refs

    def get_last_trace(self) -> dict:
        stats = dict(self._last_retrieval_stats)
        trace = dict(self._last_trace)
        trace.update(
            {
                "bm25_hits": int(stats.get("bm25_hits", 0)),
                "vec_hits": int(stats.get("vec_hits", 0)),
                "vec_used": bool(stats.get("vec_used", False)),
                "rerank_used": bool(stats.get("rerank_used", False)),
                "llm_provider_last": self._llm_provider_last,
                "dense_backend": self._dense_backend,
                "llm_primary_provider": self._llm_primary_provider,
                "query_rewrite_enabled": ENABLE_QUERY_REWRITE,
                "multi_hop_enabled": ENABLE_MULTI_HOP,
                "rerank_top_n": RERANK_TOP_N,
                "web_fallback_min_local_refs": WEB_FALLBACK_MIN_LOCAL_REFS,
            }
        )
        return trace

    def health_status(self) -> dict:
        now = time.time()
        embed_until = float(getattr(self, "_embed_disabled_until", 0.0))
        llm_until = float(getattr(self, "_llm_disabled_until", 0.0))
        vector_left = max(0.0, embed_until - now)
        llm_left = max(0.0, llm_until - now)

        embed_ready = self.embed_client is not None or self.local_embed_model is not None
        ollama_ready = self.client is not None
        bailian_ready = self._llm_fallback_client is not None
        primary_ready = bailian_ready if self._llm_primary_provider == "bailian" else ollama_ready
        fallback_ready = ollama_ready if self._llm_primary_provider == "bailian" else bailian_ready
        client_ready = ollama_ready or bailian_ready
        has_dense = self._chroma_collection is not None or self.index is not None
        index_ntotal = self._get_index_ntotal()
        dense_index_count_ok = index_ntotal > 0
        dense_ready = has_dense and dense_index_count_ok
        vector_enabled = embed_ready and dense_ready and vector_left <= 0.0
        llm_enabled = (primary_ready and llm_left <= 0.0) or fallback_ready

        if client_ready and vector_enabled and llm_enabled:
            status = "ok"
        elif client_ready:
            status = "degraded"
        else:
            status = "bm25-only"

        if self._dense_backend == "none":
            dense_module_status = "skipped"
        elif dense_ready and vector_left <= 0.0 and self._dense_disabled_reason == "":
            dense_module_status = "ready"
        else:
            dense_module_status = "failed"

        module_statuses = {
            "chunks": "ready" if len(self.chunks) > 0 else "failed",
            "bm25": "ready" if self.bm25 is not None else "failed",
            "dense": dense_module_status,
            "rerank": "ready" if ENABLE_RERANKER and (self.local_rerank_model is not None or (self.client and OPENAI_RERANK_MODEL)) else "disabled",
            "llm_primary": "ready" if primary_ready else "failed",
            "llm_fallback": "ready" if fallback_ready else "disabled",
            "web_fallback": "enabled" if ALLOW_WEB_FALLBACK else "disabled",
        }

        return {
            "status": status,
            "client_ready": client_ready,
            "vector_enabled": vector_enabled,
            "llm_enabled": llm_enabled,
            "dense_backend": self._dense_backend,
            "chroma_persist_dir": str(CHROMA_PERSIST_DIR),
            "chroma_collection": CHROMA_COLLECTION,
            "force_bm25_only": FORCE_BM25_ONLY,
            "dense_ready": dense_ready,
            "dense_disabled_reason": self._dense_disabled_reason,
            "llm_primary_provider": self._llm_primary_provider,
            "llm_primary_model": BAILIAN_MODEL if self._llm_primary_provider == "bailian" else OPENAI_MODEL,
            "llm_fallback_provider": "ollama" if self._llm_primary_provider == "bailian" else "bailian",
            "llm_fallback_model": OPENAI_MODEL if self._llm_primary_provider == "bailian" else BAILIAN_MODEL,
            "llm_fallback_enabled": bool(fallback_ready),
            "llm_provider_last": self._llm_provider_last,
            "local_embed_device": self._local_embed_device or ("remote_api" if self.embed_client is not None else ""),
            "local_rerank_device": self._local_rerank_device,
            "web_fallback_enabled": ALLOW_WEB_FALLBACK,
            "vector_cooldown_left_sec": round(vector_left, 2),
            "llm_cooldown_left_sec": round(llm_left, 2),
            "total_chunks": len(self.chunks),
            "index_ntotal": index_ntotal,
            "dense_index_count_ok": dense_index_count_ok,
            "last_bm25_hits": int(self._last_retrieval_stats.get("bm25_hits", 0)),
            "last_vec_hits": int(self._last_retrieval_stats.get("vec_hits", 0)),
            "last_vec_used": bool(self._last_retrieval_stats.get("vec_used", False)),
            "last_rerank_used": bool(self._last_retrieval_stats.get("rerank_used", False)),
            "last_trace": self.get_last_trace(),
            "module_statuses": module_statuses,
        }
