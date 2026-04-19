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
from typing import Callable

import faiss
import jieba
import numpy as np
from openai import OpenAI

MilvusClient = None
try:
    _pm = importlib.import_module("pymilvus")
    MilvusClient = getattr(_pm, "MilvusClient", None)
except Exception:
    MilvusClient = None

from .config import (
    ALLOW_WEB_FALLBACK,
    ANSWER_STYLE,
    BM25_PATH,
    CHUNKS_PATH,
    EMBED_COOLDOWN_SECONDS,
    EMBED_FAILURE_THRESHOLD,
    EMBED_TIMEOUT_SECONDS,
    ENABLE_LLM_FALLBACK,
    LLM_PRIMARY_PROVIDER,
    DENSE_BACKEND,
    ENABLE_MULTI_HOP,
    ENABLE_RERANKER,
    FAISS_PATH,
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
    ZILLIZ_COLLECTION,
    ZILLIZ_NPROBE,
    ZILLIZ_TOKEN,
    ZILLIZ_URI,
)


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


class RagEngine:
    def __init__(self, on_stage: Callable[[str], None] | None = None) -> None:
        self._on_stage = on_stage
        self._index_error = ""
        self._dense_backend = DENSE_BACKEND
        self._dense_disabled_reason = ""
        self._zilliz = None
        self.index = None
        self._last_retrieval_stats = {
            "bm25_hits": 0,
            "vec_hits": 0,
            "vec_used": False,
        }
        self._emit_stage("loading_chunks")
        self.chunks = self._load_chunks(CHUNKS_PATH)

        self._emit_stage("loading_bm25")
        self.bm25 = self._load_pickle(BM25_PATH)

        if FORCE_BM25_ONLY:
            self._emit_stage("skipping_faiss")
            self.index = None
            self._index_error = "dense retrieval disabled"
            self._dense_disabled_reason = "FORCE_BM25_ONLY=1"
        elif self._dense_backend == "none":
            self._emit_stage("skipping_faiss")
            self.index = None
            self._index_error = "dense retrieval disabled"
            self._dense_disabled_reason = "DENSE_BACKEND=none"
        elif self._dense_backend == "zilliz":
            self._emit_stage("connecting_zilliz")
            if MilvusClient is None:
                self._index_error = "pymilvus not installed"
                self._dense_disabled_reason = "pymilvus not installed"
            elif not ZILLIZ_URI or not ZILLIZ_TOKEN:
                self._index_error = "missing ZILLIZ_URI/ZILLIZ_TOKEN"
                self._dense_disabled_reason = "missing ZILLIZ_URI/ZILLIZ_TOKEN"
            else:
                try:
                    self._zilliz = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
                except Exception as e:
                    self._index_error = str(e)
                    self._dense_disabled_reason = self._index_error
        else:
            self._emit_stage("loading_faiss")
            try:
                # read_index avoids an extra in-memory raw-bytes copy for large indexes.
                self.index = faiss.read_index(str(FAISS_PATH))
            except Exception as e:
                # 大索引在内存不足时可能触发 bad_alloc，降级到 BM25-only 以保证服务可用。
                self.index = None
                self._index_error = str(e)
                self._dense_disabled_reason = self._index_error

        self._emit_stage("initializing_client")
        self.client = None
        self._llm_fallback_client = None
        self.embed_client = None
        self.local_embed_model = None
        self.local_rerank_model = None

        if ENABLE_RERANKER and LOCAL_RERANK_MODEL_PATH:
            try:
                from sentence_transformers import CrossEncoder
                self.local_rerank_model = CrossEncoder(LOCAL_RERANK_MODEL_PATH)
            except ImportError:
                print("未安装 sentence-transformers，或者模型路径不存在，请检查配置")
                pass

        if LOCAL_EMBED_MODEL_PATH:
            # 如果配置了本地下载的 HuggingFace 模型路径，则使用 sentence-transformers 直接加载入内存
            try:
                from sentence_transformers import SentenceTransformer
                self.local_embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
            except ImportError:
                print("未安装 sentence-transformers，请执行 pip install sentence-transformers")
                raise
        else:
            # 否则回退为原本的 OpenAI API 兼容形式 (如 Ollama)
            self.embed_client = OpenAI(
                api_key="ollama",
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
        return [w for w in jieba.lcut(text) if w.strip()]

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

    def _bm25_retrieve(self, q: str) -> list[ScoredChunk]:
        nq = self._normalize_query(q)
        tokens = [t for t in self._tokenize(nq) if t not in ZH_QUERY_STOPWORDS and t.strip()]
        if not tokens:
            tokens = self._tokenize(q)

        scores = self.bm25.get_scores(tokens)
        # 标题命中轻量加权，降低“什么是/如何”等问句模板对召回的干扰。
        for i in range(len(scores)):
            title = str(self.chunks[i].get("title", ""))
            scores[i] = float(scores[i]) + self._title_overlap_boost(title, tokens)

        idx = np.argsort(scores)[::-1][:TOP_K_BM25]
        return [ScoredChunk(int(i), float(scores[i])) for i in idx]

    def _vec_retrieve(self, q: str) -> list[ScoredChunk]:
        if not self.embed_client and not self.local_embed_model:
            return []
        if time.time() < self._embed_disabled_until:
            return []

        if self._dense_backend == "zilliz":
            return self._vec_retrieve_zilliz(q)
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
            return out
        except Exception as e:
            self._dense_disabled_reason = f"Embedding failure: {e!r}"
            self._embed_fail_count += 1
            if self._embed_fail_count >= EMBED_FAILURE_THRESHOLD:
                self._embed_disabled_until = time.time() + EMBED_COOLDOWN_SECONDS
                self._embed_fail_count = 0
            return []

    def _vec_retrieve_zilliz(self, q: str) -> list[ScoredChunk]:
        if self._zilliz is None:
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

            res = self._zilliz.search(
                collection_name=ZILLIZ_COLLECTION,
                data=[qv],
                limit=TOP_K_VECTOR,
                search_params={"metric_type": "IP", "params": {"nprobe": ZILLIZ_NPROBE}},
                output_fields=[],
            )

            out: list[ScoredChunk] = []
            hits = res[0] if isinstance(res, list) and res else []
            for h in hits:
                idx = h.get("id", h.get("pk", -1)) if isinstance(h, dict) else -1
                score = h.get("distance", h.get("score", 0.0)) if isinstance(h, dict) else 0.0
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < len(self.chunks):
                    out.append(ScoredChunk(idx=i, score=float(score)))
            self._embed_fail_count = 0
            self._dense_disabled_reason = ""
            return out
        except Exception as e:
            self._dense_disabled_reason = f"Zilliz/Embedding failure: {e!r}"
            self._embed_fail_count += 1
            if self._embed_fail_count >= EMBED_FAILURE_THRESHOLD:
                self._embed_disabled_until = time.time() + EMBED_COOLDOWN_SECONDS
                self._embed_fail_count = 0
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

    @staticmethod
    def _build_followup_query(question: str, refs: list[dict]) -> str:
        terms = []
        for r in refs:
            t = str(r.get("title", "")).strip()
            if t and t not in terms:
                terms.append(t)
            if len(terms) >= max(1, MULTI_HOP_EXPAND_TITLES):
                break
        if not terms:
            return question
        return f"{question} {' '.join(terms)}"

    def _rerank_refs(self, question: str, refs: list[dict]) -> list[dict]:
        if not refs:
            return refs
        if not ENABLE_RERANKER:
            return refs

        pool = refs[: max(1, min(RERANK_TOP_N, len(refs)))]
        docs = [r.get("full_text", "") for r in pool]
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
        return reranked + tail

    def retrieve(self, question: str):
        hits = self._hybrid_retrieve(question)
        refs = self._hits_to_refs(hits)

        if ENABLE_MULTI_HOP and MULTI_HOP_MAX_HOPS > 1 and refs:
            query2 = self._build_followup_query(question, refs)
            if query2.strip() and query2.strip() != question.strip():
                hop2_hits = self._hybrid_retrieve(query2)
                hop2_refs = self._hits_to_refs(hop2_hits)
                refs = self._merge_refs(refs, hop2_refs)

        refs = self._rerank_refs(question, refs)
        refs = refs[:TOP_K_FINAL]
        for r in refs:
            r.pop("_idx", None)
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
        web_enabled = ALLOW_WEB_FALLBACK if allow_web is None else bool(allow_web)
        if not web_enabled:
            return False
        if not refs:
            return True
        top_score = float(refs[0].get("score", 0.0))
        return top_score < 0.1

    def _web_search(self, query: str) -> list[dict]:
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
            return []

        refs: list[dict] = []
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

        return refs[:WEB_SEARCH_MAX_RESULTS]

    def _llm_answer(self, question: str, refs: list[dict], history: list[dict] = None) -> str:
        if not self.client and not self._llm_fallback_client:
            self._llm_provider_last = "extractive"
            return self._extractive_answer(refs)

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
            "若证据不足，请明确说明‘证据不足’。"
            + ANSWER_STYLE
        )
        joined_context = "\n\n".join(context_blocks)
        user_prompt_content = (
            f"问题：{question}\n\n"
            f"证据：\n{joined_context}\n\n"
            "请输出简洁事实型答案。"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt_content})

        def _try_generate(client: OpenAI, model: str) -> str | None:
            resp = client.with_options(timeout=LLM_TIMEOUT_SECONDS, max_retries=0).chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=LLM_MAX_TOKENS,
            )
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            return None

        ollama_available = self.client is not None and time.time() >= self._llm_disabled_until
        bailian_available = self._llm_fallback_client is not None

        providers: list[str]
        if self._llm_primary_provider == "bailian":
            providers = ["bailian", "ollama"]
        else:
            providers = ["ollama", "bailian"]

        for provider in providers:
            if provider == "ollama" and ollama_available:
                try:
                    text = _try_generate(self.client, OPENAI_MODEL)
                    if text:
                        self._llm_fail_count = 0
                        self._llm_provider_last = "ollama"
                        return text
                except Exception:
                    self._llm_fail_count += 1
                    if self._llm_fail_count >= LLM_FAILURE_THRESHOLD:
                        self._llm_disabled_until = time.time() + LLM_COOLDOWN_SECONDS
                        self._llm_fail_count = 0
            elif provider == "bailian" and bailian_available:
                try:
                    text = _try_generate(self._llm_fallback_client, BAILIAN_MODEL)
                    if text:
                        self._llm_provider_last = "bailian"
                        return text
                except Exception:
                    pass

        self._llm_provider_last = "extractive"
        return self._extractive_answer(refs)

    def ask(self, question: str, history: list[dict] = None, allow_web: bool | None = None):
        refs = self.retrieve(question)
        if self._should_try_web_fallback(refs, allow_web=allow_web):
            web_refs = self._web_search(question)
            if web_refs:
                refs = web_refs
        answer = self._llm_answer(question, refs, history=history)

        public_refs = []
        for r in refs:
            x = dict(r)
            x.pop("full_text", None)
            public_refs.append(x)
        return answer, public_refs

    def health_status(self) -> dict:
        now = time.time()
        embed_until = float(getattr(self, "_embed_disabled_until", 0.0))
        llm_until = float(getattr(self, "_llm_disabled_until", 0.0))
        vector_left = max(0.0, embed_until - now)
        llm_left = max(0.0, llm_until - now)

        embed_ready = self.embed_client is not None or self.local_embed_model is not None
        primary_ready = self.client is not None
        fallback_ready = self._llm_fallback_client is not None
        client_ready = primary_ready or fallback_ready
        has_dense = self.index is not None or self._zilliz is not None
        vector_enabled = embed_ready and has_dense and vector_left <= 0.0
        llm_enabled = (primary_ready and llm_left <= 0.0) or fallback_ready

        if client_ready and vector_enabled and llm_enabled:
            status = "ok"
        elif client_ready:
            status = "degraded"
        else:
            status = "bm25-only"

        if self._dense_backend == "none":
            dense_module_status = "skipped"
        elif has_dense and vector_left <= 0.0 and self._dense_disabled_reason == "":
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
            "force_bm25_only": FORCE_BM25_ONLY,
            "dense_ready": has_dense,
            "dense_disabled_reason": self._dense_disabled_reason,
            "llm_primary_provider": self._llm_primary_provider,
            "llm_fallback_enabled": bool(fallback_ready),
            "llm_provider_last": self._llm_provider_last,
            "web_fallback_enabled": ALLOW_WEB_FALLBACK,
            "vector_cooldown_left_sec": round(vector_left, 2),
            "llm_cooldown_left_sec": round(llm_left, 2),
            "total_chunks": len(self.chunks),
            "index_ntotal": int(self.index.ntotal) if self.index is not None else 0,
            "last_bm25_hits": int(self._last_retrieval_stats.get("bm25_hits", 0)),
            "last_vec_hits": int(self._last_retrieval_stats.get("vec_hits", 0)),
            "last_vec_used": bool(self._last_retrieval_stats.get("vec_used", False)),
            "last_rerank_used": bool(self._last_retrieval_stats.get("rerank_used", False)),
            "module_statuses": module_statuses,
        }
