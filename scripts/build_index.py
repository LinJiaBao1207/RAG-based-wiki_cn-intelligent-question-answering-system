from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import jieba
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str


def split_by_heading(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("##") and current:
            blocks.append("\n".join(current).strip())
            current = [s]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
EMBED_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EMBED_MULTI_SPACE_RE = re.compile(r"\s+")
EMBED_SYMBOL_RUN_RE = re.compile(r"([\-_=*#`~^|<>])\1{7,}")


def split_sentences(text: str) -> list[str]:
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if not lines:
        return []

    sentences: list[str] = []
    for ln in lines:
        parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(ln) if p.strip()]
        if parts:
            sentences.extend(parts)
        else:
            sentences.append(ln)
    return sentences


def char_window(text: str, max_len: int, overlap: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


def smart_chunk(text: str, max_len: int = 700, overlap: int = 100, min_len: int = 80) -> list[str]:
    if not text:
        return []

    sents = split_sentences(text)
    if not sents:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for sent in sents:
        s = sent.strip()
        if not s:
            continue

        if len(s) > max_len:
            if cur:
                merged = "".join(cur).strip()
                if len(merged) >= min_len:
                    chunks.append(merged)
                cur = []
                cur_len = 0

            for piece in char_window(s, max_len=max_len, overlap=overlap):
                piece = piece.strip()
                if len(piece) >= min_len:
                    chunks.append(piece)
            continue

        if cur_len + len(s) <= max_len:
            cur.append(s)
            cur_len += len(s)
        else:
            merged = "".join(cur).strip()
            if len(merged) >= min_len:
                chunks.append(merged)

            if overlap > 0 and chunks:
                tail = merged[-overlap:] if merged else ""
                cur = [tail, s] if tail else [s]
                cur_len = sum(len(x) for x in cur)
            else:
                cur = [s]
                cur_len = len(s)

    if cur:
        merged = "".join(cur).strip()
        if len(merged) >= min_len:
            chunks.append(merged)

    return chunks


def sanitize_for_embedding(text: str, max_chars: int) -> str:
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = EMBED_CTRL_RE.sub("", s)
    s = EMBED_SYMBOL_RUN_RE.sub(r"\1\1\1", s)
    s = EMBED_MULTI_SPACE_RE.sub(" ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def load_corpus(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_chunks(corpus_path: Path, max_len: int, overlap: int, min_len: int, progress_every: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    doc_count = 0
    for doc in load_corpus(corpus_path):
        doc_count += 1
        doc_id = doc.get("doc_id", "")
        title = doc.get("title", "")
        url = doc.get("url", "")
        content = doc.get("content", "")
        for block in split_by_heading(content):
            for piece in smart_chunk(block, max_len=max_len, overlap=overlap, min_len=min_len):
                chunk = Chunk(
                    chunk_id=f"c_{idx}",
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    text=piece,
                )
                chunks.append(chunk)
                idx += 1
        if progress_every > 0 and doc_count % progress_every == 0:
            print(f"切块进度：doc={doc_count}, chunk={len(chunks)}", flush=True)
    return chunks


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "title": c.title,
                        "url": c.url,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=str(obj.get("chunk_id", "")),
                    doc_id=str(obj.get("doc_id", "")),
                    title=str(obj.get("title", "")),
                    url=str(obj.get("url", "")),
                    text=str(obj.get("text", "")),
                )
            )
    return chunks


def tokenize_zh(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    # Windows 上 state 文件可能被杀软/索引器短暂占用，做重试避免任务中断。
    last_err: Exception | None = None
    for i in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (i + 1))

    # 回退方案：直接覆盖目标文件；若仍失败再抛出原异常。
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    except Exception:
        if last_err is not None:
            raise last_err
        raise


def load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_chunk_build_signature(corpus_path: Path, max_len: int, overlap: int, min_len: int) -> dict[str, Any]:
    return {
        "corpus_path": str(corpus_path),
        "chunk_size": int(max_len),
        "chunk_overlap": int(overlap),
        "min_chunk_len": int(min_len),
    }


def save_chunk_build_signature(path: Path, signature: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False)


def load_chunk_build_signature(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def signatures_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("corpus_path", "")) == str(b.get("corpus_path", ""))
        and int(a.get("chunk_size", -1)) == int(b.get("chunk_size", -1))
        and int(a.get("chunk_overlap", -1)) == int(b.get("chunk_overlap", -1))
        and int(a.get("min_chunk_len", -1)) == int(b.get("min_chunk_len", -1))
    )


def embed_to_memmap(
    texts: list[str],
    client: OpenAI | None,
    model: str,
    local_embed_model: Any,
    batch_size: int,
    embed_max_chars: int,
    max_retries: int,
    retry_backoff: float,
    resume: bool,
    state_path: Path,
    emb_cache_path: Path,
) -> tuple[Path, int]:
    failures_path = state_path.with_name("embedding_failures.jsonl")

    def infer_dim_via_probe() -> int:
        if local_embed_model is not None:
            return len(local_embed_model.encode(["测试"])[0])
        
        probe = ["测试"]
        for _ in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=probe)
                if resp.data and resp.data[0].embedding:
                    return int(len(resp.data[0].embedding))
            except Exception:
                time.sleep(retry_backoff)
        raise RuntimeError("无法推断 embedding 维度")

    safe_mode_batch1 = False

    def request_with_retry(single_batch: list[str], known_dim: int, batch_start: int) -> np.ndarray:
        last_err: Exception | None = None
        nan_err_count = 0
        # 先做有限次重试；若仍失败且是多条，则拆分为更小批次继续。
        for attempt in range(1, max_retries + 1):
            try:
                if local_embed_model is not None:
                    arr = local_embed_model.encode(single_batch, normalize_embeddings=True)
                    if arr.size == 0:
                        raise RuntimeError("embedding 返回为空")
                    if not np.all(np.isfinite(arr)):
                        raise RuntimeError("embedding 返回包含 NaN/Inf")
                    return np.asarray(arr, dtype=np.float32)

                resp = client.embeddings.create(model=model, input=single_batch)
                rows = sorted(resp.data, key=lambda x: x.index)
                arr = np.asarray([r.embedding for r in rows], dtype=np.float32)
                if arr.size == 0:
                    raise RuntimeError("embedding 返回为空")
                if not np.all(np.isfinite(arr)):
                    raise RuntimeError("embedding 返回包含 NaN/Inf")
                return arr
            except Exception as e:
                last_err = e
                err_text = str(e)
                is_nan_err = "unsupported value: NaN" in err_text or "NaN" in err_text
                if is_nan_err:
                    nan_err_count += 1
                if attempt >= max_retries:
                    break
                if is_nan_err and len(single_batch) > 1:
                    # NaN 常由个别异常文本触发，直接拆分比整批重试更高效。
                    break
                wait_s = retry_backoff * (2 ** (attempt - 1))
                print(
                    f"embedding 请求失败，重试 {attempt}/{max_retries}，batch={len(single_batch)}，等待 {wait_s:.1f}s，错误：{e}",
                    flush=True,
                )
                time.sleep(wait_s)

        if len(single_batch) > 1:
            mid = len(single_batch) // 2
            left = request_with_retry(single_batch[:mid], known_dim, batch_start)
            right = request_with_retry(single_batch[mid:], known_dim, batch_start + mid)
            return np.vstack([left, right])

        use_dim = known_dim
        if use_dim <= 0:
            use_dim = infer_dim_via_probe()

        failed_idx = batch_start
        with failures_path.open("a", encoding="utf-8") as fw:
            fw.write(
                json.dumps(
                    {
                        "idx": failed_idx,
                        "error": str(last_err) if last_err else "unknown",
                        "text_preview": single_batch[0][:200],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"警告：单条文本 embedding 失败 idx={failed_idx}，使用零向量兜底", flush=True)
        return np.zeros((1, use_dim), dtype=np.float32)

    total = len(texts)
    next_idx = 0
    dim = 0

    if resume and state_path.exists() and emb_cache_path.exists():
        state = load_state(state_path)
        if state.get("embed_model") != model:
            raise RuntimeError("续跑状态中的 embedding 模型与当前参数不一致")
        if int(state.get("total", -1)) != total:
            raise RuntimeError("续跑状态中的总 chunk 数与当前语料不一致")
        next_idx = int(state.get("next_idx", 0))
        dim = int(state.get("dim", 0))
        print(f"embedding 续跑：{next_idx}/{total}", flush=True)

    mm = None
    if dim > 0:
        mm = np.memmap(emb_cache_path, dtype=np.float32, mode="r+", shape=(total, dim))

    i = next_idx
    while i < total:
        cur_batch_size = 1 if safe_mode_batch1 else batch_size
        batch = texts[i : i + cur_batch_size]
        batch = [sanitize_for_embedding(x, embed_max_chars) for x in batch]
        if any(not x for x in batch):
            batch = [x if x else "空文本" for x in batch]
        arr = request_with_retry(batch, dim, i)

        # 若这批包含非有限值（理论上已被拦截），切换到安全模式，后续单条请求更稳。
        if not np.all(np.isfinite(arr)):
            safe_mode_batch1 = True
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if dim == 0:
            dim = int(arr.shape[1])
            mm = np.memmap(emb_cache_path, dtype=np.float32, mode="w+", shape=(total, dim))
        elif arr.shape[1] != dim:
            raise RuntimeError("embedding 维度发生变化，无法继续")

        arr = normalize_rows(arr)
        end = i + arr.shape[0]
        mm[i:end] = arr
        mm.flush()

        save_state(
            state_path,
            {
                "next_idx": end,
                "total": total,
                "dim": dim,
                "embed_model": model,
                "completed": False,
            },
        )
        print(f"embedding: {end}/{total}", flush=True)

        i = end

    if mm is None:
        raise RuntimeError("没有可用 embedding 数据")

    save_state(
        state_path,
        {
            "next_idx": total,
            "total": total,
            "dim": dim,
            "embed_model": model,
            "completed": True,
        },
    )
    return emb_cache_path, dim


def build_faiss_from_memmap(emb_cache_path: Path, total: int, dim: int, add_batch_size: int) -> faiss.Index:
    mm = np.memmap(emb_cache_path, dtype=np.float32, mode="r", shape=(total, dim))
    index = faiss.IndexFlatIP(dim)
    for i in range(0, total, add_batch_size):
        end = min(i + add_batch_size, total)
        index.add(np.asarray(mm[i:end], dtype=np.float32))
        print(f"index.add: {end}/{total}", flush=True)
    # Windows 上 memmap 句柄释放可能延迟，主动释放避免后续 unlink 失败。
    del mm
    gc.collect()
    return index


def validate_outputs(index: faiss.Index, chunks: list[Chunk], dim: int) -> None:
    if index.d != dim:
        raise RuntimeError(f"索引维度不一致: index.d={index.d}, dim={dim}")
    if index.ntotal != len(chunks):
        raise RuntimeError(f"索引数量不一致: ntotal={index.ntotal}, chunks={len(chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建向量索引与 BM25 索引")
    parser.add_argument("--corpus", default="../qa_web/build/corpus_simplified.jsonl")
    parser.add_argument("--build-dir", default="../qa_web/build")
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-max-chars", type=int, default=1200, help="embedding 输入最大字符数（超长将截断）")
    parser.add_argument("--embed-retries", type=int, default=6, help="embedding 请求失败后的最大重试次数")
    parser.add_argument("--embed-retry-backoff", type=float, default=1.0, help="embedding 重试基准退避秒数")
    parser.add_argument("--index-batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=700, help="单个 chunk 的目标最大字符数")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="相邻 chunk 重叠字符数")
    parser.add_argument("--min-chunk-len", type=int, default=80, help="过滤过短 chunk")
    parser.add_argument("--progress-every", type=int, default=1000, help="每处理多少条打印一次进度")
    parser.add_argument("--resume", action="store_true", help="从上次 embedding 进度续跑")
    parser.add_argument("--state-path", default="", help="embedding 状态文件，默认 build/index_embed.state.json")
    parser.add_argument("--emb-cache", default="", help="embedding 缓存文件，默认 build/embeddings.f32")
    parser.add_argument("--keep-bm25-token-cache", action="store_true", help="完成后保留 BM25 分词缓存文件")
    parser.add_argument("--force-rebuild", action="store_true", help="忽略已有中间文件并从头构建")
    parser.add_argument("--keep-emb-cache", action="store_true", help="完成后保留 embedding 缓存文件")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.index_batch_size <= 0:
        raise ValueError("--index-batch-size 必须大于 0")
    if args.embed_max_chars <= 0:
        raise ValueError("--embed-max-chars 必须大于 0")
    if args.embed_retries <= 0:
        raise ValueError("--embed-retries 必须大于 0")
    if args.embed_retry_backoff <= 0:
        raise ValueError("--embed-retry-backoff 必须大于 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size 必须大于 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap 不能小于 0")
    if args.min_chunk_len <= 0:
        raise ValueError("--min-chunk-len 必须大于 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every 必须大于 0")

    corpus_path = Path(args.corpus).resolve()
    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = build_dir / "chunks.jsonl"
    bm25_path = build_dir / "bm25.pkl"
    faiss_path = build_dir / "faiss.index"
    meta_path = build_dir / "meta.pkl"
    chunk_sig_path = build_dir / "chunks.signature.json"
    bm25_token_cache_path = build_dir / "bm25_tokens.jsonl"
    bm25_token_state_path = build_dir / "bm25_tokens.state.json"
    state_path = Path(args.state_path).resolve() if args.state_path else (build_dir / "index_embed.state.json")
    emb_cache_path = Path(args.emb_cache).resolve() if args.emb_cache else (build_dir / "embeddings.f32")
    current_chunk_sig = get_chunk_build_signature(
        corpus_path=corpus_path,
        max_len=args.chunk_size,
        overlap=args.chunk_overlap,
        min_len=args.min_chunk_len,
    )

    if args.force_rebuild:
        for p in [
            chunks_path,
            bm25_path,
            faiss_path,
            meta_path,
            state_path,
            emb_cache_path,
            chunk_sig_path,
            bm25_token_cache_path,
            bm25_token_state_path,
        ]:
            if p.exists():
                p.unlink()

    if args.resume and chunks_path.exists():
        if not chunk_sig_path.exists():
            raise RuntimeError("检测到 chunks 但缺少构建签名文件，请使用 --force-rebuild 重新构建")
        old_sig = load_chunk_build_signature(chunk_sig_path)
        if not signatures_equal(old_sig, current_chunk_sig):
            raise RuntimeError("当前切块参数/语料路径与已有 chunks 不一致，请使用 --force-rebuild")
        chunks = load_chunks_jsonl(chunks_path)
        print(f"复用已有 chunks：{len(chunks)}", flush=True)
    else:
        print("开始切块...", flush=True)
        chunks = build_chunks(
            corpus_path,
            max_len=args.chunk_size,
            overlap=args.chunk_overlap,
            min_len=args.min_chunk_len,
            progress_every=args.progress_every,
        )
        if not chunks:
            raise RuntimeError("没有可用切块，请先检查语料")
        write_chunks(chunks_path, chunks)
        save_chunk_build_signature(chunk_sig_path, current_chunk_sig)
        print(f"写入 chunks：{len(chunks)}", flush=True)

    if args.resume and bm25_path.exists():
        print("复用已有 bm25.pkl", flush=True)
    else:
        print("开始构建 BM25...", flush=True)
        tokenized: list[list[str]] = []
        processed = 0

        if args.resume and bm25_token_cache_path.exists() and bm25_token_state_path.exists():
            token_state = load_state(bm25_token_state_path)
            processed = int(token_state.get("processed", 0))
            if int(token_state.get("total", -1)) != len(chunks):
                raise RuntimeError("BM25 分词缓存与当前 chunks 数量不一致，请使用 --force-rebuild")
            if processed > 0:
                with bm25_token_cache_path.open("r", encoding="utf-8") as fr:
                    for line_idx, line in enumerate(fr, start=1):
                        if line_idx > processed:
                            break
                        tokenized.append(json.loads(line))
                print(f"复用 BM25 分词缓存：{processed}/{len(chunks)}", flush=True)
        else:
            if bm25_token_cache_path.exists():
                bm25_token_cache_path.unlink()
            if bm25_token_state_path.exists():
                bm25_token_state_path.unlink()

        with bm25_token_cache_path.open("a", encoding="utf-8") as fw:
            for i, c in enumerate(chunks[processed:], start=processed + 1):
                toks = tokenize_zh(c.text)
                tokenized.append(toks)
                fw.write(json.dumps(toks, ensure_ascii=False) + "\n")

                if i % args.progress_every == 0:
                    save_state(
                        bm25_token_state_path,
                        {
                            "processed": i,
                            "total": len(chunks),
                            "completed": False,
                        },
                    )
                    print(f"BM25 分词进度：{i}/{len(chunks)}", flush=True)

        save_state(
            bm25_token_state_path,
            {
                "processed": len(chunks),
                "total": len(chunks),
                "completed": True,
            },
        )
        print(f"BM25 分词完成：{len(chunks)}/{len(chunks)}", flush=True)

        if len(tokenized) != len(chunks):
            raise RuntimeError("BM25 分词数量与 chunks 不一致，无法继续")

        print("开始构建 BM25 对象...", flush=True)
        bm25 = BM25Okapi(tokenized)
        print("开始写入 bm25.pkl...", flush=True)
        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        print("写入 bm25.pkl", flush=True)

        if not args.keep_bm25_token_cache:
            if bm25_token_cache_path.exists():
                bm25_token_cache_path.unlink()
            if bm25_token_state_path.exists():
                bm25_token_state_path.unlink()

    texts = [c.text for c in chunks]
    
    import os
    local_model_path = os.getenv("LOCAL_EMBED_MODEL_PATH", "")
    local_embed_model = None
    client = None
    if local_model_path:
        print(f"检测到 LOCAL_EMBED_MODEL_PATH，使用本地模型: {local_model_path}")
        from sentence_transformers import SentenceTransformer
        local_embed_model = SentenceTransformer(local_model_path)
    else:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        
    print("开始 embedding...", flush=True)
    emb_file, dim = embed_to_memmap(
        texts=texts,
        client=client,
        model=args.embed_model,
        local_embed_model=local_embed_model,
        batch_size=args.batch_size,
        embed_max_chars=args.embed_max_chars,
        max_retries=args.embed_retries,
        retry_backoff=args.embed_retry_backoff,
        resume=args.resume,
        state_path=state_path,
        emb_cache_path=emb_cache_path,
    )

    index = build_faiss_from_memmap(
        emb_cache_path=emb_file,
        total=len(chunks),
        dim=dim,
        add_batch_size=args.index_batch_size,
    )
    validate_outputs(index=index, chunks=chunks, dim=dim)
    raw_index = faiss.serialize_index(index)
    faiss_path.write_bytes(raw_index.tobytes())

    meta = {
        "embed_model": args.embed_model,
        "dim": int(dim),
        "total_chunks": len(chunks),
    }
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)

    if state_path.exists():
        try:
            state = load_state(state_path)
            state["completed"] = True
            save_state(state_path, state)
        except Exception:
            pass

    if emb_cache_path.exists() and not args.keep_emb_cache:
        last_err: Exception | None = None
        for i in range(6):
            try:
                emb_cache_path.unlink()
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.2 * (i + 1))
        if last_err is not None:
            print(f"警告：无法删除 embedding 缓存（文件被占用）：{emb_cache_path}", flush=True)

    print(f"完成：chunk={len(chunks)}, dim={dim}", flush=True)


if __name__ == "__main__":
    main()
