from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

import build_index as bi


def load_existing_chunks(path: Path) -> list[bi.Chunk]:
    chunks: list[bi.Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(
                bi.Chunk(
                    chunk_id=str(obj.get("chunk_id", "")),
                    doc_id=str(obj.get("doc_id", "")),
                    title=str(obj.get("title", "")),
                    url=str(obj.get("url", "")),
                    text=str(obj.get("text", "")),
                    text_zh_hant=str(obj.get("text_zh_hant", "")),
                    content_hash=str(obj.get("content_hash", "")),
                )
            )
    return chunks


def group_chunks_by_doc(chunks: list[bi.Chunk]) -> tuple[dict[str, list[bi.Chunk]], dict[str, str]]:
    groups: dict[str, list[bi.Chunk]] = defaultdict(list)
    fingerprints: dict[str, str] = {}
    for chunk in chunks:
        groups[chunk.doc_id].append(chunk)

    for doc_id, items in groups.items():
        h = hashlib.sha1()
        for item in items:
            h.update(item.text.encode("utf-8"))
            h.update(b"\0")
        fingerprints[doc_id] = h.hexdigest()
    return groups, fingerprints


def fingerprint_doc(doc: dict[str, Any]) -> str:
    content_hash = str(doc.get("content_hash", "")).strip()
    if content_hash:
        return content_hash
    content = str(doc.get("content", ""))
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, content_hash: str, ordinal: int, text: str) -> str:
    raw = f"{doc_id}\0{content_hash}\0{ordinal}\0{text}".encode("utf-8")
    return f"u_{hashlib.sha1(raw).hexdigest()[:24]}"


def build_doc_chunks(doc: dict[str, Any], max_len: int, overlap: int, min_len: int) -> list[bi.Chunk]:
    doc_id = str(doc.get("doc_id", ""))
    title = str(doc.get("title", ""))
    url = str(doc.get("url", ""))
    content = str(doc.get("content", ""))
    content_zh_hant = str(doc.get("content_zh_hant", content))
    content_hash = fingerprint_doc(doc)

    chunks: list[bi.Chunk] = []
    ordinal = 0
    last_offset = 0
    for block in bi.split_by_heading(content):
        for piece in bi.smart_chunk(block, max_len=max_len, overlap=overlap, min_len=min_len):
            offset = content.find(piece, last_offset)
            if offset != -1:
                piece_zh_hant = content_zh_hant[offset : offset + len(piece)]
                last_offset = offset + max(1, len(piece) - overlap)
            else:
                piece_zh_hant = piece

            chunks.append(
                bi.Chunk(
                    chunk_id=make_chunk_id(doc_id, content_hash, ordinal, piece),
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    text=piece,
                    text_zh_hant=piece_zh_hant,
                    content_hash=content_hash,
                )
            )
            ordinal += 1
    return chunks


def load_meta(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_meta(path: Path, meta: dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(meta, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="增量更新 wiki-cn 知识库")
    parser.add_argument("--corpus", default="../qa_web/build/corpus_simplified.jsonl")
    parser.add_argument("--build-dir", default="../qa_web/build")
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-max-chars", type=int, default=1200)
    parser.add_argument("--embed-retries", type=int, default=6)
    parser.add_argument("--embed-retry-backoff", type=float, default=1.0)
    parser.add_argument("--index-batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--min-chunk-len", type=int, default=80)
    parser.add_argument("--chroma-dir", default="", help="Chroma persist dir, default build/chroma_db")
    parser.add_argument("--keep-missing", action="store_true", help="保留在新语料中已经不存在的文档，不从索引删除")
    parser.add_argument("--keep-emb-cache", action="store_true", help="保留本次增量 embedding 缓存")
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = build_dir / "chunks.jsonl"
    bm25_path = build_dir / "bm25.pkl"
    meta_path = build_dir / "meta.pkl"
    chunk_sig_path = build_dir / "chunks.signature.json"
    bm25_token_cache_path = build_dir / "bm25_tokens.jsonl"
    bm25_token_state_path = build_dir / "bm25_tokens.state.json"
    emb_cache_path = build_dir / "incremental_embeddings.f32"
    state_path = build_dir / "incremental_index_embed.state.json"

    if not chunks_path.exists() or not bm25_path.exists() or not meta_path.exists():
        raise RuntimeError("缺少基础索引文件，请先运行 build_index.py 完成全量构建")

    current_sig = bi.get_chunk_build_signature(
        corpus_path=corpus_path,
        max_len=args.chunk_size,
        overlap=args.chunk_overlap,
        min_len=args.min_chunk_len,
    )
    if chunk_sig_path.exists():
        old_sig = bi.load_chunk_build_signature(chunk_sig_path)
        if not bi.signatures_equal(old_sig, current_sig):
            raise RuntimeError("切块参数或语料路径发生变化，请先重新全量构建")

    old_meta = load_meta(meta_path)
    collection_name = str(old_meta.get("collection_name", "wiki_cn_dense"))
    chroma_dir = Path(args.chroma_dir).resolve() if args.chroma_dir else (build_dir / "chroma_db")
    dense_backend = str(old_meta.get("dense_backend", "chroma"))
    if dense_backend != "chroma":
        raise RuntimeError("当前索引后端不是 Chroma，无法执行增量更新")

    old_chunks = load_existing_chunks(chunks_path)
    old_groups, old_fingerprints = group_chunks_by_doc(old_chunks)
    old_doc_ids = set(old_groups)

    new_docs: dict[str, dict[str, Any]] = {}
    for doc in bi.load_corpus(corpus_path):
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        new_docs[doc_id] = doc

    new_doc_ids = set(new_docs)
    added_doc_ids = sorted(new_doc_ids - old_doc_ids)
    deleted_doc_ids = [] if args.keep_missing else sorted(old_doc_ids - new_doc_ids)
    kept_doc_ids = sorted(old_doc_ids & new_doc_ids)
    updated_doc_ids = [doc_id for doc_id in kept_doc_ids if fingerprint_doc(new_docs[doc_id]) != old_fingerprints.get(doc_id, "")]
    changed_doc_ids = sorted(set(added_doc_ids) | set(updated_doc_ids))

    if not changed_doc_ids and not deleted_doc_ids:
        print("没有发现需要更新的文档，知识库保持最新。")
        return

    collection_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = collection_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    delete_chunk_ids: list[str] = []
    for doc_id in deleted_doc_ids + updated_doc_ids:
        delete_chunk_ids.extend([c.chunk_id for c in old_groups.get(doc_id, [])])
    if delete_chunk_ids:
        collection.delete(ids=delete_chunk_ids)

    new_chunks: list[bi.Chunk] = []
    for doc_id in changed_doc_ids:
        new_chunks.extend(build_doc_chunks(new_docs[doc_id], args.chunk_size, args.chunk_overlap, args.min_chunk_len))

    new_dim = 0
    if new_chunks:
        local_model_path = os.getenv("LOCAL_EMBED_MODEL_PATH", "")
        local_embed_model = None
        client = None
        if local_model_path:
            from sentence_transformers import SentenceTransformer

            local_embed_model = SentenceTransformer(local_model_path)
        else:
            client = OpenAI(api_key=args.api_key, base_url=args.base_url)

        temp_state_path = state_path
        temp_cache_path = emb_cache_path
        emb_file, new_dim = bi.embed_to_memmap(
            texts=[c.text for c in new_chunks],
            client=client,
            model=args.embed_model,
            local_embed_model=local_embed_model,
            batch_size=args.batch_size,
            embed_max_chars=args.embed_max_chars,
            max_retries=args.embed_retries,
            retry_backoff=args.embed_retry_backoff,
            resume=False,
            state_path=temp_state_path,
            emb_cache_path=temp_cache_path,
        )
        mm = np.memmap(emb_file, dtype=np.float32, mode="r", shape=(len(new_chunks), new_dim))
        for i in range(0, len(new_chunks), args.index_batch_size):
            batch = new_chunks[i : i + args.index_batch_size]
            batch_embeddings = np.asarray(mm[i : i + len(batch)], dtype=np.float32).tolist()
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": str(c.doc_id),
                        "title": c.title,
                        "url": c.url,
                        "content_hash": str(c.content_hash or ""),
                    }
                    for c in batch
                ],
            )
        del mm
        gc.collect()

    merged_groups: dict[str, list[bi.Chunk]] = {}
    for doc_id, items in old_groups.items():
        if doc_id in deleted_doc_ids or doc_id in updated_doc_ids:
            continue
        merged_groups[doc_id] = items
    for doc_id in changed_doc_ids:
        merged_groups[doc_id] = build_doc_chunks(new_docs[doc_id], args.chunk_size, args.chunk_overlap, args.min_chunk_len)

    merged_chunks: list[bi.Chunk] = []
    for doc_id in sorted(merged_groups):
        merged_chunks.extend(merged_groups[doc_id])

    bi.write_chunks(chunks_path, merged_chunks)
    bi.save_chunk_build_signature(chunk_sig_path, current_sig)

    tokenized = [bi.tokenize_zh(c.text) for c in merged_chunks]
    with bm25_token_cache_path.open("w", encoding="utf-8") as fw:
        for toks in tokenized:
            fw.write(json.dumps(toks, ensure_ascii=False) + "\n")
    with bm25_token_state_path.open("w", encoding="utf-8") as fw:
        json.dump({"processed": len(merged_chunks), "total": len(merged_chunks), "completed": True}, fw, ensure_ascii=False)
    bm25 = BM25Okapi(tokenized)
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)

    dim = int(new_dim or old_meta.get("dim", 0))

    new_meta = dict(old_meta)
    new_meta.update(
        {
            "embed_model": args.embed_model,
            "dim": dim,
            "total_chunks": len(merged_chunks),
            "dense_backend": "chroma",
            "chroma_dir": str(chroma_dir),
            "collection_name": collection_name,
            "incremental_update": True,
            "added_docs": len(added_doc_ids),
            "updated_docs": len(updated_doc_ids),
            "deleted_docs": len(deleted_doc_ids),
        }
    )
    dump_meta(meta_path, new_meta)

    if emb_cache_path.exists() and not args.keep_emb_cache:
        try:
            emb_cache_path.unlink()
        except Exception:
            pass
    if bm25_token_cache_path.exists():
        bm25_token_cache_path.unlink(missing_ok=True)
    if bm25_token_state_path.exists():
        bm25_token_state_path.unlink(missing_ok=True)

    print(
        f"增量更新完成：新增 {len(added_doc_ids)} 篇，更新 {len(updated_doc_ids)} 篇，删除 {len(deleted_doc_ids)} 篇，总切块 {len(merged_chunks)}。",
        flush=True,
    )


if __name__ == "__main__":
    main()
