from __future__ import annotations

import argparse
import gc
import pickle
import shutil
from pathlib import Path

from build_index import build_chroma_collection, load_chunks_jsonl, validate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Chroma from existing chunks.jsonl and embeddings.f32 cache")
    parser.add_argument("--build-dir", default="../qa_web/build")
    parser.add_argument("--emb-cache", default="", help="embedding cache file, default build/embeddings.f32")
    parser.add_argument("--chroma-dir", default="", help="target Chroma persist dir, default build/chroma_db")
    parser.add_argument("--collection-name", default="", help="target collection name, default from meta.pkl or wiki_cn_dense")
    parser.add_argument("--index-batch-size", type=int, default=4096)
    parser.add_argument("--peek-limit", type=int, default=3, help="number of peek samples to print after rebuild")
    parser.add_argument("--show-snippet-chars", type=int, default=120, help="max chars to show for peeked document snippets")
    args = parser.parse_args()

    if args.index_batch_size <= 0:
        raise ValueError("--index-batch-size must be greater than 0")
    if args.peek_limit < 0:
        raise ValueError("--peek-limit must be greater than or equal to 0")
    if args.show_snippet_chars <= 0:
        raise ValueError("--show-snippet-chars must be greater than 0")

    build_dir = Path(args.build_dir).resolve()
    chunks_path = build_dir / "chunks.jsonl"
    meta_path = build_dir / "meta.pkl"
    emb_cache_path = Path(args.emb_cache).resolve() if args.emb_cache else (build_dir / "embeddings.f32")

    if not chunks_path.exists():
        raise FileNotFoundError(f"missing chunks file: {chunks_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta file: {meta_path}")
    if not emb_cache_path.exists():
        raise FileNotFoundError(f"missing embedding cache file: {emb_cache_path}")

    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    dim = int(meta.get("dim", 0))
    total_chunks = int(meta.get("total_chunks", 0))
    collection_name = args.collection_name or str(meta.get("collection_name", "wiki_cn_dense"))
    chroma_dir = Path(args.chroma_dir).resolve() if args.chroma_dir else (build_dir / "chroma_db")

    if dim <= 0:
        raise RuntimeError(f"invalid embedding dim in meta.pkl: {dim}")
    if total_chunks <= 0:
        raise RuntimeError(f"invalid total_chunks in meta.pkl: {total_chunks}")

    expected_size = total_chunks * dim * 4
    actual_size = emb_cache_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"embedding cache size mismatch: actual={actual_size}, expected={expected_size}, "
            f"total_chunks={total_chunks}, dim={dim}"
        )

    print(f"loading chunks: {chunks_path}", flush=True)
    chunks = load_chunks_jsonl(chunks_path)
    if len(chunks) != total_chunks:
        raise RuntimeError(f"chunks count mismatch: len(chunks)={len(chunks)}, meta.total_chunks={total_chunks}")

    if chroma_dir.exists():
        print(f"removing existing chroma dir: {chroma_dir}", flush=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"rebuilding Chroma from cache: chunks={len(chunks)}, dim={dim}, "
        f"collection={collection_name}, dir={chroma_dir}",
        flush=True,
    )
    collection = build_chroma_collection(
        emb_cache_path=emb_cache_path,
        chunks=chunks,
        total=len(chunks),
        dim=dim,
        add_batch_size=args.index_batch_size,
        persist_dir=chroma_dir,
        collection_name=collection_name,
    )
    validate_outputs(collection=collection, chunks=chunks, dim=dim)

    count = int(collection.count())
    print(f"done: collection.count()={count}", flush=True)

    if args.peek_limit > 0:
        try:
            sample = collection.peek(limit=args.peek_limit)
            ids = sample.get("ids") or []
            metadatas = sample.get("metadatas") or []
            documents = sample.get("documents") or []
            print(f"peek: {len(ids)} sample(s)", flush=True)
            for i, chunk_id in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""
                snippet = str(doc).replace("\r", " ").replace("\n", " ").strip()
                if len(snippet) > args.show_snippet_chars:
                    snippet = snippet[: args.show_snippet_chars] + "..."
                print(
                    f"[{i}] id={chunk_id} title={meta.get('title', '')} "
                    f"doc_id={meta.get('doc_id', '')} snippet={snippet}",
                    flush=True,
                )
        except Exception as e:
            print(f"warning: peek failed: {e!r}", flush=True)

    del collection
    gc.collect()


if __name__ == "__main__":
    main()
