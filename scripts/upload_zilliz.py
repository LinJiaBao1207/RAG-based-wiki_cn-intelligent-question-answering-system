from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def get_total_dim(build_dir: Path) -> tuple[int, int]:
    state_path = build_dir / "index_embed.state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        total = int(state.get("total", 0))
        dim = int(state.get("dim", 0))
        if total > 0 and dim > 0:
            return total, dim

    chunks_path = build_dir / "chunks.jsonl"
    meta_path = build_dir / "meta.pkl"
    if not chunks_path.exists():
        raise RuntimeError("缺少 chunks.jsonl，无法推断 total")
    if not meta_path.exists():
        raise RuntimeError("缺少 meta.pkl 或 index_embed.state.json，无法推断 dim")

    import pickle

    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    dim = int(meta.get("dim", 0))
    total = count_jsonl_lines(chunks_path)
    if total <= 0 or dim <= 0:
        raise RuntimeError("无法推断 total/dim")
    return total, dim


def ensure_collection(name: str, dim: int) -> None:
    if utility.has_collection(name):
        c = Collection(name)
        c.load()
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="wiki-cn dense vectors")
    c = Collection(name=name, schema=schema)

    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "IP",
        "params": {},
    }
    c.create_index(field_name="vector", index_params=index_params)
    c.load()


def upsert_range(
    collection_name: str,
    emb_path: Path,
    total: int,
    dim: int,
    start_id: int,
    end_id: int,
    batch_size: int,
) -> None:
    mm = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(total, dim))
    col = Collection(collection_name)

    end_id = min(end_id, total)
    if start_id >= end_id:
        print(f"无需上传：start_id={start_id}, end_id={end_id}, total={total}")
        return

    for s in range(start_id, end_id, batch_size):
        e = min(s + batch_size, end_id)
        ids = list(range(s, e))
        vecs = mm[s:e].tolist()
        col.insert([ids, vecs])
        print(f"uploaded {e}/{end_id}", flush=True)

    col.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="上传 embeddings.f32 到 Zilliz Cloud")
    p.add_argument("--uri", required=True, help="Zilliz endpoint，例如 https://xxx.api.zillizcloud.com")
    p.add_argument("--token", required=True, help="Zilliz token")
    p.add_argument("--collection", default="wiki_cn_dense", help="Collection 名称")
    p.add_argument("--build-dir", default="./build", help="build 目录")
    p.add_argument("--emb-path", default="", help="embeddings.f32 路径，默认 build/embeddings.f32")
    p.add_argument("--start-id", type=int, default=0, help="从哪个 id 开始上传")
    p.add_argument("--end-id", type=int, default=0, help="上传到哪个 id（不含），0 表示 total")
    p.add_argument("--batch-size", type=int, default=1000, help="每批上传数量")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    emb_path = Path(args.emb_path).resolve() if args.emb_path else (build_dir / "embeddings.f32")

    if not emb_path.exists():
        raise FileNotFoundError(f"找不到 embeddings 文件: {emb_path}")

    total, dim = get_total_dim(build_dir)
    end_id = args.end_id if args.end_id > 0 else total

    connections.connect(alias="default", uri=args.uri, token=args.token)
    ensure_collection(args.collection, dim)
    upsert_range(
        collection_name=args.collection,
        emb_path=emb_path,
        total=total,
        dim=dim,
        start_id=max(0, args.start_id),
        end_id=end_id,
        batch_size=max(1, args.batch_size),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
