from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from opencc import OpenCC


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_title(content: str) -> str:
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("##"):
            return line.lstrip("#").strip()
    return ""


def iter_wiki_files(data_root: Path):
    for p in sorted(data_root.glob("wiki_*")):
        if p.is_file():
            yield p


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

    last_err: Exception | None = None
    for i in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (i + 1))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 wiki-cn 并统一转简体")
    parser.add_argument("--data-root", default="../wiki-cn", help="wiki-cn 目录")
    parser.add_argument("--output", default="../qa_web/build/corpus_simplified.jsonl", help="输出 jsonl")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条，0 表示全量")
    parser.add_argument("--resume", action="store_true", help="从上次中断位置续跑")
    parser.add_argument("--state-path", default="", help="状态文件路径，默认输出文件同目录")
    parser.add_argument("--save-every", type=int, default=1000, help="每处理多少条保存一次状态")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path).resolve() if args.state_path else out_path.with_suffix(".state.json")

    if args.save_every <= 0:
        raise ValueError("--save-every 必须大于 0")

    cc = OpenCC("t2s")

    count = 0
    last_file = ""
    last_line_no = 0
    start_file = ""
    start_line_no = 0
    mode = "w"

    if args.resume:
        if state_path.exists():
            state = load_state(state_path)
            start_file = str(state.get("last_file", ""))
            start_line_no = int(state.get("last_line_no", 0))
            count = int(state.get("count", 0))
            mode = "a"
            print(f"续跑：file={start_file}, line={start_line_no}, count={count}")
        else:
            if out_path.exists():
                raise RuntimeError("检测到输出文件存在但无状态文件，无法安全续跑；请删除输出后重跑，或指定正确 --state-path")
            print("未找到状态文件，将从头开始")

    try:
        with out_path.open(mode, encoding="utf-8") as fw:
            for file_path in iter_wiki_files(data_root):
                file_name = file_path.name
                if start_file and file_name < start_file:
                    continue

                with file_path.open("r", encoding="utf-8") as fr:
                    for line_no, line in enumerate(fr, start=1):
                        if file_name == start_file and line_no <= start_line_no:
                            continue

                        last_file = file_name
                        last_line_no = line_no

                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        content = obj.get("content", "")
                        if not content:
                            continue

                        content = clean_text(content)
                        content = cc.convert(content)
                        title = extract_title(content)

                        row = {
                            "doc_id": str(obj.get("doc_id", obj.get("id", ""))),
                            "id": str(obj.get("id", "")),
                            "source_id": str(obj.get("source_id", "")),
                            "url": obj.get("data_url", ""),
                            "title": title,
                            "content": content,
                        }
                        fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1

                        if count % args.save_every == 0:
                            save_state(
                                state_path,
                                {
                                    "last_file": last_file,
                                    "last_line_no": last_line_no,
                                    "count": count,
                                    "completed": False,
                                },
                            )
                            print(
                                f"进度：{count} 条（file={last_file}, line={last_line_no}）",
                                flush=True,
                            )

                        if args.limit > 0 and count >= args.limit:
                            save_state(
                                state_path,
                                {
                                    "last_file": last_file,
                                    "last_line_no": last_line_no,
                                    "count": count,
                                    "completed": False,
                                },
                            )
                            print(f"完成：{count} 条（触发 limit）")
                            return

        save_state(
            state_path,
            {
                "last_file": last_file,
                "last_line_no": last_line_no,
                "count": count,
                "completed": True,
            },
        )
    except KeyboardInterrupt:
        save_state(
            state_path,
            {
                "last_file": last_file,
                "last_line_no": last_line_no,
                "count": count,
                "completed": False,
            },
        )
        print("检测到中断，已保存续跑状态")
        raise

    print(f"完成：{count} 条")


if __name__ == "__main__":
    main()
