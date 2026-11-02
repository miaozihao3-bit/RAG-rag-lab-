import json
import re
from pathlib import Path
from typing import List, Tuple

from .config import CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_DIR, INDEX_PATH
from .types import Chunk


def read_corpus(corpus_dir: Path = CORPUS_DIR) -> List[Tuple[str, str]]:
    """读取 data/corpus 下的 Markdown，返回 (文件名, 全文)。"""
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def simple_chunk(
    text: str,
    source: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """M1 的基线切分：按字符窗口切，尽量落在换行处，带 overlap。"""
    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    n = len(text)
    idx = 0

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            # 优先在段落边界切，其次在换行处切；找不到就硬切
            cut = text.rfind("\n\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind("\n", start + max_chars // 2, end)
            if cut != -1:
                end = cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    id="%s-%03d" % (source, idx),
                    source=source,
                    heading="",
                    text=piece,
                    start=start,
                    end=end,
                )
            )
            idx += 1

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


# 只认 # / ## / ### 三级标题，且必须是行首
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.M)


def _section_slice(text: str) -> List[Tuple[int, int, str]]:
    """按标题切成 (正文起点, 正文终点, 标题路径)。"""
    marks = [
        (m.start(), m.end(), len(m.group(1)), m.group(2).strip())
        for m in HEADING_RE.finditer(text)
    ]
    if not marks:
        return [(0, len(text), "")]

    sections: List[Tuple[int, int, str]] = []
    if marks[0][0] > 0:
        sections.append((0, marks[0][0], ""))

    stack: List[Tuple[int, str]] = []
    for i, (_start, head_end, level, title) in enumerate(marks):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(item[1] for item in stack)
        body_end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append((head_end, body_end, path))

    return sections


def chunk_markdown(
    text: str,
    source: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """标题感知切分：先按标题切成 section，再在 section 内部按窗口打包。

    和 simple_chunk 的区别只有两点：
    1. 下刀位置不跨 section，标题是天生的语义边界；
    2. 每块记下当前标题路径，检索结果就能显示"出自哪一节"。
    """
    if not text.strip():
        return []

    chunks: List[Chunk] = []
    idx = 0

    for body_start, body_end, heading in _section_slice(text):
        start = body_start
        while start < body_end:
            end = min(start + max_chars, body_end)

            if end < body_end:
                # 优先在段落边界切，其次在换行处切
                cut = text.rfind("\n\n", start + max_chars // 2, end)
                if cut == -1:
                    cut = text.rfind("\n", start + max_chars // 2, end)
                if cut != -1:
                    end = cut

            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id="%s-%03d" % (source, idx),
                        source=source,
                        heading=heading,
                        text=piece,
                        start=start,
                        end=end,
                    )
                )
                idx += 1

            if end >= body_end:
                break
            start = max(end - overlap, start + 1)

    return chunks


def build_chunks(mode: str = "auto") -> List[Chunk]:
    """按 mode 切分全部语料。

    mode=simple：只用基线切分
    mode=heading：只用标题感知切分（M2 未完成会抛错）
    mode=auto：优先标题感知，未实现则回退基线
    """
    docs = read_corpus()
    chunks: List[Chunk] = []

    for source, text in docs:
        if mode in ("auto", "heading"):
            try:
                part = chunk_markdown(text, source)
            except NotImplementedError:
                if mode == "heading":
                    raise
                part = simple_chunk(text, source)
        else:
            part = simple_chunk(text, source)
        chunks.extend(part)

    return chunks


def save_chunks(chunks: List[Chunk], path: Path = INDEX_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": [chunk.__dict__ for chunk in chunks],
        "meta": {
            "count": len(chunks),
            "sources": sorted({chunk.source for chunk in chunks}),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path = INDEX_PATH) -> Tuple[List[Chunk], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = [Chunk(**row) for row in payload["chunks"]]
    return chunks, payload.get("meta", {})


def build_index(mode: str = "auto") -> dict:
    chunks = build_chunks(mode)
    save_chunks(chunks)
    sources = sorted({chunk.source for chunk in chunks})
    avg_len = round(sum(len(chunk.text) for chunk in chunks) / len(chunks), 1) if chunks else 0
    return {"count": len(chunks), "sources": sources, "avg_len": avg_len}
