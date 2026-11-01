import json
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


def chunk_markdown(
    text: str,
    source: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """TODO(M2): 标题感知切分。

    要求：
    1. 识别 Markdown 标题（# / ## / ###），把当前标题路径写进 Chunk.heading；
    2. 按段落打包，单块不超过 max_chars；
    3. 相邻块保留 overlap 个字符的上下文；
    4. 返回 Chunk 列表，start/end 指向原文位置，供引用使用。

    提示：可以先按标题切成 section，再对每个 section 复用 simple_chunk，
    最后把 heading 补上。不要直接抄 simple_chunk，先想清楚"标题边界"和
    "overlap 属于同一 section 还是跨 section"。
    """
    raise NotImplementedError("M2: 请实现 chunk_markdown（标题感知 + overlap）")


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
