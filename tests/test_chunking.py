import pytest

from rag.ingest import chunk_markdown, simple_chunk


def test_simple_chunk_respects_max_chars_and_source():
    text = "段落一。" * 200
    chunks = simple_chunk(text, "demo.md", max_chars=100, overlap=20)
    assert chunks
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert {chunk.source for chunk in chunks} == {"demo.md"}


def test_simple_chunk_empty_text():
    assert simple_chunk("", "demo.md") == []


@pytest.mark.xfail(raises=NotImplementedError, strict=False)
def test_chunk_markdown_heading_and_overlap():
    text = (
        "# 总标题\n\n"
        "## 一、Serving\n\n" + ("服务优化。" * 40) + "\n\n"
        "## 二、RAG\n\n" + ("检索增强。" * 40)
    )
    chunks = chunk_markdown(text, "demo.md", max_chars=120, overlap=30)
    assert len(chunks) >= 2
    assert any(chunk.heading for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    # start/end 指向原文，且相邻块有重叠
    assert chunks[1].start <= chunks[0].end
