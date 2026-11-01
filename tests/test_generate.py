import pytest

from rag.generate import build_prompt, extractive_answer, generate_answer, should_refuse
from rag.types import Chunk, Hit


def _hits(score: float = 0.8):
    return [
        Hit(
            Chunk(
                "c1",
                "01-45A样卷与答案.md",
                "一、vLLM",
                "vLLM 以 Continuous Batching 和 PagedAttention 为核心优化技术。",
                0,
                10,
            ),
            score,
        )
    ]


def test_extractive_answer_contains_source_marker():
    text = extractive_answer("vLLM 的核心技术？", _hits())
    assert "[1]" in text
    assert "01-45A样卷与答案.md" in text


def test_generate_answer_refuses_low_score_without_m4():
    answer = generate_answer("一个笔记里没有的问题", _hits(0.01), threshold=0.2)
    assert answer.refused is True
    assert answer.sources == []


@pytest.mark.xfail(raises=NotImplementedError, strict=False)
def test_build_prompt_contains_question_and_refusal_rule():
    prompt = build_prompt("vLLM 的核心技术？", _hits())
    assert "vLLM 的核心技术" in prompt
    assert "[1]" in prompt
    assert ("资料不足" in prompt) or ("无法回答" in prompt)


@pytest.mark.xfail(raises=NotImplementedError, strict=False)
def test_should_refuse_threshold():
    assert should_refuse(0.05, 0.2)
    assert not should_refuse(0.5, 0.2)
