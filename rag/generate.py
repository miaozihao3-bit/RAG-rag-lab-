import time
from typing import List

import requests

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, REFUSE_THRESHOLD
from .types import Answer, Hit, Source


def should_refuse(top_score: float, threshold: float = REFUSE_THRESHOLD) -> bool:
    """最高检索分低于阈值就拒答；一个命中都没有也拒答。

    这就是一个阈值判定：分数不够，宁可说"没有"，也不让模型硬编。
    """
    if top_score <= 0:
        return True
    return top_score < threshold


def build_prompt(question: str, hits: List[Hit]) -> str:
    """组装带引用约束的 Prompt：只许用资料、逐条标 [n]、不够就说不足。

    Prompt 里的三条规矩，对应的就是"不许编、要能查、答不了就认"。
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        heading = hit.chunk.heading or "正文"
        blocks.append(
            "[%d] 来源：%s ｜ 小节：%s\n%s"
            % (i, hit.chunk.source, heading, hit.chunk.text.strip())
        )
    context = "\n\n".join(blocks) if blocks else "（无资料）"

    return (
        "你是备考知识库助手。三条规则必须遵守：\n"
        "1. 只能使用下面资料里的事实，不许编造。\n"
        "2. 每个结论后面用 [1][2] 标注出处编号。\n"
        "3. 资料不足时直接输出“资料不足，无法回答”，不要猜。\n\n"
        "资料：\n%s\n\n"
        "问题：%s\n\n"
        "答案：" % (context, question)
    )


def simple_prompt(question: str, hits: List[Hit]) -> str:
    """M1 的兜底 Prompt：没有 M4 实现时也能跑。"""
    context = "\n\n".join(
        "[%d] %s / %s\n%s"
        % (i, hit.chunk.source, hit.chunk.heading or "正文", hit.chunk.text)
        for i, hit in enumerate(hits, 1)
    )
    return (
        "你是备考知识库助手。只依据下面的资料回答；资料不足就说“资料不足，无法回答”。\n"
        "每个结论后用 [1][2] 标注来源。\n\n"
        "资料：\n%s\n\n问题：%s\n\n答案：" % (context, question)
    )


def extractive_answer(question: str, hits: List[Hit]) -> str:
    """不调用大模型：把检索片段按来源拼成可核对的答案。"""
    lines = ["根据检索到的资料（抽取式模式，未调用大模型）：", ""]
    for i, hit in enumerate(hits, 1):
        heading = hit.chunk.heading or "正文"
        snippet = hit.chunk.text.strip().replace("\n", " ")[:240]
        lines.append("[%d] %s / %s（相似度 %.3f）" % (i, hit.chunk.source, heading, hit.score))
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip()


def llm_complete(prompt: str, timeout: int = 30) -> str:
    """调用 OpenAI 兼容的 /chat/completions；没有配置就抛错。"""
    if not (LLM_BASE_URL and LLM_API_KEY):
        raise RuntimeError("LLM_BASE_URL / LLM_API_KEY 未配置")

    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": "Bearer %s" % LLM_API_KEY},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    if not (200 <= resp.status_code < 300):
        raise RuntimeError("LLM HTTP %s" % resp.status_code)

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM 响应结构异常") from exc


def generate_answer(question: str, hits: List[Hit], threshold: float = REFUSE_THRESHOLD) -> Answer:
    start = time.perf_counter()

    if not hits:
        return Answer(
            text="资料中没有检索到相关内容，无法回答。",
            sources=[],
            refused=True,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    top_score = hits[0].score
    try:
        refused = should_refuse(top_score, threshold)
    except NotImplementedError:
        refused = top_score < threshold

    if refused:
        return Answer(
            text="资料中没有足够依据回答这个问题。建议换个问法，或查阅原始笔记。",
            sources=[],
            refused=True,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    try:
        prompt = build_prompt(question, hits)
    except NotImplementedError:
        prompt = simple_prompt(question, hits)

    if LLM_BASE_URL and LLM_API_KEY:
        try:
            text = llm_complete(prompt)
        except Exception as exc:
            text = "（大模型调用失败：%s）\n\n%s" % (exc, extractive_answer(question, hits))
    else:
        text = extractive_answer(question, hits)

    sources = [
        Source(
            source=hit.chunk.source,
            heading=hit.chunk.heading,
            score=round(hit.score, 4),
            text=hit.chunk.text[:200],
        )
        for hit in hits
    ]
    return Answer(
        text=text,
        sources=sources,
        refused=False,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
