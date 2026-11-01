import numpy as np
import pytest

from rag.retrieve import TfidfRetriever, mmr_select
from rag.types import Chunk


def _chunks():
    return [
        Chunk("c1", "a.md", "", "PagedAttention 把 KV Cache 分页管理，减少显存碎片。", 0, 10),
        Chunk("c2", "b.md", "", "投机解码用小模型起草，大模型校验。", 0, 10),
        Chunk("c3", "c.md", "", "Agent 的终止条件包括最大轮次与超时。", 0, 10),
    ]


def test_tfidf_search_returns_relevant_chunk():
    retriever = TfidfRetriever(_chunks())
    hits = retriever.search("PagedAttention", top_k=1)
    assert hits[0].chunk.id == "c1"
    assert hits[0].score > 0


@pytest.mark.xfail(raises=NotImplementedError, strict=False)
def test_mmr_prefers_diversity_over_near_duplicate():
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array(
        [
            [0.994, 0.110, 0.0],   # 与 query 最相关
            [0.992, 0.134, 0.0],   # 与第一个近似重复
            [0.700, 0.000, 0.714], # 相关但更多样
        ]
    )
    selected = mmr_select(query, candidates, [0, 1, 2], k=2, lambda_mult=0.5)
    assert selected[0] == 0
    assert selected[1] == 2
    assert len(selected) == 2
    assert len(set(selected)) == 2
