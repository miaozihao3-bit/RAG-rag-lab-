from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CANDIDATE_K, MMR_LAMBDA, TOP_K
from .types import Chunk, Hit


def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_indices: List[int],
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    """TODO(M3): 最大边际相关性（MMR）重排。

    输入：
      query_vec: 1D 数组，查询向量
      candidate_vecs: 2D 数组，候选片段向量（每行一个）
      candidate_indices: 候选片段在原始索引中的下标（仅用于调试/返回）
      k: 选几个
      lambda_mult: 相关性权重；1=纯相关性，0=纯多样性

    返回：
      list[int]，选中的下标，**指向 candidate_vecs 的行号**（0..len-1），
      按选择顺序排列。

    要求：
      1. 用余弦相似度（先对向量做 L2 归一化）；
      2. 第一轮选与 query 最相似的候选；
      3. 之后每轮选
         lambda*sim(query, doc) - (1-lambda)*max(sim(doc, selected))
         最大的候选；
      4. 已选的不要重复选。

    测试在 tests/test_retrieve.py；写完跑 pytest tests/test_retrieve.py。
    """
    raise NotImplementedError("M3: 请实现 mmr_select")


class TfidfRetriever:
    """字符 n-gram TF-IDF 召回；M1 的基线，不依赖下载模型。"""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = TOP_K, candidate_k: int = CANDIDATE_K) -> List[Hit]:
        if not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()

        n = len(scores)
        candidate_k = min(max(candidate_k, top_k), n)
        candidate_idx = np.argsort(scores)[::-1][:candidate_k].tolist()
        candidate_vecs = self.matrix[candidate_idx].toarray()

        try:
            selected = mmr_select(
                query_vec.toarray()[0],
                candidate_vecs,
                candidate_idx,
                k=min(top_k, len(candidate_idx)),
                lambda_mult=MMR_LAMBDA,
            )
        except NotImplementedError:
            # M3 未完成：退回纯相关性 Top-K，保证项目始终可运行
            selected = list(range(min(top_k, len(candidate_idx))))

        hits: List[Hit] = []
        for pos in selected:
            original = candidate_idx[pos]
            hits.append(Hit(chunk=self.chunks[original], score=float(scores[original])))
        return hits
