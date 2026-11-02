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
    """最大边际相关性（MMR）重排。

    一句话：选第 2 条时，不只看"像不像问题"，还要看"像不像已经选中的" ——
    太像就扣分。返回的是 candidate_vecs 的行号，按选择顺序排列。
    """
    vecs = np.asarray(candidate_vecs, dtype=float)
    if vecs.ndim != 2 or len(vecs) == 0 or k <= 0:
        return []

    k = min(int(k), len(vecs))

    # 1. 先做 L2 归一化，之后点积就是余弦相似度
    query = np.asarray(query_vec, dtype=float).ravel()
    query_norm = float(np.linalg.norm(query)) or 1.0
    query = query / query_norm

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    sim_query = vecs.dot(query)      # 每个候选与问题的相似度
    sim_cross = vecs.dot(vecs.T)     # 候选两两之间的相似度

    # 2. 第一轮直接选最相关的
    selected: List[int] = [int(np.argmax(sim_query))]

    # 3. 之后每轮：相关性 − 与已选的最大相似度
    while len(selected) < k:
        best_pos = -1
        best_score = None
        for i in range(len(vecs)):
            if i in selected:
                continue
            redundancy = float(max(sim_cross[i, j] for j in selected))
            score = lambda_mult * float(sim_query[i]) - (1.0 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_pos = i
        if best_pos < 0:
            break
        selected.append(best_pos)

    return selected


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
