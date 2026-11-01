import json
import time
from typing import List

from .config import TOP_K
from .generate import generate_answer
from .ingest import build_chunks, load_chunks, save_chunks
from .retrieve import TfidfRetriever
from .types import Answer, Chunk, Hit


class Pipeline:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.retriever = TfidfRetriever(chunks)

    @classmethod
    def load_or_build(cls) -> "Pipeline":
        try:
            chunks, _ = load_chunks()
        except (FileNotFoundError, json.JSONDecodeError):
            chunks = []

        if not chunks:
            chunks = build_chunks()
            save_chunks(chunks)

        return cls(chunks)

    def search(self, question: str, top_k: int = TOP_K) -> List[Hit]:
        return self.retriever.search(question, top_k=top_k)

    def ask(self, question: str, top_k: int = TOP_K) -> Answer:
        start = time.perf_counter()
        hits = self.search(question, top_k=top_k)
        answer = generate_answer(question, hits)
        answer.latency_ms = int((time.perf_counter() - start) * 1000)
        return answer
