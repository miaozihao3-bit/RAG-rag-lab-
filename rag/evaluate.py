import json
from pathlib import Path
from typing import List

from .config import EVAL_PATH
from .pipeline import Pipeline


def run_eval(path: Path = EVAL_PATH) -> dict:
    pipeline = Pipeline.load_or_build()

    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]

    recall_hits = 0
    citation_ok = 0
    refusal_ok = 0
    latencies = []

    for row in rows:
        hits = pipeline.search(row["question"], top_k=3)
        answer = pipeline.ask(row["question"], top_k=3)
        latencies.append(answer.latency_ms)

        if row.get("answerable"):
            gold = set(row.get("gold_sources", []))
            if any(hit.chunk.source in gold for hit in hits):
                recall_hits += 1
            if answer.sources and not answer.refused:
                citation_ok += 1
            if not answer.refused:
                refusal_ok += 1
        else:
            if answer.refused:
                refusal_ok += 1

    total = len(rows)
    return {
        "questions": total,
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall@3": round(recall_hits / len(answerable), 4) if answerable else 0,
        "refusal_accuracy": round(refusal_ok / total, 4) if total else 0,
        "citation_coverage": round(citation_ok / len(answerable), 4) if answerable else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }
