import argparse
import json

from .config import TOP_K
from .evaluate import run_eval
from .ingest import build_index
from .pipeline import Pipeline


def cmd_index(args: argparse.Namespace) -> None:
    meta = build_index()
    print("sources:", len(meta["sources"]))
    print("chunks:", meta["count"])
    print("avg_chunk_len:", meta["avg_len"])
    print("source_list:", ", ".join(meta["sources"]))


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load_or_build()
    answer = pipeline.ask(args.question, args.top_k)
    print("answer:\n%s\n" % answer.text)
    print("refused:", answer.refused)
    print("latency_ms:", answer.latency_ms)
    print("sources:")
    for source in answer.sources:
        print("- %s / %s (%.4f)" % (source.source, source.heading or "正文", source.score))


def cmd_eval(args: argparse.Namespace) -> None:
    metrics = run_eval()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="备考知识库 RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="重建索引").set_defaults(func=cmd_index)

    ask_parser = sub.add_parser("ask", help="问一个问题")
    ask_parser.add_argument("question", help="问题文本")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K, dest="top_k")
    ask_parser.set_defaults(func=cmd_ask)

    sub.add_parser("eval", help="跑 20 题评测").set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
