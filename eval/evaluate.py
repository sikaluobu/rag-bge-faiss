"""Offline evaluation of retrieval quality.

Retrieval is scored automatically: a question counts as a hit when at least one
of the top-k chunks comes from a document that actually contains the answer
(``relevant_sources`` in questions.json). That gives reproducible Hit@k and MRR
numbers without needing a human rater.

Pass ``--llm`` to additionally generate answers and save them for manual
rubric scoring (correctness / groundedness / relevance / clarity).

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --top-k 3
    python eval/evaluate.py --llm --out eval/results/answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag_engine import RAGEngine  # noqa: E402


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def first_relevant_rank(results: list[dict], relevant: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk from a relevant source."""
    for rank, item in enumerate(results, start=1):
        if item["meta"].get("source") in relevant:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="also generate answers")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    engine = RAGEngine()

    rows, ranks = [], []
    for q in questions:
        results = engine.retrieve(q["question"], top_k=args.top_k)
        relevant = set(q["relevant_sources"])
        rank = first_relevant_rank(results, relevant)
        ranks.append(rank)

        top = results[0] if results else {"score": 0.0, "meta": {}}
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "expected": q["relevant_sources"],
                "rank": rank,
                "top1_source": top["meta"].get("source"),
                "top1_score": round(top["score"], 4),
            }
        )

    n = len(rows)
    hit_at = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5)}
    mrr = sum(1.0 / r for r in ranks if r) / n

    print(f"\nRetrieval evaluation — {n} questions, top_k={args.top_k}\n")
    print(f"{'ID':<4}{'type':<15}{'rank':<6}{'top1 source':<28}{'score':<8}")
    print("-" * 61)
    for row in rows:
        rank = row["rank"] if row["rank"] else "miss"
        print(
            f"{row['id']:<4}{row['type']:<15}{str(rank):<6}"
            f"{str(row['top1_source']):<28}{row['top1_score']:<8}"
        )

    print("-" * 61)
    print(f"Hit@1 = {hit_at[1]:.3f}   Hit@3 = {hit_at[3]:.3f}   Hit@5 = {hit_at[5]:.3f}")
    print(f"MRR   = {mrr:.3f}\n")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "n_questions": n,
        "metrics": {"hit@1": hit_at[1], "hit@3": hit_at[3], "hit@5": hit_at[5], "mrr": mrr},
        "rows": rows,
    }

    if args.llm:
        from backend.llm_client import generate_answer

        for row in rows:
            results = engine.retrieve(row["question"], top_k=args.top_k)
            prompt = engine.build_prompt(row["question"], results)
            try:
                row["answer"] = generate_answer(prompt)
            except RuntimeError as exc:
                row["answer"] = f"<error: {exc}>"
            print(f"[{row['id']}] answered")

    out = Path(args.out) if args.out else ROOT / "eval" / "results" / "retrieval_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
