"""Compare retrieval strategies against a small, hand-labeled query set."""

import json

from .embeddings import LOGGER, PROJECT_ROOT, SongRAG, configure_logging


def ranking_metrics(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    hits = len(set(retrieved_ids) & relevant)
    reciprocal_rank = next(
        (1 / rank for rank, song_id in enumerate(retrieved_ids, 1) if song_id in relevant),
        0.0,
    )
    return {
        "precision_at_3": hits / len(retrieved_ids) if retrieved_ids else 0.0,
        "recall_at_3": hits / len(relevant) if relevant else 0.0,
        "hit_at_3": float(hits > 0),
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate(rag, cases):
    if not cases:
        raise ValueError("Evaluation requires at least one labeled query.")
    catalog_ids = {song["id"] for song in rag.songs}
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("query"), str)
            or not isinstance(case.get("relevant_ids"), list)
            or not case["relevant_ids"]
            or any(song_id not in catalog_ids for song_id in case["relevant_ids"])
        ):
            raise ValueError("Evaluation cases must cite known catalog IDs.")
    report = {"case_count": len(cases), "strategies": {}}
    for strategy in ("metadata", "embedding", "hybrid"):
        rows = []
        for case in cases:
            matches = rag.retrieve(case["query"], k=3, strategy=strategy)
            ids = [song["id"] for song, _ in matches]
            rows.append({**case, "retrieved_ids": ids, **ranking_metrics(ids, case["relevant_ids"])})
        metrics = ("precision_at_3", "recall_at_3", "hit_at_3", "reciprocal_rank")
        report["strategies"][strategy] = {
            "averages": {name: sum(row[name] for row in rows) / len(rows) for name in metrics},
            "cases": rows,
        }
    return report


def main():
    configure_logging()
    print("Evaluating retrieval locally; no Gemini calls are made.")
    try:
        cases = json.loads((PROJECT_ROOT / "data" / "retrieval_eval.json").read_text())
        report = evaluate(SongRAG(), cases)
        directory = PROJECT_ROOT / "reports"
        directory.mkdir(exist_ok=True)
        path = directory / "retrieval-evaluation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("evaluation_failed error_type=%s", type(error).__name__)
        print("Evaluation could not run. Check the catalog, evaluation data, and model download connection.")
        return 1
    print(f"\nHand-labeled requests: {report['case_count']}")
    print(f"{'Strategy':<12} {'Precision@3':>12} {'Recall@3':>10} {'Hit@3':>8} {'MRR@3':>8}")
    for name, result in report["strategies"].items():
        metrics = result["averages"]
        print(
            f"{name:<12} {metrics['precision_at_3']:>12.3f} "
            f"{metrics['recall_at_3']:>10.3f} {metrics['hit_at_3']:>8.3f} "
            f"{metrics['reciprocal_rank']:>8.3f}"
        )
    print(f"\nFull per-request results: {path}")
    print("This small diagnostic set measures retrieval; it does not measure explanation factuality.")
    return 0
