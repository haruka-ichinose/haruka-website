"""RAGAS evaluation over the golden set, gated on thresholds.

Metrics: faithfulness, answer relevancy, context precision, context recall, plus
a route hit rate (share of items whose ``expected_url`` was retrieved; reported,
not gated).
Judge LLM + embeddings are Azure OpenAI. Prints a per-metric report, appends the
run (tagged with ``run_datetime``) to the ``eval_results.json`` history array,
and exits non-zero if any metric is below threshold (so it works as a CI quality
gate).

Run:  uv run python -m mecha_llm.eval.ragas_eval
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from ._models import ragas_embeddings, ragas_llm
from .dataset import GOLDEN
from .rag import answer_all

logger = logging.getLogger("mecha_llm.eval.ragas")

# Minimum acceptable mean score per metric (keyed by the metric's `.name`).
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "llm_context_precision_with_reference": 0.60,
    "context_recall": 0.60,
}
DEFAULT_THRESHOLD = 0.60

RESULTS_PATH = Path("eval_results.json")


def _append_result(entry: dict) -> int:
    """Append one run to the results file (a JSON array), returning the new
    count. A legacy single-run object is migrated to the first array element."""
    history: list = []
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if isinstance(existing, list):
            history = existing
        elif isinstance(existing, dict):
            history = [existing]
    history.append(entry)
    RESULTS_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return len(history)


def _build_dataset(results, golden) -> EvaluationDataset:
    return EvaluationDataset.from_list(
        [
            {
                "user_input": g.question,
                "response": r.answer,
                "retrieved_contexts": r.contexts,
                "reference": g.reference,
            }
            for g, r in zip(golden, results)
        ]
    )


def _route_hit_rate(results, golden) -> float | None:
    """Fraction of items with an ``expected_url`` whose url was retrieved."""
    hits = [g.expected_url in r.urls for g, r in zip(golden, results) if g.expected_url]
    return sum(hits) / len(hits) if hits else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Running RAG over %d golden questions…", len(GOLDEN))
    results = asyncio.run(answer_all([g.question for g in GOLDEN]))

    dataset = _build_dataset(results, GOLDEN)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    logger.info("Scoring with RAGAS (Azure OpenAI judge)…")
    scores = evaluate(dataset, metrics=metrics, llm=ragas_llm(), embeddings=ragas_embeddings())
    df = scores.to_pandas()

    summary = {m.name: float(df[m.name].mean()) for m in metrics if m.name in df.columns}

    print("\n=== RAGAS scores ===")
    failed: list[str] = []
    for name, value in summary.items():
        threshold = THRESHOLDS.get(name, DEFAULT_THRESHOLD)
        ok = value >= threshold
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<40} {value:.3f}  (>= {threshold})")
        if not ok:
            failed.append(name)

    route_hit_rate = _route_hit_rate(results, GOLDEN)
    if route_hit_rate is not None:
        print(f"  INFO  {'route_hit_rate':<40} {route_hit_rate:.3f}")

    entry = {
        "run_datetime": datetime.now(timezone.utc).isoformat(),
        **summary,
        "route_hit_rate": route_hit_rate,
    }
    total = _append_result(entry)
    logger.info("Appended run to %s (%d runs recorded)", RESULTS_PATH, total)

    if failed:
        print(f"\n❌ Below threshold: {', '.join(failed)}")
        sys.exit(1)
    print("\n✅ All RAGAS metrics passed thresholds.")


if __name__ == "__main__":
    main()
