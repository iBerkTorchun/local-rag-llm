from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation"
RESULTS_PATH = EVALUATION_DIRECTORY / "final_cuda_results.json"
REPORT_PATH = EVALUATION_DIRECTORY / "final_cuda_evaluation_report.md"
CPU_RESULTS_PATH = EVALUATION_DIRECTORY / "tuned_results.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import build_summary, evaluate_case, load_cases  # noqa: E402
from warm_request_timing import (  # noqa: E402
    InstrumentedRAGService,
    TimedChatClient,
    TimedEmbeddingClient,
    describe_model,
)


def summarize_stage(values: list[float]) -> dict[str, float]:
    return {
        "average": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def compare_with_cpu(
    final_results: list[dict[str, Any]],
) -> dict[str, Any]:
    cpu_payload = json.loads(CPU_RESULTS_PATH.read_text(encoding="utf-8"))
    cpu_by_id = {result["id"]: result for result in cpu_payload["results"]}
    cases: list[dict[str, Any]] = []

    for final_result in final_results:
        cpu_result = cpu_by_id[final_result["id"]]
        cpu_sources = [
            f"{source['source']}#{source['chunk_index']}"
            for source in cpu_result["retrieved_sources"]
        ]
        final_sources = [
            f"{source['source']}#{source['chunk_index']}"
            for source in final_result["retrieved_sources"]
        ]
        cpu_scores = {
            f"{source['source']}#{source['chunk_index']}": float(source["score"])
            for source in cpu_result["retrieved_sources"]
        }
        final_scores = {
            f"{source['source']}#{source['chunk_index']}": float(source["score"])
            for source in final_result["retrieved_sources"]
        }
        common_sources = sorted(set(cpu_sources) & set(final_sources))
        cases.append(
            {
                "id": final_result["id"],
                "cpu_sources": cpu_sources,
                "final_sources": final_sources,
                "top_1_same": cpu_sources[:1] == final_sources[:1],
                "top_3_set_same": set(cpu_sources) == set(final_sources),
                "top_3_order_same": cpu_sources == final_sources,
                "maximum_common_score_difference": max(
                    (
                        abs(cpu_scores[source] - final_scores[source])
                        for source in common_sources
                    ),
                    default=None,
                ),
                "cpu_answer": cpu_result["generated_answer"],
                "final_answer": final_result["generated_answer"],
                "cpu_fallback_success": cpu_result.get("fallback_success"),
                "final_fallback_success": final_result.get("fallback_success"),
                "human_semantic_review_required": True,
            }
        )

    score_differences = [
        case["maximum_common_score_difference"]
        for case in cases
        if case["maximum_common_score_difference"] is not None
    ]
    return {
        "baseline_artifact": str(CPU_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "case_count": len(cases),
        "top_1_same_count": sum(case["top_1_same"] for case in cases),
        "top_3_set_same_count": sum(
            case["top_3_set_same"] for case in cases
        ),
        "top_3_order_same_count": sum(
            case["top_3_order_same"] for case in cases
        ),
        "maximum_common_score_difference": max(score_differences),
        "fallback_behavior_changes": [
            case["id"]
            for case in cases
            if case["cpu_fallback_success"] != case["final_fallback_success"]
        ],
        "cases": cases,
    }


def create_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    latency = summary["latency_seconds"]
    stages = payload["stage_latency_seconds"]
    comparison = payload["cpu_retrieval_comparison"]
    return "\n".join(
        [
            "# Final Production CUDA Evaluation",
            "",
            "## Runtime",
            "",
            f"- Service startup: {payload['startup']['total_seconds']:.3f} seconds",
            f"- Embedding model: `{payload['models']['embedding']['selected_id']}`",
            f"- Embedding runtime: {payload['models']['embedding']['selected_runtime']['device_type']} / `{payload['models']['embedding']['selected_runtime']['execution_provider']}`",
            f"- Chat model: `{payload['models']['chat']['selected_id']}`",
            f"- Chat runtime: {payload['models']['chat']['selected_runtime']['device_type']} / `{payload['models']['chat']['selected_runtime']['execution_provider']}`",
            f"- Cached models were reused: {'yes' if payload['models']['embedding']['selected_cached'] and payload['models']['chat']['selected_cached'] else 'no'}",
            "",
            "## Metrics",
            "",
            f"- Hit@1: {summary['retrieval']['hit_at_1']['count']}/{summary['retrieval']['hit_at_1']['total']}",
            f"- Hit@3: {summary['retrieval']['hit_at_3']['count']}/{summary['retrieval']['hit_at_3']['total']}",
            f"- Exact fallback: {summary['grounding']['fallback_success']['count']}/{summary['grounding']['fallback_success']['total']}",
            f"- Case errors: {summary['case_errors']}",
            f"- Query latency average/median/min/max: {latency['average']:.3f} / {latency['median']:.3f} / {latency['minimum']:.3f} / {latency['maximum']:.3f} seconds",
            f"- Embedding average: {stages['embedding']['average']:.3f} seconds",
            f"- Retrieval average: {stages['retrieval']['average']:.3f} seconds",
            f"- Generation average: {stages['generation']['average']:.3f} seconds",
            "",
            "## CPU-baseline retrieval comparison",
            "",
            f"- Same Top-1: {comparison['top_1_same_count']}/{comparison['case_count']}",
            f"- Same Top-3 set: {comparison['top_3_set_same_count']}/{comparison['case_count']}",
            f"- Same Top-3 order: {comparison['top_3_order_same_count']}/{comparison['case_count']}",
            f"- Maximum shared-source score difference: {comparison['maximum_common_score_difference']:.6f}",
            f"- Fallback-behavior changes: {', '.join(comparison['fallback_behavior_changes']) or 'none'}",
            "- Semantic correctness and wording differences require human review.",
            "",
            "## Cleanup",
            "",
            f"- Models unloaded: {'yes' if payload['cleanup']['models_unloaded'] else 'no'}",
        ]
    ) + "\n"


def main() -> int:
    if RESULTS_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Refusing to overwrite final evaluation artifacts.")

    cases = load_cases()
    service = InstrumentedRAGService()
    cleanup: dict[str, Any] = {"models_unloaded": False}
    results: list[dict[str, Any]] = []

    startup_started = time.perf_counter()
    service.start()
    startup_seconds = time.perf_counter() - startup_started
    if (
        service._embedding_model is None
        or service._chat_model is None
        or service._embedding_client is None
        or service._chat_client is None
    ):
        raise RuntimeError("Production service did not expose its loaded resources.")

    embedding_client = TimedEmbeddingClient(service._embedding_client)
    chat_client = TimedChatClient(service._chat_client)
    service._embedding_client = embedding_client
    service._chat_client = chat_client

    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index:02d}/{len(cases):02d}] {case['id']}")
            embedding_client.last_seconds = None
            chat_client.last_seconds = None
            service.last_top_chunks_seconds = None
            result = evaluate_case(service, case)
            if (
                embedding_client.last_seconds is None
                or chat_client.last_seconds is None
                or service.last_top_chunks_seconds is None
            ):
                raise RuntimeError(f"Timing was not captured for {case['id']}.")
            result["timing_breakdown"] = {
                "embedding_seconds": round(embedding_client.last_seconds, 6),
                "retrieval_seconds": round(
                    max(
                        0.0,
                        service.last_top_chunks_seconds
                        - embedding_client.last_seconds,
                    ),
                    6,
                ),
                "generation_seconds": round(chat_client.last_seconds, 6),
            }
            results.append(result)
            print(f"  total={result['latency_seconds']:.3f}s")
    finally:
        try:
            service.close()
            cleanup["models_unloaded"] = not service.is_started
        except Exception as error:
            cleanup["error"] = f"{type(error).__name__}: {error}"

    summary = build_summary(results)
    stage_latency = {
        stage: summarize_stage(
            [result["timing_breakdown"][f"{stage}_seconds"] for result in results]
        )
        for stage in ("embedding", "retrieval", "generation")
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "startup": {
            "total_seconds": round(startup_seconds, 6),
            "provider_registration_time_included": True,
            "provider_component_download_time_included": False,
            "model_download_time_included": False,
            "cached_model_load_and_sqlite_read_included": True,
        },
        "models": {
            "embedding": describe_model(service._embedding_model),
            "chat": describe_model(service._chat_model),
        },
        "configuration": {
            "top_k": 3,
            "temperature": 0.0,
            "random_seed": 0,
            "max_tokens": 96,
            "chunk_count": 18,
            "embedding_dimension": 1024,
        },
        "summary": summary,
        "stage_latency_seconds": stage_latency,
        "cpu_retrieval_comparison": compare_with_cpu(results),
        "cleanup": cleanup,
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(create_report(payload), encoding="utf-8")
    print(f"Results: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")
    return 0 if summary["case_errors"] == 0 and cleanup["models_unloaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
