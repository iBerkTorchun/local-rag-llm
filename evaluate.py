from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_service import (
    CHAT_MODEL_ALIAS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_ALIAS,
    INSUFFICIENT_CONTEXT_RESPONSE,
    RAGService,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation"
CASES_PATH = EVALUATION_DIRECTORY / "evaluation_cases.json"
RESULTS_PATH = EVALUATION_DIRECTORY / "evaluation_results.json"
REPORT_PATH = EVALUATION_DIRECTORY / "evaluation_report.md"


def load_cases() -> list[dict[str, Any]]:
    """Load and validate the human-authored baseline evaluation set."""
    with CASES_PATH.open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a non-empty JSON array.")

    seen_ids: set[str] = set()
    answerable_count = 0
    unanswerable_count = 0

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be a JSON object.")

        case_id = case.get("id")
        case_type = case.get("type")
        question = case.get("question")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every case must have a non-empty string id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)

        if case_type not in {"answerable", "unanswerable"}:
            raise ValueError(f"Case {case_id} has an invalid type.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {case_id} has an invalid question.")

        if case_type == "answerable":
            answerable_count += 1
            expected_sources = case.get("expected_sources")
            expected_concepts = case.get("expected_concepts")
            if (
                not isinstance(expected_sources, list)
                or not expected_sources
                or any(
                    not isinstance(source, str) or not source.strip()
                    for source in expected_sources
                )
            ):
                raise ValueError(
                    f"Answerable case {case_id} needs expected_sources."
                )
            if (
                not isinstance(expected_concepts, list)
                or not expected_concepts
                or any(
                    not isinstance(concept, str) or not concept.strip()
                    for concept in expected_concepts
                )
            ):
                raise ValueError(
                    f"Answerable case {case_id} needs expected_concepts."
                )
        else:
            unanswerable_count += 1
            if case.get("expected_fallback") is not True:
                raise ValueError(
                    f"Unanswerable case {case_id} must expect the fallback."
                )

    if answerable_count != 10 or unanswerable_count != 5:
        raise ValueError(
            "The baseline set must contain 10 answerable and 5 unanswerable cases."
        )

    return cases


def validate_sources(
    sources: object,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate source metadata returned by the existing RAG service."""
    if not isinstance(sources, list) or len(sources) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} retrieved sources.")

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("A retrieved source is not an object.")
        if not isinstance(source.get("source"), str) or not source["source"].strip():
            raise ValueError("A retrieved source has no filename.")
        if (
            not isinstance(source.get("chunk_index"), int)
            or isinstance(source["chunk_index"], bool)
        ):
            raise ValueError("A retrieved source has an invalid chunk index.")
        score = source.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("A retrieved source has an invalid similarity score.")
        if not isinstance(source.get("content"), str) or not source["content"].strip():
            raise ValueError("A retrieved source has no content.")

    return sources


def evaluate_case(
    service: RAGService,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Run one case and calculate only the approved automated checks."""
    result: dict[str, Any] = {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
    }

    if case["type"] == "answerable":
        result["expected_sources"] = list(case["expected_sources"])
        result["expected_concepts"] = list(case["expected_concepts"])
    else:
        result["expected_fallback"] = True

    started_at = time.perf_counter()
    try:
        response = service.answer_query(case["question"], top_k=DEFAULT_TOP_K)
        sources = validate_sources(
            response.get("sources"),
            min(DEFAULT_TOP_K, service.chunk_count),
        )
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The generated answer is empty.")

        result["retrieved_sources"] = [dict(source) for source in sources]
        result["generated_answer"] = answer.strip()

        if case["type"] == "answerable":
            expected_sources = set(case["expected_sources"])
            retrieved_names = [source["source"] for source in sources]
            result["hit_at_1"] = retrieved_names[0] in expected_sources
            result["hit_at_3"] = any(
                source_name in expected_sources
                for source_name in retrieved_names
            )
        else:
            result["fallback_success"] = (
                answer.strip() == INSUFFICIENT_CONTEXT_RESPONSE
            )
    except Exception as error:
        result["retrieved_sources"] = []
        result["generated_answer"] = ""
        result["error"] = f"{type(error).__name__}: {error}"
        if case["type"] == "answerable":
            result["hit_at_1"] = False
            result["hit_at_3"] = False
        else:
            result["fallback_success"] = False
    finally:
        result["latency_seconds"] = round(
            time.perf_counter() - started_at,
            6,
        )

    return result


def metric(count: int, total: int) -> dict[str, int | float]:
    percentage = (count / total * 100.0) if total else 0.0
    return {
        "count": count,
        "total": total,
        "percentage": round(percentage, 2),
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [result for result in results if result["type"] == "answerable"]
    unanswerable = [
        result for result in results if result["type"] == "unanswerable"
    ]
    latencies = [float(result["latency_seconds"]) for result in results]

    hit_at_1_count = sum(result["hit_at_1"] for result in answerable)
    hit_at_3_count = sum(result["hit_at_3"] for result in answerable)
    fallback_count = sum(
        result["fallback_success"] for result in unanswerable
    )

    return {
        "queries": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "retrieval": {
            "hit_at_1": metric(hit_at_1_count, len(answerable)),
            "hit_at_3": metric(hit_at_3_count, len(answerable)),
        },
        "grounding": {
            "fallback_success": metric(fallback_count, len(unanswerable)),
        },
        "latency_seconds": {
            "average": round(statistics.mean(latencies), 6),
            "median": round(statistics.median(latencies), 6),
            "minimum": round(min(latencies), 6),
            "maximum": round(max(latencies), 6),
        },
        "case_errors": sum("error" in result for result in results),
    }


def append_quote(lines: list[str], text: str) -> None:
    for line in text.splitlines() or [""]:
        lines.append(f"> {line}" if line else ">")


def create_report(payload: dict[str, Any]) -> str:
    configuration = payload["configuration"]
    summary = payload["summary"]
    retrieval = summary["retrieval"]
    grounding = summary["grounding"]
    latency = summary["latency_seconds"]

    lines = [
        "# Baseline RAG Evaluation Report",
        "",
        (
            "This report measures source-file retrieval and exact unsupported-question "
            "fallback behavior. **Semantic answer correctness requires human review**; "
            "expected concepts are guidance and are not automatically scored."
        ),
        "",
        "## Test configuration",
        "",
        f"- Embedding model alias: `{configuration['embedding_model_alias']}`",
        f"- Selected embedding model ID: `{configuration['embedding_model_id']}`",
        f"- Chat model alias: `{configuration['chat_model_alias']}`",
        f"- Selected chat model ID: `{configuration['chat_model_id']}`",
        f"- Knowledge-base chunks: {configuration['knowledge_base_chunks']}",
        f"- Top-K: {configuration['top_k']}",
        f"- Answerable cases: {summary['answerable_cases']}",
        f"- Unanswerable cases: {summary['unanswerable_cases']}",
        "- Query latency timing begins after model initialization.",
        "",
        "## Retrieval metrics",
        "",
        (
            f"- Hit@1: {retrieval['hit_at_1']['count']}/"
            f"{retrieval['hit_at_1']['total']} "
            f"({retrieval['hit_at_1']['percentage']:.2f}%)"
        ),
        (
            f"- Hit@3: {retrieval['hit_at_3']['count']}/"
            f"{retrieval['hit_at_3']['total']} "
            f"({retrieval['hit_at_3']['percentage']:.2f}%)"
        ),
        "",
        "## Grounding metrics",
        "",
        (
            f"- Unsupported-question fallback success: "
            f"{grounding['fallback_success']['count']}/"
            f"{grounding['fallback_success']['total']} "
            f"({grounding['fallback_success']['percentage']:.2f}%)"
        ),
        "",
        "## Performance",
        "",
        f"- Queries: {summary['queries']}",
        f"- Average latency: {latency['average']:.3f} seconds",
        f"- Median latency: {latency['median']:.3f} seconds",
        f"- Minimum latency: {latency['minimum']:.3f} seconds",
        f"- Maximum latency: {latency['maximum']:.3f} seconds",
        "",
        "## Case-by-case results",
        "",
    ]

    for result in payload["results"]:
        lines.extend(
            [
                f"### {result['id']} - {result['type']}",
                "",
                f"**Question:** {result['question']}",
                "",
                f"**Latency:** {result['latency_seconds']:.3f} seconds",
                "",
            ]
        )

        if result["type"] == "answerable":
            lines.append(
                "**Expected source(s):** "
                + ", ".join(
                    f"`{source}`" for source in result["expected_sources"]
                )
            )
            lines.extend(["", "**Expected concepts (human-review guidance):**", ""])
            lines.extend(
                f"- {concept}" for concept in result["expected_concepts"]
            )
            lines.extend(
                [
                    "",
                    f"**Automated retrieval:** Hit@1 = "
                    f"{'PASS' if result['hit_at_1'] else 'FAIL'}; Hit@3 = "
                    f"{'PASS' if result['hit_at_3'] else 'FAIL'}",
                    "",
                    "**Semantic answer correctness:** Human review required.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "**Automated fallback check:** "
                        f"{'PASS' if result['fallback_success'] else 'FAIL'}"
                    ),
                    "",
                ]
            )

        lines.extend(["**Retrieved sources:**", ""])
        if result["retrieved_sources"]:
            for rank, source in enumerate(result["retrieved_sources"], start=1):
                lines.append(
                    f"{rank}. `{source['source']}` - chunk {source['chunk_index']} - "
                    f"similarity {source['score']:.6f}"
                )
        else:
            lines.append("No sources were recorded.")

        lines.extend(["", "**Generated answer:**", ""])
        append_quote(lines, result["generated_answer"] or "No answer was recorded.")
        if "error" in result:
            lines.extend(["", f"**Execution error:** {result['error']}"])
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Cleanup",
            "",
            (
                "- Models unloaded successfully: "
                f"{'yes' if payload['cleanup']['models_unloaded'] else 'no'}"
            ),
        ]
    )
    if payload["cleanup"].get("error"):
        lines.append(f"- Cleanup error: {payload['cleanup']['error']}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    cases = load_cases()
    service = RAGService()
    results: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {"models_unloaded": False}

    service.start()
    configuration = {
        "embedding_model_alias": EMBEDDING_MODEL_ALIAS,
        "embedding_model_id": service.embedding_model_id,
        "chat_model_alias": CHAT_MODEL_ALIAS,
        "chat_model_id": service.chat_model_id,
        "knowledge_base_chunks": service.chunk_count,
        "embedding_dimension": service.embedding_dimension,
        "top_k": DEFAULT_TOP_K,
        "service_start_count": service.load_count,
    }

    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index:02d}/{len(cases):02d}] {case['id']}: {case['question']}")
            result = evaluate_case(service, case)
            results.append(result)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Completed in {result['latency_seconds']:.3f} seconds.")
    finally:
        try:
            service.close()
            cleanup["models_unloaded"] = not service.is_started
        except Exception as error:
            cleanup["error"] = f"{type(error).__name__}: {error}"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "summary": build_summary(results),
        "cleanup": cleanup,
        "results": results,
    }

    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(create_report(payload), encoding="utf-8")

    summary = payload["summary"]
    print("\nBaseline summary")
    print(
        "Hit@1: "
        f"{summary['retrieval']['hit_at_1']['count']}/"
        f"{summary['retrieval']['hit_at_1']['total']}"
    )
    print(
        "Hit@3: "
        f"{summary['retrieval']['hit_at_3']['count']}/"
        f"{summary['retrieval']['hit_at_3']['total']}"
    )
    print(
        "Fallback: "
        f"{summary['grounding']['fallback_success']['count']}/"
        f"{summary['grounding']['fallback_success']['total']}"
    )
    print(
        "Latency (average / median / min / max): "
        f"{summary['latency_seconds']['average']:.3f} / "
        f"{summary['latency_seconds']['median']:.3f} / "
        f"{summary['latency_seconds']['minimum']:.3f} / "
        f"{summary['latency_seconds']['maximum']:.3f} seconds"
    )
    print(f"Results: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(
        "Models unloaded successfully: "
        f"{'yes' if cleanup['models_unloaded'] else 'no'}"
    )

    if summary["case_errors"] or not cleanup["models_unloaded"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
