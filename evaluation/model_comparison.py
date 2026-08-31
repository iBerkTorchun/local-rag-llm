from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import load_cases  # noqa: E402
from rag_service import (  # noqa: E402
    CHAT_MODEL_ALIAS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_ALIAS,
    GROUNDING_SYSTEM_INSTRUCTION,
    INSUFFICIENT_CONTEXT_RESPONSE,
    RAGService,
)


RESULTS_PATH = PROJECT_ROOT / "evaluation" / "model_comparison_results.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "model_comparison_report.md"

PHI_KEY = "phi_3_5_mini"
QWEN_KEY = "qwen2_5_0_5b"
QWEN_MODEL_ALIAS = "qwen2.5-0.5b"

GENERATION_SETTINGS = {
    "temperature": 0.0,
    "max_tokens": 96,
    "random_seed": 0,
}


def configure_chat_client(client: Any) -> dict[str, Any]:
    """Apply equivalent deterministic settings where the client supports them."""
    applied: dict[str, Any] = {}
    unsupported: list[str] = []

    for name, value in GENERATION_SETTINGS.items():
        if not hasattr(client.settings, name):
            unsupported.append(name)
            continue
        setattr(client.settings, name, value)
        applied[name] = getattr(client.settings, name)

    return {"applied": applied, "unsupported": unsupported}


def download_if_needed(model: Any, label: str) -> None:
    """Download a catalog model before timing any model queries."""
    if model.is_cached:
        print(f"[OK] {label} model is already downloaded.")
        return

    model.download(
        lambda percent: print(
            f"\rDownloading {model.alias}: {percent:5.1f}%",
            end="",
            flush=True,
        )
    )
    print()
    if not model.is_cached:
        raise RuntimeError(f"{label} model download completed but is not cached.")
    print(f"[OK] {label} model downloaded.")


def build_messages(question: str, augmented_context: str) -> list[dict[str, str]]:
    """Match the production RAG service's tuned prompt and user-message format."""
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n\n{augmented_context}"
                f"\n\nQuestion:\n{question}"
            ),
        },
    ]


def complete_answer(client: Any, question: str, context: str) -> str:
    response = client.complete_chat(build_messages(question, context))
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("The chat model returned an empty answer.")

    answer = response.choices[0].message.content.strip()
    if not answer:
        raise RuntimeError("The chat model returned an empty answer.")
    return answer


def retrieve_shared_contexts(
    service: RAGService,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retrieve once so both chat models receive identical Top-3 context."""
    records: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        print(f"[retrieve {index:02d}/{len(cases):02d}] {case['id']}")
        started_at = time.perf_counter()
        sources = service.get_top_chunks(case["question"], top_k=DEFAULT_TOP_K)
        retrieval_latency = time.perf_counter() - started_at

        if len(sources) != DEFAULT_TOP_K:
            raise RuntimeError(
                f"Case {case['id']} returned {len(sources)} sources instead of "
                f"{DEFAULT_TOP_K}."
            )

        # Use the production formatter directly without changing rag_service.py.
        context = RAGService._build_augmented_context(sources)
        record: dict[str, Any] = {
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "retrieval_latency_seconds": round(retrieval_latency, 6),
            "retrieved_sources": [dict(source) for source in sources],
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "augmented_context": context,
        }

        if case["type"] == "answerable":
            record["expected_sources"] = list(case["expected_sources"])
            record["expected_concepts"] = list(case["expected_concepts"])
            expected_sources = set(case["expected_sources"])
            retrieved_names = [source["source"] for source in sources]
            record["hit_at_1"] = retrieved_names[0] in expected_sources
            record["hit_at_3"] = any(
                source_name in expected_sources for source_name in retrieved_names
            )
        else:
            record["expected_fallback"] = True

        records.append(record)

    return records


def benchmark_model(
    model_key: str,
    client: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate answers using the already-retrieved shared contexts."""
    outputs: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        print(f"[{model_key} {index:02d}/{len(records):02d}] {record['id']}")
        started_at = time.perf_counter()
        output: dict[str, Any] = {}

        try:
            answer = complete_answer(
                client,
                record["question"],
                record["augmented_context"],
            )
            output["generated_answer"] = answer
            if record["type"] == "unanswerable":
                output["fallback_success"] = (
                    answer == INSUFFICIENT_CONTEXT_RESPONSE
                )
        except Exception as error:
            output["generated_answer"] = ""
            output["error"] = f"{type(error).__name__}: {error}"
            if record["type"] == "unanswerable":
                output["fallback_success"] = False

        generation_latency = time.perf_counter() - started_at
        output["generation_latency_seconds"] = round(generation_latency, 6)
        output["query_latency_seconds"] = round(
            record["retrieval_latency_seconds"] + generation_latency,
            6,
        )
        outputs.append(output)

        if "error" in output:
            print(f"  ERROR: {output['error']}")
        else:
            print(
                f"  generation={output['generation_latency_seconds']:.3f}s, "
                f"shared-retrieval+generation={output['query_latency_seconds']:.3f}s"
            )

    return outputs


def rate(count: int, total: int) -> dict[str, int | float]:
    return {
        "count": count,
        "total": total,
        "percentage": round((count / total * 100.0) if total else 0.0, 2),
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("At least one successful latency value is required.")
    return {
        "average": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [record for record in records if record["type"] == "answerable"]
    retrieval_latencies = [
        float(record["retrieval_latency_seconds"]) for record in records
    ]
    return {
        "hit_at_1": rate(
            sum(record["hit_at_1"] for record in answerable),
            len(answerable),
        ),
        "hit_at_3": rate(
            sum(record["hit_at_3"] for record in answerable),
            len(answerable),
        ),
        "shared_retrieval_latency_seconds": latency_summary(retrieval_latencies),
    }


def summarize_model(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [output for output in outputs if "error" not in output]
    unanswerable_outputs = [
        output
        for record, output in zip(records, outputs)
        if record["type"] == "unanswerable"
    ]
    return {
        "exact_fallback": rate(
            sum(output["fallback_success"] for output in unanswerable_outputs),
            len(unanswerable_outputs),
        ),
        "query_latency_seconds": latency_summary(
            [float(output["query_latency_seconds"]) for output in successful]
        ),
        "generation_latency_seconds": latency_summary(
            [float(output["generation_latency_seconds"]) for output in successful]
        ),
        "case_errors": len(outputs) - len(successful),
    }


def append_quote(lines: list[str], text: str) -> None:
    for line in text.splitlines() or [""]:
        lines.append(f"> {line}" if line else ">")


def create_report(payload: dict[str, Any]) -> str:
    configuration = payload["configuration"]
    retrieval = payload["retrieval_summary"]
    phi = payload["model_summaries"][PHI_KEY]
    qwen = payload["model_summaries"][QWEN_KEY]
    phi_latency = phi["query_latency_seconds"]
    qwen_latency = qwen["query_latency_seconds"]
    query_speedup = phi_latency["average"] / qwen_latency["average"]
    query_reduction = (
        1.0 - qwen_latency["average"] / phi_latency["average"]
    ) * 100.0
    generation_speedup = (
        phi["generation_latency_seconds"]["average"]
        / qwen["generation_latency_seconds"]["average"]
    )

    lines = [
        "# Controlled Chat-Model Quality and Latency Comparison",
        "",
        (
            "This experiment compares the production Phi-3.5 Mini model with "
            "Qwen2.5-0.5B without changing the production configuration. Semantic "
            "answer quality requires human review and is not automatically scored."
        ),
        "",
        "## Test configuration",
        "",
        f"- Embedding model: `{configuration['embedding_model_alias']}` "
        f"(`{configuration['embedding_model_id']}`)",
        f"- Production chat model: `{configuration['models'][PHI_KEY]['alias']}` "
        f"(`{configuration['models'][PHI_KEY]['id']}`)",
        f"- Comparison chat model: `{configuration['models'][QWEN_KEY]['alias']}` "
        f"(`{configuration['models'][QWEN_KEY]['id']}`)",
        f"- Top-K: {configuration['top_k']}",
        f"- Cases: {configuration['case_count']}",
        f"- Grounding prompt SHA-256: `{configuration['grounding_prompt_sha256']}`",
        (
            "- Generation settings applied to both clients: temperature 0.0, "
            "max tokens 96, random seed 0"
        ),
        (
            "- Comparable query latency is shared retrieval latency plus chat "
            "generation latency; model download and load time are excluded."
        ),
        "",
        "## Retrieval identity",
        "",
        (
            "Every question was retrieved exactly once through the existing "
            "`RAGService.get_top_chunks()` path. The resulting context was reused "
            "unchanged for both chat models."
        ),
        "",
        f"- Hit@1: {retrieval['hit_at_1']['count']}/"
        f"{retrieval['hit_at_1']['total']} "
        f"({retrieval['hit_at_1']['percentage']:.2f}%)",
        f"- Hit@3: {retrieval['hit_at_3']['count']}/"
        f"{retrieval['hit_at_3']['total']} "
        f"({retrieval['hit_at_3']['percentage']:.2f}%)",
        "",
        "## Model comparison",
        "",
        "| Metric | Phi-3.5 Mini | Qwen2.5-0.5B |",
        "| --- | ---: | ---: |",
        (
            f"| Exact fallback | {phi['exact_fallback']['count']}/"
            f"{phi['exact_fallback']['total']} "
            f"({phi['exact_fallback']['percentage']:.2f}%) | "
            f"{qwen['exact_fallback']['count']}/"
            f"{qwen['exact_fallback']['total']} "
            f"({qwen['exact_fallback']['percentage']:.2f}%) |"
        ),
        f"| Average query latency | {phi_latency['average']:.3f} s | "
        f"{qwen_latency['average']:.3f} s |",
        f"| Median query latency | {phi_latency['median']:.3f} s | "
        f"{qwen_latency['median']:.3f} s |",
        f"| Minimum query latency | {phi_latency['minimum']:.3f} s | "
        f"{qwen_latency['minimum']:.3f} s |",
        f"| Maximum query latency | {phi_latency['maximum']:.3f} s | "
        f"{qwen_latency['maximum']:.3f} s |",
        "",
        "## Performance interpretation",
        "",
        (
            f"Qwen's average comparable query latency was {query_speedup:.2f}x "
            f"faster ({query_reduction:.1f}% lower) than Phi's. Its average chat "
            f"generation time alone was {generation_speedup:.2f}x faster. These "
            "figures establish a latency difference, not a model-quality verdict."
        ),
        "",
        "## Human review",
        "",
        (
            "Review the paired answers below for false refusals, unsupported claims, "
            "fallback-format failures, generated references, incomplete text, and "
            "other grounding defects. No semantic score is generated by this script."
        ),
        "",
        "## Case-by-case outputs",
        "",
    ]

    for result in payload["results"]:
        lines.extend(
            [
                f"### `{result['id']}` - {result['type']}",
                "",
                f"**Question:** {result['question']}",
                "",
                f"**Shared retrieval latency:** "
                f"{result['retrieval_latency_seconds']:.3f} seconds",
                "",
                "**Shared Top-3:**",
                "",
            ]
        )
        for rank, source in enumerate(result["retrieved_sources"], start=1):
            lines.append(
                f"{rank}. `{source['source']}` - chunk {source['chunk_index']} - "
                f"similarity {source['score']:.6f}"
            )

        for model_key, label in (
            (PHI_KEY, "Phi-3.5 Mini"),
            (QWEN_KEY, "Qwen2.5-0.5B"),
        ):
            output = result["models"][model_key]
            lines.extend(
                [
                    "",
                    f"**{label} answer** "
                    f"(generation {output['generation_latency_seconds']:.3f} s; "
                    f"query {output['query_latency_seconds']:.3f} s):",
                    "",
                ]
            )
            append_quote(
                lines,
                output["generated_answer"] or "No answer was recorded.",
            )
            if result["type"] == "unanswerable":
                lines.extend(
                    [
                        "",
                        "Exact fallback: "
                        f"{'PASS' if output['fallback_success'] else 'FAIL'}",
                    ]
                )
            if "error" in output:
                lines.extend(["", f"Execution error: {output['error']}"])

        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Cleanup",
            "",
            f"- Phi and embedding models unloaded: "
            f"{'yes' if payload['cleanup']['phi_and_embedding_unloaded'] else 'no'}",
            f"- Qwen model unloaded: "
            f"{'yes' if payload['cleanup']['qwen_unloaded'] else 'no'}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    cases = load_cases()
    service = RAGService()
    shared_records: list[dict[str, Any]] = []
    phi_outputs: list[dict[str, Any]] = []
    qwen_outputs: list[dict[str, Any]] = []
    cleanup = {
        "phi_and_embedding_unloaded": False,
        "qwen_unloaded": False,
    }
    manager = None
    phi_configuration: dict[str, Any] = {}
    embedding_model_id: str | None = None

    try:
        service.start()
        manager = service._manager
        phi_client = service._chat_client
        if manager is None or phi_client is None:
            raise RuntimeError("The production RAG service did not expose its clients.")

        embedding_model_id = service.embedding_model_id
        phi_configuration = {
            "alias": CHAT_MODEL_ALIAS,
            "id": service.chat_model_id,
            "settings": configure_chat_client(phi_client),
        }
        shared_records = retrieve_shared_contexts(service, cases)
        phi_outputs = benchmark_model(PHI_KEY, phi_client, shared_records)
    finally:
        try:
            service.close()
            cleanup["phi_and_embedding_unloaded"] = not service.is_started
        except Exception as error:
            cleanup["phi_cleanup_error"] = f"{type(error).__name__}: {error}"

    if manager is None:
        raise RuntimeError("Foundry Local manager was not initialized.")
    if not cleanup["phi_and_embedding_unloaded"]:
        raise RuntimeError("Phi or embedding model cleanup failed before Qwen run.")

    qwen_model = manager.catalog.get_model(QWEN_MODEL_ALIAS)
    if qwen_model is None:
        raise RuntimeError(
            f"Comparison model alias {QWEN_MODEL_ALIAS!r} was not found."
        )

    qwen_configuration: dict[str, Any] = {
        "alias": QWEN_MODEL_ALIAS,
        "id": qwen_model.id,
    }
    qwen_loaded = False

    try:
        download_if_needed(qwen_model, "Qwen comparison")
        qwen_model.load()
        qwen_loaded = True
        if not qwen_model.is_loaded:
            raise RuntimeError("Qwen comparison model is not reported as loaded.")
        print(f"[OK] Qwen comparison model loaded: {qwen_model.id}")

        qwen_client = qwen_model.get_chat_client()
        qwen_configuration["settings"] = configure_chat_client(qwen_client)
        qwen_outputs = benchmark_model(QWEN_KEY, qwen_client, shared_records)
    finally:
        if qwen_loaded:
            try:
                qwen_model.unload()
                if qwen_model.is_loaded:
                    raise RuntimeError("Qwen comparison model is still loaded.")
                cleanup["qwen_unloaded"] = True
                print("[OK] Qwen comparison model unloaded.")
            except Exception as error:
                cleanup["qwen_cleanup_error"] = (
                    f"{type(error).__name__}: {error}"
                )

    if len(phi_outputs) != len(shared_records) or len(qwen_outputs) != len(
        shared_records
    ):
        raise RuntimeError("A model did not produce one output per evaluation case.")

    results: list[dict[str, Any]] = []
    for record, phi_output, qwen_output in zip(
        shared_records,
        phi_outputs,
        qwen_outputs,
    ):
        serialized_record = {
            key: value
            for key, value in record.items()
            if key != "augmented_context"
        }
        serialized_record["models"] = {
            PHI_KEY: phi_output,
            QWEN_KEY: qwen_output,
        }
        results.append(serialized_record)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "embedding_model_alias": EMBEDDING_MODEL_ALIAS,
            "embedding_model_id": embedding_model_id,
            "top_k": DEFAULT_TOP_K,
            "case_count": len(cases),
            "generation_settings_requested": GENERATION_SETTINGS,
            "grounding_prompt_sha256": hashlib.sha256(
                GROUNDING_SYSTEM_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "models": {
                PHI_KEY: phi_configuration,
                QWEN_KEY: qwen_configuration,
            },
        },
        "retrieval_summary": summarize_retrieval(shared_records),
        "model_summaries": {
            PHI_KEY: summarize_model(shared_records, phi_outputs),
            QWEN_KEY: summarize_model(shared_records, qwen_outputs),
        },
        "cleanup": cleanup,
        "results": results,
    }

    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(create_report(payload), encoding="utf-8")

    print("\nModel comparison summary")
    for model_key, label in (
        (PHI_KEY, "Phi-3.5 Mini"),
        (QWEN_KEY, "Qwen2.5-0.5B"),
    ):
        summary = payload["model_summaries"][model_key]
        latency = summary["query_latency_seconds"]
        fallback = summary["exact_fallback"]
        print(
            f"{label}: fallback {fallback['count']}/{fallback['total']}; "
            f"latency avg/median/min/max "
            f"{latency['average']:.3f}/{latency['median']:.3f}/"
            f"{latency['minimum']:.3f}/{latency['maximum']:.3f}s"
        )
    print(f"Results: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")

    any_errors = any(
        output["error"]
        for output in phi_outputs + qwen_outputs
        if "error" in output
    )
    cleanup_ok = (
        cleanup["phi_and_embedding_unloaded"] and cleanup["qwen_unloaded"]
    )
    return 1 if any_errors or not cleanup_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
