from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation"
RESULTS_PATH = EVALUATION_DIRECTORY / "empty_cache_cuda_results.json"
REPORT_PATH = EVALUATION_DIRECTORY / "empty_cache_cuda_report.md"
CASES_PATH = EVALUATION_DIRECTORY / "evaluation_cases.json"
CPU_RESULTS_PATH = EVALUATION_DIRECTORY / "tuned_results.json"
CPU_TIMING_PATH = EVALUATION_DIRECTORY / "warm_request_timing_results.json"
PRODUCTION_MODEL_CACHE = (
    Path.home() / ".foundry_local_rag" / "cache" / "models"
)

EMBEDDING_ALIAS = "qwen3-embedding-0.6b"
CHAT_ALIAS = "phi-3.5-mini"
CUDA_PROVIDER = "CUDAExecutionProvider"
EXPERIMENT_APP_NAME = "foundry_local_rag_empty_cache_experiment"
DIFFICULT_CASE_IDS = {
    "prompt_01",
    "architecture_01",
    "unknown_02",
    "rag_02",
    "embeddings_01",
    "embeddings_02",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import build_summary, evaluate_case, load_cases  # noqa: E402
from foundry_local_sdk import Configuration, FoundryLocalManager  # noqa: E402
from rag_service import DEFAULT_TOP_K, cosine_similarity  # noqa: E402
from warm_request_timing import (  # noqa: E402
    QUESTION_A,
    QUESTION_B,
    InstrumentedRAGService,
    TimedChatClient,
    TimedEmbeddingClient,
    run_request,
    summarize_question,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def production_file_hashes() -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "rag_service.py",
        PROJECT_ROOT / "ingest.py",
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "data" / "rag.db",
        CASES_PATH,
        *sorted((PROJECT_ROOT / "knowledge_base").glob("*.md")),
        *sorted((PROJECT_ROOT / "static").rglob("*.*")),
        *sorted((PROJECT_ROOT / "templates").rglob("*.*")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def directory_metadata_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "metadata_sha256": None,
        }

    entries = [
        {
            "path": str(file.relative_to(path)),
            "size": file.stat().st_size,
            "modified_ns": file.stat().st_mtime_ns,
        }
        for file in sorted(path.rglob("*"), key=lambda item: str(item).casefold())
        if file.is_file()
    ]
    serialized = json.dumps(entries, sort_keys=True).encode("utf-8")
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "metadata_sha256": hashlib.sha256(serialized).hexdigest().upper(),
    }


def runtime_metadata(model: Any) -> dict[str, str] | None:
    runtime = model.info.runtime
    if runtime is None:
        return None
    return {
        "device_type": runtime.device_type.value,
        "execution_provider": runtime.execution_provider,
    }


def describe_model(model: Any) -> dict[str, Any]:
    return {
        "alias": model.alias,
        "selected_id": model.id,
        "selected_runtime": runtime_metadata(model),
        "selected_cached": model.is_cached,
        "all_variants_uncached": all(
            not variant.is_cached for variant in model.variants
        ),
        "variants": [
            {
                "id": variant.id,
                "runtime": runtime_metadata(variant),
                "cached": variant.is_cached,
            }
            for variant in model.variants
        ],
    }


def discover(manager: FoundryLocalManager) -> list[dict[str, Any]]:
    return [
        {"name": provider.name, "is_registered": provider.is_registered}
        for provider in manager.discover_eps()
    ]


def is_cuda_model(model: Any) -> bool:
    runtime = runtime_metadata(model)
    if runtime is None:
        return False
    return (
        runtime["device_type"].upper() == "GPU"
        and runtime["execution_provider"] == CUDA_PROVIDER
    )


def percentage_reduction(cpu_seconds: float, gpu_seconds: float) -> float:
    return round((cpu_seconds - gpu_seconds) / cpu_seconds * 100.0, 2)


def download_selected_model(model: Any, label: str) -> None:
    if model.is_cached:
        raise RuntimeError(
            f"{label} was unexpectedly cached before the isolated download."
        )

    last_reported = -10

    def show_progress(percent: float) -> None:
        nonlocal last_reported
        whole_percent = max(0, min(100, int(percent)))
        if whole_percent == 100 or whole_percent >= last_reported + 10:
            print(f"{label} download: {whole_percent}%")
            last_reported = whole_percent

    model.download(show_progress)
    if not model.is_cached:
        raise RuntimeError(f"{label} download completed but is_cached is false.")


def validate_embeddings(embeddings: list[list[float]]) -> int:
    if not embeddings or not embeddings[0]:
        raise RuntimeError("The accelerated embedding batch is empty.")
    dimension = len(embeddings[0])
    if any(len(embedding) != dimension for embedding in embeddings):
        raise RuntimeError("Accelerated embeddings have inconsistent dimensions.")
    if any(
        not math.isfinite(float(value))
        for embedding in embeddings
        for value in embedding
    ):
        raise RuntimeError("Accelerated embeddings contain a non-finite value.")
    return dimension


def load_cpu_results() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evaluation_payload = json.loads(CPU_RESULTS_PATH.read_text(encoding="utf-8"))
    timing_payload = json.loads(CPU_TIMING_PATH.read_text(encoding="utf-8"))
    return (
        {result["id"]: result for result in evaluation_payload["results"]},
        timing_payload,
    )


def compare_retrieval(
    cases: list[dict[str, Any]],
    cpu_results: dict[str, dict[str, Any]],
    gpu_results: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu_by_id = {result["id"]: result for result in gpu_results}
    comparisons: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["id"]
        cpu_sources = cpu_results[case_id]["retrieved_sources"]
        gpu_sources = gpu_by_id[case_id]["retrieved_sources"]
        cpu_ids = [
            f"{source['source']}#{source['chunk_index']}" for source in cpu_sources
        ]
        gpu_ids = [
            f"{source['source']}#{source['chunk_index']}" for source in gpu_sources
        ]
        cpu_scores = {
            f"{source['source']}#{source['chunk_index']}": float(source["score"])
            for source in cpu_sources
        }
        gpu_scores = {
            f"{source['source']}#{source['chunk_index']}": float(source["score"])
            for source in gpu_sources
        }
        common_ids = sorted(set(cpu_ids) & set(gpu_ids))
        score_comparisons = [
            {
                "source_id": source_id,
                "cpu_score": cpu_scores[source_id],
                "cuda_score": gpu_scores[source_id],
                "absolute_difference": abs(
                    cpu_scores[source_id] - gpu_scores[source_id]
                ),
            }
            for source_id in common_ids
        ]
        comparisons.append(
            {
                "id": case_id,
                "cpu_order": cpu_ids,
                "cuda_order": gpu_ids,
                "top_1_same": cpu_ids[:1] == gpu_ids[:1],
                "top_3_set_same": set(cpu_ids) == set(gpu_ids),
                "top_3_order_same": cpu_ids == gpu_ids,
                "score_comparisons": score_comparisons,
            }
        )

    all_score_differences = [
        score["absolute_difference"]
        for comparison in comparisons
        for score in comparison["score_comparisons"]
    ]
    return {
        "cases": comparisons,
        "top_1_same_count": sum(item["top_1_same"] for item in comparisons),
        "top_3_set_same_count": sum(
            item["top_3_set_same"] for item in comparisons
        ),
        "top_3_order_same_count": sum(
            item["top_3_order_same"] for item in comparisons
        ),
        "case_count": len(comparisons),
        "maximum_common_source_score_difference": (
            max(all_score_differences) if all_score_differences else None
        ),
        "material_ranking_difference": any(
            not item["top_1_same"] or not item["top_3_set_same"]
            for item in comparisons
        ),
    }


def timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    averages = {
        field: round(statistics.mean(record[field] for record in records), 6)
        for field in (
            "embedding_seconds",
            "retrieval_seconds",
            "generation_seconds",
            "total_seconds",
        )
    }
    totals = [record["total_seconds"] for record in records]
    return {
        "averages": averages,
        "total_median_seconds": round(statistics.median(totals), 6),
        "total_minimum_seconds": round(min(totals), 6),
        "total_maximum_seconds": round(max(totals), 6),
    }


def run_cuda_pipeline(
    manager: FoundryLocalManager,
    embedding_model: Any,
    chat_model: Any,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "performed": True,
        "document_embeddings_persisted": False,
        "production_database_used_read_only": True,
        "top_k": DEFAULT_TOP_K,
        "generation_settings": {
            "temperature": 0.0,
            "random_seed": 0,
            "max_tokens": 96,
        },
    }
    service = InstrumentedRAGService()
    cleanup_succeeded = False

    try:
        download_selected_model(embedding_model, "Embedding model")
        download_selected_model(chat_model, "Chat model")
        result["models_after_download"] = {
            "embedding": describe_model(embedding_model),
            "chat": describe_model(chat_model),
        }
        if not is_cuda_model(embedding_model) or not is_cuda_model(chat_model):
            raise RuntimeError(
                "A selected model changed away from CUDA after download."
            )

        service._manager = manager
        service._embedding_model = embedding_model
        service._chat_model = chat_model

        embedding_model.load()
        service._embedding_model_loaded = True
        if not embedding_model.is_loaded:
            raise RuntimeError("The CUDA embedding model did not report loaded.")

        chat_model.load()
        service._chat_model_loaded = True
        if not chat_model.is_loaded:
            raise RuntimeError("The CUDA chat model did not report loaded.")

        cpu_chunks, cpu_dimension = service._load_stored_chunks()
        raw_embedding_client = embedding_model.get_embedding_client()
        raw_chat_client = chat_model.get_chat_client()
        raw_chat_client.settings.temperature = 0.0
        raw_chat_client.settings.random_seed = 0
        raw_chat_client.settings.max_tokens = 96

        chunk_contents = [chunk[3] for chunk in cpu_chunks]
        corpus_started_at = time.perf_counter()
        response = raw_embedding_client.generate_embeddings(chunk_contents)
        corpus_seconds = time.perf_counter() - corpus_started_at
        items = sorted(response.data, key=lambda item: item.index)
        gpu_embeddings = [item.embedding for item in items]
        if len(gpu_embeddings) != len(cpu_chunks):
            raise RuntimeError(
                "Accelerated document embedding count does not match chunk count."
            )
        gpu_dimension = validate_embeddings(gpu_embeddings)
        if gpu_dimension != cpu_dimension:
            raise RuntimeError(
                "CPU and CUDA document embedding dimensionality differs."
            )

        cross_provider_cosines = [
            cosine_similarity(cpu_chunk[4], gpu_embedding)
            for cpu_chunk, gpu_embedding in zip(cpu_chunks, gpu_embeddings)
        ]
        service._stored_chunks = [
            (
                chunk_id,
                source,
                chunk_index,
                content,
                gpu_embedding,
            )
            for (
                chunk_id,
                source,
                chunk_index,
                content,
                _cpu_embedding,
            ), gpu_embedding in zip(cpu_chunks, gpu_embeddings)
        ]
        service._embedding_dimension = gpu_dimension
        timed_embedding_client = TimedEmbeddingClient(raw_embedding_client)
        timed_chat_client = TimedChatClient(raw_chat_client)
        service._embedding_client = timed_embedding_client
        service._chat_client = timed_chat_client
        service._started = True
        service._load_count = 1

        result["in_memory_cuda_corpus"] = {
            "chunk_count": len(gpu_embeddings),
            "embedding_dimension": gpu_dimension,
            "all_values_finite": True,
            "generation_seconds": round(corpus_seconds, 6),
            "cpu_cuda_vector_cosine": {
                "average": statistics.mean(cross_provider_cosines),
                "minimum": min(cross_provider_cosines),
                "maximum": max(cross_provider_cosines),
            },
        }

        timing_records: list[dict[str, Any]] = []
        for prefix, question in (("A", QUESTION_A), ("B", QUESTION_B)):
            for run_number in range(1, 4):
                label = f"{prefix}{run_number}"
                print(f"Running CUDA timing {label}: {question}")
                request_result = run_request(
                    service,
                    timed_embedding_client,
                    timed_chat_client,
                    label,
                    question,
                )
                timing_records.append(request_result)
                print(
                    "  embedding={embedding_seconds:.3f}s, "
                    "retrieval={retrieval_seconds:.3f}s, "
                    "generation={generation_seconds:.3f}s, "
                    "total={total_seconds:.3f}s".format(**request_result)
                )

        _cpu_results, cpu_timing = load_cpu_results()
        cpu_records = cpu_timing["requests"]
        cpu_summary = timing_summary(cpu_records)
        cuda_summary = timing_summary(timing_records)
        result["timing"] = {
            "questions": {"A": QUESTION_A, "B": QUESTION_B},
            "cuda_requests": timing_records,
            "cuda_summary": cuda_summary,
            "cpu_baseline_summary": cpu_summary,
            "question_summaries": {
                "A": summarize_question(timing_records[:3]),
                "B": summarize_question(timing_records[3:]),
            },
            "percentage_reduction": {
                "embedding": percentage_reduction(
                    cpu_summary["averages"]["embedding_seconds"],
                    cuda_summary["averages"]["embedding_seconds"],
                ),
                "generation": percentage_reduction(
                    cpu_summary["averages"]["generation_seconds"],
                    cuda_summary["averages"]["generation_seconds"],
                ),
                "total": percentage_reduction(
                    cpu_summary["averages"]["total_seconds"],
                    cuda_summary["averages"]["total_seconds"],
                ),
            },
        }

        evaluation_results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            print(f"Running CUDA evaluation [{index:02d}/{len(cases):02d}] {case['id']}")
            evaluation_results.append(evaluate_case(service, case))
        evaluation_summary = build_summary(evaluation_results)
        cpu_results, _cpu_timing = load_cpu_results()
        result["evaluation"] = {
            "summary": evaluation_summary,
            "results": evaluation_results,
            "difficult_case_comparison": [
                {
                    "id": gpu_result["id"],
                    "cpu_answer": cpu_results[gpu_result["id"]][
                        "generated_answer"
                    ],
                    "cuda_answer": gpu_result["generated_answer"],
                    "cpu_sources": cpu_results[gpu_result["id"]][
                        "retrieved_sources"
                    ],
                    "cuda_sources": gpu_result["retrieved_sources"],
                    "human_review_required": True,
                }
                for gpu_result in evaluation_results
                if gpu_result["id"] in DIFFICULT_CASE_IDS
            ],
        }
        result["retrieval_consistency"] = compare_retrieval(
            cases,
            cpu_results,
            evaluation_results,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            service.close()
            cleanup_succeeded = not service.is_started
        except Exception as cleanup_error:
            result["cleanup_error"] = (
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        result["models_unloaded"] = cleanup_succeeded

    return result


def model_row(description: dict[str, Any]) -> str:
    runtime = description["selected_runtime"] or {}
    return (
        f"`{description['selected_id']}` | "
        f"{runtime.get('device_type', 'unknown')} | "
        f"`{runtime.get('execution_provider', 'unknown')}` | "
        f"{'yes' if description['selected_cached'] else 'no'}"
    )


def create_report(payload: dict[str, Any]) -> str:
    selection = payload.get("selection_after_registration", {})
    lines = [
        "# Empty-Cache CUDA Alias-Selection Experiment",
        "",
        "## Configuration",
        "",
        f"- SDK: `foundry-local-sdk-winml=={payload['sdk']['version']}`",
        f"- Isolated model cache: `{payload['isolated_cache']['model_cache_dir']}`",
        "- Model cache was empty before Foundry Local initialization: yes",
        "- Registered provider: `CUDAExecutionProvider` only",
        "- Concrete model/device/provider selection was not forced",
        "",
        "## Registration",
        "",
        f"- Success: `{str(payload.get('registration', {}).get('success')).lower()}`",
        f"- CUDA registered in the same process: `{str(payload.get('cuda_registered')).lower()}`",
        f"- Status: {payload.get('registration', {}).get('status', 'not available')}",
        "",
        "## Fresh-cache automatic selection",
        "",
        "| Alias | Selected concrete ID | Device | Execution provider | Cached before download |",
        "| --- | --- | --- | --- | --- |",
    ]
    if selection:
        lines.extend(
            [
                f"| `{EMBEDDING_ALIAS}` | {model_row(selection['embedding'])} |",
                f"| `{CHAT_ALIAS}` | {model_row(selection['chat'])} |",
                "",
                f"**Outcome:** {payload['outcome']['summary']}",
            ]
        )
    else:
        lines.extend(["", "Selection was not completed."])

    benchmark = payload.get("cuda_pipeline", {})
    lines.extend(["", "## Performance", ""])
    if benchmark.get("performed") and "timing" in benchmark:
        timing = benchmark["timing"]
        cpu = timing["cpu_baseline_summary"]["averages"]
        cuda = timing["cuda_summary"]["averages"]
        reduction = timing["percentage_reduction"]
        lines.extend(
            [
                "| Stage | CPU average | CUDA average | Reduction |",
                "| --- | ---: | ---: | ---: |",
                f"| Query embedding | {cpu['embedding_seconds']:.3f}s | {cuda['embedding_seconds']:.3f}s | {reduction['embedding']:.2f}% |",
                f"| Retrieval | {cpu['retrieval_seconds']:.3f}s | {cuda['retrieval_seconds']:.3f}s | not a tuning target |",
                f"| Generation | {cpu['generation_seconds']:.3f}s | {cuda['generation_seconds']:.3f}s | {reduction['generation']:.2f}% |",
                f"| Total | {cpu['total_seconds']:.3f}s | {cuda['total_seconds']:.3f}s | {reduction['total']:.2f}% |",
            ]
        )
    else:
        lines.append(
            benchmark.get(
                "reason",
                "No accelerated benchmark was performed because both aliases did not automatically select CUDA.",
            )
        )

    lines.extend(["", "## Retrieval consistency and grounding", ""])
    if "retrieval_consistency" in benchmark:
        consistency = benchmark["retrieval_consistency"]
        summary = benchmark["evaluation"]["summary"]
        lines.extend(
            [
                f"- Embedding dimension: {benchmark['in_memory_cuda_corpus']['embedding_dimension']}",
                f"- Same Top-1: {consistency['top_1_same_count']}/{consistency['case_count']}",
                f"- Same Top-3 set: {consistency['top_3_set_same_count']}/{consistency['case_count']}",
                f"- Same Top-3 order: {consistency['top_3_order_same_count']}/{consistency['case_count']}",
                f"- Material ranking difference: {'yes' if consistency['material_ranking_difference'] else 'no'}",
                f"- Hit@1: {summary['retrieval']['hit_at_1']['count']}/{summary['retrieval']['hit_at_1']['total']}",
                f"- Hit@3: {summary['retrieval']['hit_at_3']['count']}/{summary['retrieval']['hit_at_3']['total']}",
                f"- Exact fallback: {summary['grounding']['fallback_success']['count']}/{summary['grounding']['fallback_success']['total']}",
                "- Semantic answer and grounding quality require human review.",
            ]
        )
    else:
        lines.append("Not run because the automatic CUDA-selection gate was not met.")

    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Production files unchanged: {'yes' if payload['integrity']['production_files_unchanged'] else 'no'}",
            f"- Production model-cache metadata unchanged: {'yes' if payload['integrity']['production_model_cache_unchanged'] else 'no'}",
            f"- Temporary cache removed: {'yes' if payload['cleanup']['temporary_root_removed'] else 'no'}",
            f"- Experiment models unloaded: {payload['cleanup']['models_unloaded_status']}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    if RESULTS_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment artifact.")

    cases = load_cases()
    temporary_root = Path(
        tempfile.mkdtemp(prefix="foundry-local-empty-cache-")
    ).resolve()
    model_cache_dir = temporary_root / "models"
    app_data_dir = temporary_root / "app-data"
    logs_dir = temporary_root / "logs"
    model_cache_dir.mkdir(parents=True)
    initial_cache_entries = list(model_cache_dir.iterdir())
    if initial_cache_entries:
        raise RuntimeError("The isolated model cache was not initially empty.")

    production_hashes_before = production_file_hashes()
    production_cache_before = directory_metadata_fingerprint(
        PRODUCTION_MODEL_CACHE
    )
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sdk": {
            "distribution": "foundry-local-sdk-winml",
            "version": version("foundry-local-sdk-winml"),
            "core_distribution": "foundry-local-core-winml",
            "core_version": version("foundry-local-core-winml"),
        },
        "configuration": {
            "app_name": EXPERIMENT_APP_NAME,
            "app_data_dir": str(app_data_dir),
            "model_cache_dir": str(model_cache_dir),
            "logs_dir": str(logs_dir),
            "lookup": "manager.catalog.get_model(alias)",
            "select_variant_called": False,
            "device_or_provider_constraint": False,
        },
        "isolated_cache": {
            "temporary_root": str(temporary_root),
            "model_cache_dir": str(model_cache_dir),
            "initial_entries": [],
            "initially_empty": True,
            "production_model_cache": str(PRODUCTION_MODEL_CACHE),
        },
    }
    exit_code = 0
    models_unloaded_status = "not applicable; no models loaded"

    try:
        FoundryLocalManager.initialize(
            Configuration(
                app_name=EXPERIMENT_APP_NAME,
                app_data_dir=str(app_data_dir),
                model_cache_dir=str(model_cache_dir),
                logs_dir=str(logs_dir),
            )
        )
        manager = FoundryLocalManager.instance
        payload["execution_providers_before_registration"] = discover(manager)

        last_ep_progress = -10

        def show_ep_progress(name: str, percent: float) -> None:
            nonlocal last_ep_progress
            whole_percent = max(0, min(100, int(percent)))
            if whole_percent == 100 or whole_percent >= last_ep_progress + 10:
                print(f"{name}: {whole_percent}%")
                last_ep_progress = whole_percent

        registration = manager.download_and_register_eps(
            names=[CUDA_PROVIDER],
            progress_callback=show_ep_progress,
        )
        payload["registration"] = {
            "success": registration.success,
            "status": registration.status,
            "registered_eps": registration.registered_eps,
            "failed_eps": registration.failed_eps,
        }
        providers_after = discover(manager)
        payload["execution_providers_after_registration"] = providers_after
        cuda_registered = any(
            provider["name"] == CUDA_PROVIDER and provider["is_registered"]
            for provider in providers_after
        )
        payload["cuda_registered"] = cuda_registered
        if (
            not registration.success
            or registration.failed_eps
            or not cuda_registered
        ):
            raise RuntimeError("CUDA registration did not succeed in this process.")

        refreshed_models = manager.catalog.list_models()
        embedding_model = manager.catalog.get_model(EMBEDDING_ALIAS)
        chat_model = manager.catalog.get_model(CHAT_ALIAS)
        if embedding_model is None or chat_model is None:
            raise RuntimeError("A required model alias was not found after refresh.")

        embedding_description = describe_model(embedding_model)
        chat_description = describe_model(chat_model)
        payload["catalog_refresh"] = {
            "method": "manager.catalog.list_models()",
            "model_count": len(refreshed_models),
        }
        payload["selection_after_registration"] = {
            "embedding": embedding_description,
            "chat": chat_description,
        }

        all_variants_uncached = (
            embedding_description["all_variants_uncached"]
            and chat_description["all_variants_uncached"]
        )
        both_cuda = is_cuda_model(embedding_model) and is_cuda_model(chat_model)
        if both_cuda and all_variants_uncached:
            payload["outcome"] = {
                "category": 1,
                "summary": (
                    "Empty cache caused alias-based CUDA selection; cached CPU "
                    "variants were influencing the previous selection."
                ),
            }
            payload["cuda_pipeline"] = run_cuda_pipeline(
                manager,
                embedding_model,
                chat_model,
                cases,
            )
            models_unloaded_status = (
                "yes"
                if payload["cuda_pipeline"].get("models_unloaded")
                else "no"
            )
            if (
                payload["cuda_pipeline"].get("error")
                or not payload["cuda_pipeline"].get("models_unloaded")
            ):
                exit_code = 1
        elif not both_cuda and all_variants_uncached:
            selected_runtimes = {
                runtime_metadata(embedding_model)["device_type"],
                runtime_metadata(chat_model)["device_type"],
            }
            if selected_runtimes == {"CPU"}:
                category = 2
                summary = (
                    "Empty cache still selected CPU despite registered CUDA and "
                    "visible, uncached CUDA variants."
                )
            else:
                category = 3
                summary = (
                    "The empty cache produced mixed automatic device selection; "
                    "both aliases did not select CUDA."
                )
            payload["outcome"] = {"category": category, "summary": summary}
            payload["cuda_pipeline"] = {
                "performed": False,
                "reason": (
                    "Both aliases did not automatically select CUDA, so no model "
                    "was downloaded or forced."
                ),
            }
        else:
            payload["outcome"] = {
                "category": 3,
                "summary": (
                    "The isolated-cache precondition was violated because at least "
                    "one catalog variant reported cached before download."
                ),
            }
            payload["cuda_pipeline"] = {
                "performed": False,
                "reason": "The empty-cache precondition was not satisfied.",
            }
            exit_code = 1
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        production_hashes_after = production_file_hashes()
        production_cache_after = directory_metadata_fingerprint(
            PRODUCTION_MODEL_CACHE
        )
        payload["integrity"] = {
            "production_files_before": production_hashes_before,
            "production_files_after": production_hashes_after,
            "production_files_unchanged": (
                production_hashes_before == production_hashes_after
            ),
            "production_model_cache_before": production_cache_before,
            "production_model_cache_after": production_cache_after,
            "production_model_cache_unchanged": (
                production_cache_before == production_cache_after
            ),
        }

        temporary_root_removed = False
        cleanup_error: str | None = None
        try:
            system_temp = Path(tempfile.gettempdir()).resolve()
            if (
                system_temp not in temporary_root.parents
                or not temporary_root.name.startswith("foundry-local-empty-cache-")
            ):
                raise RuntimeError("Refusing to remove an unexpected temporary path.")
            shutil.rmtree(temporary_root)
            temporary_root_removed = not temporary_root.exists()
        except Exception as error:
            cleanup_error = f"{type(error).__name__}: {error}"
            exit_code = 1

        payload["cleanup"] = {
            "models_unloaded_status": models_unloaded_status,
            "temporary_root_removed": temporary_root_removed,
            "error": cleanup_error,
        }
        RESULTS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        REPORT_PATH.write_text(create_report(payload), encoding="utf-8")

    print(f"Results: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Outcome: {payload.get('outcome', {}).get('summary', 'not available')}")
    print(f"Temporary cache removed: {payload['cleanup']['temporary_root_removed']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
