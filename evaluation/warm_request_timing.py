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

from foundry_local_sdk import __version__ as foundry_local_sdk_version  # noqa: E402
from rag_service import (  # noqa: E402
    CHAT_MODEL_ALIAS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_ALIAS,
    RAGService,
    SourceResult,
)


RESULTS_PATH = PROJECT_ROOT / "evaluation" / "warm_request_timing_results.json"

QUESTION_A = "Why is SQLite suitable for this local project?"
QUESTION_B = "What role does Foundry Local play in the application?"


class TimedEmbeddingClient:
    """Record one embedding call without changing the delegated SDK client."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.last_seconds: float | None = None

    def generate_embedding(self, input_text: str) -> Any:
        started_at = time.perf_counter()
        try:
            return self._delegate.generate_embedding(input_text)
        finally:
            self.last_seconds = time.perf_counter() - started_at

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class TimedChatClient:
    """Record one generation call without changing the delegated SDK client."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.last_seconds: float | None = None

    def complete_chat(self, messages: list[dict[str, str]]) -> Any:
        started_at = time.perf_counter()
        try:
            return self._delegate.complete_chat(messages)
        finally:
            self.last_seconds = time.perf_counter() - started_at

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class InstrumentedRAGService(RAGService):
    """Time boundaries around the existing service without changing its logic."""

    def __init__(self) -> None:
        super().__init__()
        self.startup_sqlite_load_seconds: float | None = None
        self.last_top_chunks_seconds: float | None = None

    def _load_stored_chunks(self):
        started_at = time.perf_counter()
        try:
            return super()._load_stored_chunks()
        finally:
            self.startup_sqlite_load_seconds = time.perf_counter() - started_at

    def get_top_chunks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SourceResult]:
        started_at = time.perf_counter()
        try:
            return super().get_top_chunks(query, top_k)
        finally:
            self.last_top_chunks_seconds = time.perf_counter() - started_at


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
        "variants": [
            {
                "id": variant.id,
                "runtime": runtime_metadata(variant),
                "cached": variant.is_cached,
            }
            for variant in model.variants
        ],
    }


def run_request(
    service: InstrumentedRAGService,
    embedding_client: TimedEmbeddingClient,
    chat_client: TimedChatClient,
    label: str,
    question: str,
) -> dict[str, Any]:
    embedding_client.last_seconds = None
    chat_client.last_seconds = None
    service.last_top_chunks_seconds = None

    started_at = time.perf_counter()
    response = service.answer_query(question, top_k=DEFAULT_TOP_K)
    total_seconds = time.perf_counter() - started_at

    embedding_seconds = embedding_client.last_seconds
    generation_seconds = chat_client.last_seconds
    top_chunks_seconds = service.last_top_chunks_seconds
    if (
        embedding_seconds is None
        or generation_seconds is None
        or top_chunks_seconds is None
    ):
        raise RuntimeError(f"Timing boundaries were not captured for {label}.")
    if len(response["sources"]) != DEFAULT_TOP_K:
        raise RuntimeError(f"{label} did not return exactly {DEFAULT_TOP_K} sources.")
    if not response["answer"].strip():
        raise RuntimeError(f"{label} returned an empty answer.")

    # SQLite is read once during service.start(). The per-request remainder of
    # get_top_chunks() is validation, cosine scoring, sorting, and metadata work.
    retrieval_seconds = max(0.0, top_chunks_seconds - embedding_seconds)
    other_seconds = max(
        0.0,
        total_seconds - top_chunks_seconds - generation_seconds,
    )

    return {
        "label": label,
        "question": question,
        "embedding_seconds": round(embedding_seconds, 6),
        "retrieval_seconds": round(retrieval_seconds, 6),
        "generation_seconds": round(generation_seconds, 6),
        "other_pipeline_seconds": round(other_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "source_ids": [
            f"{source['source']}#{source['chunk_index']}"
            for source in response["sources"]
        ],
        "answer_sha256": hashlib.sha256(
            response["answer"].encode("utf-8")
        ).hexdigest(),
        "answer": response["answer"],
    }


def summarize_question(records: list[dict[str, Any]]) -> dict[str, float]:
    repeat_totals = [record["total_seconds"] for record in records[1:]]
    return {
        "first_total_seconds": records[0]["total_seconds"],
        "repeat_average_total_seconds": round(
            statistics.mean(repeat_totals),
            6,
        ),
        "all_average_total_seconds": round(
            statistics.mean(record["total_seconds"] for record in records),
            6,
        ),
    }


def main() -> int:
    service = InstrumentedRAGService()
    cleanup_succeeded = False

    try:
        service.start()
        if (
            service._manager is None
            or service._embedding_model is None
            or service._chat_model is None
            or service._embedding_client is None
            or service._chat_client is None
        ):
            raise RuntimeError("The RAG service did not expose its loaded resources.")

        manager = service._manager
        embedding_model = service._embedding_model
        chat_model = service._chat_model
        embedding_client = TimedEmbeddingClient(service._embedding_client)
        chat_client = TimedChatClient(service._chat_client)
        service._embedding_client = embedding_client
        service._chat_client = chat_client

        execution_providers = [
            {
                "name": provider.name,
                "is_registered": provider.is_registered,
            }
            for provider in manager.discover_eps()
        ]

        requests: list[dict[str, Any]] = []
        for prefix, question in (("A", QUESTION_A), ("B", QUESTION_B)):
            for run_number in range(1, 4):
                label = f"{prefix}{run_number}"
                print(f"Running {label}: {question}")
                result = run_request(
                    service,
                    embedding_client,
                    chat_client,
                    label,
                    question,
                )
                requests.append(result)
                print(
                    "  embedding={embedding_seconds:.3f}s, "
                    "retrieval={retrieval_seconds:.3f}s, "
                    "generation={generation_seconds:.3f}s, "
                    "total={total_seconds:.3f}s".format(**result)
                )

        question_a_records = requests[:3]
        question_b_records = requests[3:]
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sdk": {
                "distribution": "foundry-local-sdk-winml",
                "version": foundry_local_sdk_version,
            },
            "configuration": {
                "embedding_alias": EMBEDDING_MODEL_ALIAS,
                "chat_alias": CHAT_MODEL_ALIAS,
                "top_k": DEFAULT_TOP_K,
                "model_load_time_included": False,
                "model_download_time_included": False,
                "sqlite_read_scope": (
                    "Stored chunks are read once during service startup; "
                    "retrieval_seconds measures per-request in-memory cosine "
                    "ranking after query embedding."
                ),
            },
            "models": {
                "embedding": describe_model(embedding_model),
                "chat": describe_model(chat_model),
            },
            "discoverable_execution_providers": execution_providers,
            "startup": {
                "sqlite_chunk_load_seconds": round(
                    service.startup_sqlite_load_seconds or 0.0,
                    6,
                ),
                "chunk_count": service.chunk_count,
                "embedding_dimension": service.embedding_dimension,
                "service_load_count": service.load_count,
            },
            "requests": requests,
            "question_summaries": {
                "A": summarize_question(question_a_records),
                "B": summarize_question(question_b_records),
            },
        }
    finally:
        try:
            service.close()
            cleanup_succeeded = not service.is_started
        except Exception as error:
            print(f"[ERROR] Model cleanup failed: {error}")

    payload["cleanup"] = {"models_unloaded": cleanup_succeeded}
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {RESULTS_PATH}")
    print(f"Models unloaded: {cleanup_succeeded}")
    return 0 if cleanup_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
