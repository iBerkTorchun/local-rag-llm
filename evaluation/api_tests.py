from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app, rag_service  # noqa: E402
from rag_service import INSUFFICIENT_CONTEXT_RESPONSE  # noqa: E402


class CheckRecorder:
    def __init__(self) -> None:
        self.total = 0
        self.failures: list[str] = []

    def check(self, condition: bool, label: str) -> None:
        self.total += 1
        if condition:
            print(f"[PASS] {label}")
        else:
            print(f"[FAIL] {label}")
            self.failures.append(label)


def has_valid_sources(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        return False

    return all(
        isinstance(source, dict)
        and isinstance(source.get("source"), str)
        and bool(source["source"].strip())
        and isinstance(source.get("chunk_index"), int)
        and not isinstance(source["chunk_index"], bool)
        and isinstance(source.get("score"), (int, float))
        and not isinstance(source["score"], bool)
        and math.isfinite(float(source["score"]))
        and isinstance(source.get("content"), str)
        and bool(source["content"].strip())
        for source in sources
    )


def has_json_error(response: Any) -> bool:
    payload = response.get_json(silent=True)
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), str)
        and bool(payload["error"].strip())
    )


def main() -> int:
    checks = CheckRecorder()
    initial_load_count = rag_service.load_count
    app.config.update(TESTING=True)

    try:
        with app.test_client() as client:
            health_response = client.get("/api/health")
            checks.check(health_response.status_code == 200, "GET /api/health returns 200")
            checks.check(
                health_response.get_json(silent=True) == {"status": "ok"},
                "GET /api/health returns the expected JSON",
            )

            invalid_requests = [
                ("missing JSON body", lambda: client.post("/api/ask")),
                ("empty object", lambda: client.post("/api/ask", json={})),
                (
                    "empty question",
                    lambda: client.post("/api/ask", json={"question": ""}),
                ),
                (
                    "whitespace-only question",
                    lambda: client.post("/api/ask", json={"question": "   \t"}),
                ),
                (
                    "non-string question",
                    lambda: client.post("/api/ask", json={"question": 42}),
                ),
            ]

            for label, request_factory in invalid_requests:
                response = request_factory()
                checks.check(
                    response.status_code == 400,
                    f"POST /api/ask rejects {label} with 400",
                )
                checks.check(
                    has_json_error(response),
                    f"POST /api/ask returns a JSON error for {label}",
                )

            checks.check(
                rag_service.load_count == initial_load_count,
                "Validation requests do not initialize the RAG models",
            )

            valid_response = client.post(
                "/api/ask",
                json={"question": "Why is SQLite suitable for this local project?"},
            )
            valid_payload = valid_response.get_json(silent=True)
            checks.check(valid_response.status_code == 200, "Valid question returns 200")
            checks.check(
                isinstance(valid_payload, dict)
                and isinstance(valid_payload.get("answer"), str)
                and bool(valid_payload["answer"].strip()),
                "Valid response contains a non-empty answer",
            )
            checks.check(
                has_valid_sources(valid_payload),
                "Valid response contains exactly three complete source objects",
            )

            unsupported_response = client.post(
                "/api/ask",
                json={
                    "question": (
                        "How many vacation days do users of this application receive?"
                    )
                },
            )
            unsupported_payload = unsupported_response.get_json(silent=True)
            checks.check(
                unsupported_response.status_code == 200,
                "Unsupported valid question remains an HTTP 200 response",
            )
            checks.check(
                isinstance(unsupported_payload, dict)
                and unsupported_payload.get("answer")
                == INSUFFICIENT_CONTEXT_RESPONSE,
                "Unsupported question returns the exact grounded fallback",
            )
            checks.check(
                has_valid_sources(unsupported_payload),
                "Unsupported response still contains three retrieved sources",
            )
            checks.check(
                rag_service.load_count == initial_load_count + 1,
                "Both valid requests reuse one loaded RAG service",
            )
    finally:
        try:
            rag_service.close()
            checks.check(
                not rag_service.is_started,
                "RAG models unload after API verification",
            )
        except Exception as error:
            print(f"[FAIL] RAG model cleanup raised {type(error).__name__}: {error}")
            checks.total += 1
            checks.failures.append("RAG model cleanup")

    print(f"\nAPI checks: {checks.total - len(checks.failures)}/{checks.total} passed")
    if checks.failures:
        print("Failed checks:")
        for failure in checks.failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
