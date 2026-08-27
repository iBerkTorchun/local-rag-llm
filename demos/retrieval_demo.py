import json
import math
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "rag.db"
MODEL_ALIAS = "qwen3-embedding-0.6b"
DEFAULT_TOP_K = 3

TEST_QUERIES = [
    "Why is SQLite a good database for this local application?",
    "How does this project reduce hallucinations when answering questions?",
    "What does Foundry Local do in this system?",
    "What should the assistant do when the supplied information is insufficient?",
]

StoredChunk = tuple[int, str, int, str, list[float]]
RetrievalResult = tuple[float, int, str, int, str]


def show_download_progress(percent: float) -> None:
    print(f"\rDownloading {MODEL_ALIAS}: {percent:5.1f}%", end="", flush=True)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")

    dot_product = sum(a_value * b_value for a_value, b_value in zip(a, b))
    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def load_stored_chunks() -> tuple[list[StoredChunk], int]:
    if not DATABASE_PATH.is_file():
        raise RuntimeError(f"Database not found: {DATABASE_PATH}")

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        rows = connection.execute(
            """
            SELECT id, source, chunk_index, content, embedding
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise RuntimeError("The chunks table does not contain any rows.")

    stored_chunks: list[StoredChunk] = []
    expected_dimension: int | None = None

    for chunk_id, source, chunk_index, content, serialized_embedding in rows:
        try:
            embedding = json.loads(serialized_embedding)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Chunk {chunk_id} has invalid embedding JSON.") from error

        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Chunk {chunk_id} has an empty embedding.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in embedding
        ):
            raise RuntimeError(f"Chunk {chunk_id} has a non-numeric embedding value.")

        numeric_embedding = [float(value) for value in embedding]
        if any(not math.isfinite(value) for value in numeric_embedding):
            raise RuntimeError(f"Chunk {chunk_id} has a non-finite embedding value.")

        if expected_dimension is None:
            expected_dimension = len(numeric_embedding)
        elif len(numeric_embedding) != expected_dimension:
            raise RuntimeError("Stored embeddings have inconsistent dimensionality.")

        stored_chunks.append(
            (chunk_id, source, chunk_index, content, numeric_embedding)
        )

    if expected_dimension is None:
        raise RuntimeError("Could not determine stored embedding dimensionality.")

    return stored_chunks, expected_dimension


def get_top_chunks(
    query: str,
    client,
    stored_chunks: list[StoredChunk],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_response = client.generate_embedding(query)
    if len(query_response.data) != 1:
        raise RuntimeError("Expected exactly one query embedding.")

    query_embedding = query_response.data[0].embedding
    stored_dimension = len(stored_chunks[0][4])
    if not query_embedding or len(query_embedding) != stored_dimension:
        raise RuntimeError("Query and stored embeddings have different dimensionality.")

    ranked_results: list[RetrievalResult] = []
    for chunk_id, source, chunk_index, content, stored_embedding in stored_chunks:
        score = cosine_similarity(query_embedding, stored_embedding)
        if not math.isfinite(score):
            raise RuntimeError("A cosine similarity score is not finite.")
        ranked_results.append((score, chunk_id, source, chunk_index, content))

    ranked_results.sort(key=lambda result: result[0], reverse=True)
    top_results = ranked_results[:top_k]

    expected_count = min(top_k, len(stored_chunks))
    if len(top_results) != expected_count:
        raise RuntimeError("The retriever returned an unexpected number of results.")
    if any(
        top_results[index][0] < top_results[index + 1][0]
        for index in range(len(top_results) - 1)
    ):
        raise RuntimeError("Retrieval results are not sorted by decreasing score.")
    if any(
        not source.strip() or not content.strip()
        for _, _, source, _, content in top_results
    ):
        raise RuntimeError("A retrieval result is missing source or content metadata.")

    return top_results


def print_results(query: str, results: list[RetrievalResult]) -> None:
    print(f"\nQuery:\n{query}")
    print(f"\nTop {len(results)} chunks:")

    for rank, (score, chunk_id, source, chunk_index, content) in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{rank}. {score:.4f} | {source} | chunk {chunk_index} | id {chunk_id}"
        )
        print("   " + content.replace("\n", "\n   "))


def main() -> None:
    FoundryLocalManager.initialize(Configuration(app_name="foundry_local_rag"))
    manager = FoundryLocalManager.instance
    print("[OK] SDK initialization succeeded.")

    model = manager.catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model alias {MODEL_ALIAS!r} was not found in the catalog.")
    print(f"[OK] Model found: {model.id}")

    if model.is_cached:
        print("[OK] Model is already downloaded.")
    else:
        model.download(show_download_progress)
        print()
        if not model.is_cached:
            raise RuntimeError("Model download finished, but the model is not cached.")
        print("[OK] Model downloaded.")

    model.load()
    try:
        if not model.is_loaded:
            raise RuntimeError("Model load finished, but the model is not reported as loaded.")
        print("[OK] Model loaded.")

        stored_chunks, stored_dimension = load_stored_chunks()
        client = model.get_embedding_client()

        print(f"[OK] Chunks loaded from SQLite: {len(stored_chunks)}")
        print(f"[OK] Stored embedding dimensionality: {stored_dimension}")

        for query in TEST_QUERIES:
            results = get_top_chunks(query, client, stored_chunks)
            print_results(query, results)

        print(
            f"\n[OK] All stored/query embeddings used dimensionality {stored_dimension}."
        )
        print("[OK] Result count, finite-score, ordering, and metadata checks passed.")
    finally:
        model.unload()
        if model.is_loaded:
            raise RuntimeError("Model unload finished, but the model is still reported as loaded.")
        print("[OK] Embedding model unloaded.")


if __name__ == "__main__":
    main()
