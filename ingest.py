import json
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from chunking import chunk_text


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DATABASE_PATH = PROJECT_ROOT / "data" / "rag.db"
MODEL_ALIAS = "qwen3-embedding-0.6b"

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL
)
"""


def show_download_progress(percent: float) -> None:
    print(f"\rDownloading {MODEL_ALIAS}: {percent:5.1f}%", end="", flush=True)


def collect_chunks(document_paths: list[Path]) -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []

    for document_path in document_paths:
        document_text = document_path.read_text(encoding="utf-8")
        document_chunks = chunk_text(document_text)
        for chunk_index, content in enumerate(document_chunks):
            records.append((document_path.name, chunk_index, content))

    if not records or any(not content.strip() for _, _, content in records):
        raise RuntimeError("Every source document must produce non-empty chunks.")

    return records


def store_chunks(
    chunk_records: list[tuple[str, int, str]],
    embeddings: list[list[float]],
) -> tuple[int, list[tuple[int, str, int, str]]]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.execute(CREATE_CHUNKS_TABLE)
        connection.commit()

        rows_to_insert = [
            (source, chunk_index, content, json.dumps(embedding))
            for (source, chunk_index, content), embedding in zip(
                chunk_records, embeddings
            )
        ]

        try:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM chunks")
            connection.execute(
                "DELETE FROM sqlite_sequence WHERE name = ?",
                ("chunks",),
            )
            connection.executemany(
                """
                INSERT INTO chunks (source, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

        row_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        preview_rows = connection.execute(
            """
            SELECT id, source, chunk_index, content
            FROM chunks
            ORDER BY id
            LIMIT ?
            """,
            (5,),
        ).fetchall()
        return row_count, preview_rows
    finally:
        connection.close()


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

        document_paths = sorted(
            KNOWLEDGE_BASE_DIR.glob("*.md"),
            key=lambda path: path.name.casefold(),
        )
        if not document_paths:
            raise RuntimeError(
                f"No Markdown documents found in {KNOWLEDGE_BASE_DIR}."
            )

        chunk_records = collect_chunks(document_paths)
        chunk_texts = [content for _, _, content in chunk_records]

        client = model.get_embedding_client()
        embedding_response = client.generate_embeddings(chunk_texts)
        embedding_items = sorted(
            embedding_response.data,
            key=lambda item: item.index,
        )
        embeddings = [item.embedding for item in embedding_items]

        if len(embeddings) != len(chunk_records):
            raise RuntimeError("The embedding count does not match the chunk count.")
        if not embeddings or not embeddings[0]:
            raise RuntimeError("The model returned an empty embedding vector.")

        embedding_dimensionality = len(embeddings[0])
        if any(
            len(embedding) != embedding_dimensionality
            for embedding in embeddings
        ):
            raise RuntimeError("The embeddings have inconsistent dimensionality.")

        row_count, preview_rows = store_chunks(chunk_records, embeddings)
        if row_count != len(chunk_records):
            raise RuntimeError("The SQLite row count does not match the chunk count.")

        print("\nIngestion summary:")
        print(f"Source documents: {len(document_paths)}")
        print("Discovered files: " + ", ".join(path.name for path in document_paths))
        print(f"Total chunks: {len(chunk_records)}")
        print(f"Embedding dimensionality: {embedding_dimensionality}")
        print(f"Rows stored in SQLite: {row_count}")
        print(f"Database path: {DATABASE_PATH}")

        print("\nStored row preview:")
        for row_id, source, chunk_index, content in preview_rows:
            compact_content = " ".join(content.split())
            if len(compact_content) > 80:
                compact_content = compact_content[:77] + "..."
            print(f"{row_id} | {source} | chunk {chunk_index} | {compact_content}")
    finally:
        model.unload()
        if model.is_loaded:
            raise RuntimeError("Model unload finished, but the model is still reported as loaded.")
        print("[OK] Embedding model unloaded.")


if __name__ == "__main__":
    main()
