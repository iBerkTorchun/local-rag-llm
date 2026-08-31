import json
import math
import sqlite3
from pathlib import Path
from typing import TypedDict

from foundry_local_sdk import Configuration, FoundryLocalManager


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "rag.db"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"
DEFAULT_TOP_K = 3

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The information is not available in the supplied context."
)
GROUNDING_SYSTEM_INSTRUCTION = (
    "You are a grounded question-answering assistant. Use only the supplied "
    "retrieved context. If the context contains information that directly supports "
    "an answer, answer with that information. A faithful paraphrase is allowed; do "
    "not return the fallback merely because the context and question use different "
    "wording. Answer only the supported portion and do not fill gaps with outside "
    "knowledge, assumptions, or invented facts. Do not combine separate facts into "
    "a new causal, explanatory, or logical relationship unless the context itself "
    "supports that relationship. If the context does not contain enough information "
    "to answer the requested fact, output exactly this sentence and nothing else: "
    f"{INSUFFICIENT_CONTEXT_RESPONSE} "
    "Do not add an explanation, qualification, or speculation to the fallback. Do "
    "not generate citations, references, source sections, filenames, or chunk labels "
    "unless the question explicitly requires one as part of the substantive answer; "
    "the application displays retrieval sources separately. When an answer is "
    "supported, respond directly in one to three concise, factual sentences without "
    "unnecessary introductory language."
)


class SourceResult(TypedDict):
    source: str
    chunk_index: int
    score: float
    content: str


class AnswerResult(TypedDict):
    answer: str
    sources: list[SourceResult]


StoredChunk = tuple[int, str, int, str, list[float]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity without an external numeric library."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")

    dot_product = sum(a_value * b_value for a_value, b_value in zip(a, b))
    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class RAGService:
    """Load the local RAG resources once and reuse them for multiple questions."""

    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = database_path
        self._manager = None
        self._embedding_model = None
        self._chat_model = None
        self._embedding_client = None
        self._chat_client = None
        self._stored_chunks: list[StoredChunk] = []
        self._embedding_dimension: int | None = None
        self._embedding_model_loaded = False
        self._chat_model_loaded = False
        self._started = False
        self._load_count = 0

    @property
    def embedding_model_id(self) -> str | None:
        return self._embedding_model.id if self._embedding_model is not None else None

    @property
    def chat_model_id(self) -> str | None:
        return self._chat_model.id if self._chat_model is not None else None

    @property
    def chunk_count(self) -> int:
        return len(self._stored_chunks)

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @property
    def load_count(self) -> int:
        """Number of successful service starts, useful for lifecycle verification."""
        return self._load_count

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Initialize Foundry Local and load both models once."""
        if self._started:
            return

        try:
            FoundryLocalManager.initialize(
                Configuration(app_name="foundry_local_rag")
            )
            self._manager = FoundryLocalManager.instance
            print("[OK] SDK initialization succeeded.")

            self._embedding_model = self._manager.catalog.get_model(
                EMBEDDING_MODEL_ALIAS
            )
            if self._embedding_model is None:
                raise RuntimeError(
                    f"Embedding model alias {EMBEDDING_MODEL_ALIAS!r} was not found."
                )

            self._chat_model = self._manager.catalog.get_model(CHAT_MODEL_ALIAS)
            if self._chat_model is None:
                raise RuntimeError(
                    f"Required chat model alias {CHAT_MODEL_ALIAS!r} was not found. "
                    "No substitute model will be used."
                )

            print(f"Embedding model ID: {self._embedding_model.id}")
            print(f"Chat model ID: {self._chat_model.id}")

            self._download_if_needed(self._embedding_model, "Embedding")
            self._download_if_needed(self._chat_model, "Chat")

            self._embedding_model.load()
            self._embedding_model_loaded = True
            if not self._embedding_model.is_loaded:
                raise RuntimeError("Embedding model is not reported as loaded.")
            print("[OK] Embedding model loaded.")

            self._chat_model.load()
            self._chat_model_loaded = True
            if not self._chat_model.is_loaded:
                raise RuntimeError("Chat model is not reported as loaded.")
            print("[OK] Chat model loaded.")

            self._stored_chunks, self._embedding_dimension = (
                self._load_stored_chunks()
            )
            print(f"[OK] Chunks loaded from SQLite: {self.chunk_count}")
            print(
                "[OK] Stored embedding dimensionality: "
                f"{self._embedding_dimension}"
            )

            self._embedding_client = self._embedding_model.get_embedding_client()
            self._chat_client = self._chat_model.get_chat_client()
            self._chat_client.settings.temperature = 0.0
            self._chat_client.settings.max_tokens = 96
            self._chat_client.settings.random_seed = 0

            self._started = True
            self._load_count += 1
        except Exception:
            try:
                self.close()
            except Exception as cleanup_error:
                print(f"[ERROR] Cleanup after startup failure: {cleanup_error}")
            raise

    def answer_query(self, question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
        """Retrieve relevant chunks and generate one grounded answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        self.start()
        clean_question = question.strip()
        sources = self.get_top_chunks(clean_question, top_k)
        augmented_context = self._build_augmented_context(sources)

        response = self._chat_client.complete_chat(
            [
                {"role": "system", "content": GROUNDING_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": (
                        f"Retrieved context:\n\n{augmented_context}"
                        f"\n\nQuestion:\n{clean_question}"
                    ),
                },
            ]
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("The chat model returned an empty answer.")
        answer = response.choices[0].message.content.strip()
        if not answer:
            raise RuntimeError("The chat model returned an empty answer.")

        return {"answer": answer, "sources": sources}

    def get_top_chunks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SourceResult]:
        """Embed a query and rank all stored chunks by cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer of at least 1.")

        self.start()
        query_response = self._embedding_client.generate_embedding(query.strip())
        if len(query_response.data) != 1:
            raise RuntimeError("Expected exactly one query embedding.")

        query_embedding = query_response.data[0].embedding
        if (
            not query_embedding
            or self._embedding_dimension is None
            or len(query_embedding) != self._embedding_dimension
        ):
            raise RuntimeError(
                "Query and stored embeddings have different dimensionality."
            )
        if any(not math.isfinite(float(value)) for value in query_embedding):
            raise RuntimeError("The query embedding contains a non-finite value.")

        ranked_results: list[tuple[float, int, str, int, str]] = []
        for chunk_id, source, chunk_index, content, embedding in self._stored_chunks:
            score = cosine_similarity(query_embedding, embedding)
            if not math.isfinite(score):
                raise RuntimeError("A cosine similarity score is not finite.")
            ranked_results.append(
                (score, chunk_id, source, chunk_index, content)
            )

        ranked_results.sort(key=lambda result: result[0], reverse=True)
        top_results = ranked_results[:top_k]
        expected_count = min(top_k, len(self._stored_chunks))

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
            raise RuntimeError(
                "A retrieval result is missing source or content metadata."
            )

        return [
            {
                "source": source,
                "chunk_index": chunk_index,
                "score": score,
                "content": content,
            }
            for score, _, source, chunk_index, content in top_results
        ]

    def close(self) -> None:
        """Unload every model that was loaded, even if another unload fails."""
        cleanup_errors: list[str] = []

        if self._chat_model_loaded and self._chat_model is not None:
            try:
                self._chat_model.unload()
                if self._chat_model.is_loaded:
                    raise RuntimeError("Chat model is still reported as loaded.")
                self._chat_model_loaded = False
                print("[OK] Chat model unloaded.")
            except Exception as error:
                cleanup_errors.append(f"Chat model cleanup failed: {error}")

        if self._embedding_model_loaded and self._embedding_model is not None:
            try:
                self._embedding_model.unload()
                if self._embedding_model.is_loaded:
                    raise RuntimeError("Embedding model is still reported as loaded.")
                self._embedding_model_loaded = False
                print("[OK] Embedding model unloaded.")
            except Exception as error:
                cleanup_errors.append(f"Embedding model cleanup failed: {error}")

        self._started = False
        self._embedding_client = None
        self._chat_client = None
        self._stored_chunks = []
        self._embedding_dimension = None

        if cleanup_errors:
            raise RuntimeError(" ".join(cleanup_errors))

    def _download_if_needed(self, model, label: str) -> None:
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
            raise RuntimeError(
                f"{label} model download finished, but it is not cached."
            )
        print(f"[OK] {label} model downloaded.")

    def _load_stored_chunks(self) -> tuple[list[StoredChunk], int]:
        if not self.database_path.is_file():
            raise RuntimeError(f"Database not found: {self.database_path}")

        connection = sqlite3.connect(self.database_path)
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
                raise RuntimeError(
                    f"Chunk {chunk_id} has invalid embedding JSON."
                ) from error

            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(f"Chunk {chunk_id} has an empty embedding.")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in embedding
            ):
                raise RuntimeError(
                    f"Chunk {chunk_id} has a non-numeric embedding value."
                )

            numeric_embedding = [float(value) for value in embedding]
            if any(not math.isfinite(value) for value in numeric_embedding):
                raise RuntimeError(
                    f"Chunk {chunk_id} has a non-finite embedding value."
                )

            if expected_dimension is None:
                expected_dimension = len(numeric_embedding)
            elif len(numeric_embedding) != expected_dimension:
                raise RuntimeError(
                    "Stored embeddings have inconsistent dimensionality."
                )

            stored_chunks.append(
                (chunk_id, source, chunk_index, content, numeric_embedding)
            )

        if expected_dimension is None:
            raise RuntimeError(
                "Could not determine stored embedding dimensionality."
            )

        return stored_chunks, expected_dimension

    @staticmethod
    def _build_augmented_context(sources: list[SourceResult]) -> str:
        return "\n\n".join(
            (
                f"[Source: {source['source']} | chunk: {source['chunk_index']}]\n"
                f"{source['content']}"
            )
            for source in sources
        )
