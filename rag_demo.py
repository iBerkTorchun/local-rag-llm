import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from retrieval_demo import (
    DEFAULT_TOP_K,
    RetrievalResult,
    StoredChunk,
    get_top_chunks,
    load_stored_chunks,
)


EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"
INSUFFICIENT_CONTEXT_RESPONSE = (
    "The information is not available in the supplied context."
)
GROUNDING_SYSTEM_INSTRUCTION = (
    "You are a grounded question-answering assistant. Answer using only facts "
    "explicitly present in the supplied retrieved context. Do not use outside "
    "knowledge, make unsupported assumptions, or invent missing facts. Every claim "
    "in the answer must be directly supported by the context; do not combine facts "
    "to infer a new mechanism. Prefer a concise, direct restatement of the relevant "
    "context. If the retrieved context does not contain enough information to answer "
    "the question, reply exactly with this sentence and nothing else: "
    f"{INSUFFICIENT_CONTEXT_RESPONSE}"
)

ANSWERABLE_QUESTIONS = [
    "Why is SQLite suitable for this local project?",
    "How does RAG help reduce hallucinations?",
    "What role does Foundry Local play in the application?",
    "What should the assistant do when retrieved information is insufficient?",
]
UNANSWERABLE_QUESTION = "How many vacation days do users of this application receive?"


def download_if_needed(model, label: str) -> None:
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
        raise RuntimeError(f"{label} model download finished, but it is not cached.")
    print(f"[OK] {label} model downloaded.")


def build_augmented_context(results: list[RetrievalResult]) -> str:
    context_blocks = []
    for _, _, source, chunk_index, content in results:
        context_blocks.append(
            f"[Source: {source} | chunk: {chunk_index}]\n{content}"
        )
    return "\n\n".join(context_blocks)


def answer_query(
    question: str,
    embedding_client,
    chat_client,
    stored_chunks: list[StoredChunk],
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, list[RetrievalResult]]:
    retrieved_results = get_top_chunks(
        question,
        embedding_client,
        stored_chunks,
        top_k,
    )
    augmented_context = build_augmented_context(retrieved_results)

    response = chat_client.complete_chat(
        [
            {"role": "system", "content": GROUNDING_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n\n{augmented_context}"
                    f"\n\nQuestion:\n{question}"
                ),
            },
        ]
    )

    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("The chat model returned an empty answer.")
    answer = response.choices[0].message.content.strip()
    if not answer:
        raise RuntimeError("The chat model returned an empty answer.")

    return answer, retrieved_results


def print_rag_result(
    question: str,
    answer: str,
    retrieved_results: list[RetrievalResult],
) -> None:
    print(f"\nQuestion:\n{question}")
    print("\nRetrieved context:")

    for rank, (score, _, source, chunk_index, content) in enumerate(
        retrieved_results,
        start=1,
    ):
        print(f"\n{rank}. {score:.4f} | {source} | chunk {chunk_index}")
        print("   " + content.replace("\n", "\n   "))

    print(f"\nGenerated answer:\n{answer}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    FoundryLocalManager.initialize(Configuration(app_name="foundry_local_rag"))
    manager = FoundryLocalManager.instance
    print("[OK] SDK initialization succeeded.")

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if embedding_model is None:
        raise RuntimeError(
            f"Embedding model alias {EMBEDDING_MODEL_ALIAS!r} was not found."
        )

    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    if chat_model is None:
        raise RuntimeError(
            f"Required chat model alias {CHAT_MODEL_ALIAS!r} was not found. "
            "No substitute model will be used."
        )

    print(f"Embedding model ID: {embedding_model.id}")
    print(f"Chat model ID: {chat_model.id}")

    download_if_needed(embedding_model, "Embedding")
    download_if_needed(chat_model, "Chat")

    embedding_model_loaded = False
    chat_model_loaded = False
    cleanup_errors: list[str] = []

    try:
        embedding_model.load()
        embedding_model_loaded = True
        if not embedding_model.is_loaded:
            raise RuntimeError("Embedding model is not reported as loaded.")
        print("[OK] Embedding model loaded.")

        chat_model.load()
        chat_model_loaded = True
        if not chat_model.is_loaded:
            raise RuntimeError("Chat model is not reported as loaded.")
        print("[OK] Chat model loaded.")

        stored_chunks, stored_dimension = load_stored_chunks()
        print(f"[OK] Chunks loaded from SQLite: {len(stored_chunks)}")
        print(f"[OK] Stored embedding dimensionality: {stored_dimension}")

        embedding_client = embedding_model.get_embedding_client()
        chat_client = chat_model.get_chat_client()
        chat_client.settings.temperature = 0.0
        chat_client.settings.max_tokens = 96
        chat_client.settings.random_seed = 0

        for question in ANSWERABLE_QUESTIONS:
            answer, retrieved_results = answer_query(
                question,
                embedding_client,
                chat_client,
                stored_chunks,
            )
            print_rag_result(question, answer, retrieved_results)

        answer, retrieved_results = answer_query(
            UNANSWERABLE_QUESTION,
            embedding_client,
            chat_client,
            stored_chunks,
        )
        print_rag_result(UNANSWERABLE_QUESTION, answer, retrieved_results)
        if answer != INSUFFICIENT_CONTEXT_RESPONSE:
            raise RuntimeError(
                "The unanswerable question did not produce the required fallback."
            )

        print("\n[OK] Retrieval and non-empty-answer validations passed.")
        print("[OK] The unanswerable question produced the required fallback.")
    finally:
        if chat_model_loaded:
            try:
                chat_model.unload()
                if chat_model.is_loaded:
                    raise RuntimeError("Chat model is still reported as loaded.")
                print("[OK] Chat model unloaded.")
            except Exception as error:
                cleanup_errors.append(f"Chat model cleanup failed: {error}")

        if embedding_model_loaded:
            try:
                embedding_model.unload()
                if embedding_model.is_loaded:
                    raise RuntimeError("Embedding model is still reported as loaded.")
                print("[OK] Embedding model unloaded.")
            except Exception as error:
                cleanup_errors.append(f"Embedding model cleanup failed: {error}")

        if cleanup_errors:
            raise RuntimeError(" ".join(cleanup_errors))


if __name__ == "__main__":
    main()
