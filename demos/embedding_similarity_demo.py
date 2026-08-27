import math

from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "qwen3-embedding-0.6b"
SENTENCES = [
    "Python functions package reusable logic into named blocks of code.",
    "Git records revisions to source code so software teams can collaborate safely.",
    "Relational databases organize structured data into tables linked by keys.",
    "A border collie needs regular exercise and enjoys learning new commands.",
    "Airplane wings create lift as air flows around them during flight.",
    "Bakers use yeast and warm dough to make bread rise before baking.",
]
QUERY = "How can developers preserve and review the history of changes to a program?"


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

        client = model.get_embedding_client()

        batch_response = client.generate_embeddings(SENTENCES)
        batch_items = sorted(batch_response.data, key=lambda item: item.index)
        embeddings = [item.embedding for item in batch_items]

        if len(embeddings) != len(SENTENCES):
            raise RuntimeError("The embedding count does not match the sentence count.")
        if not embeddings or not embeddings[0]:
            raise RuntimeError("The model returned an empty embedding vector.")

        dimensionality = len(embeddings[0])
        if any(len(embedding) != dimensionality for embedding in embeddings):
            raise RuntimeError("The sentence embeddings have inconsistent dimensionality.")

        print(f"Generated embeddings: {len(embeddings)}")
        print(f"Embedding dimensionality: {dimensionality}")
        preview = ", ".join(f"{value:.6f}" for value in embeddings[0][:5])
        print(f"First vector preview: [{preview}, ...]")

        query_response = client.generate_embedding(QUERY)
        if len(query_response.data) != 1:
            raise RuntimeError("Expected exactly one query embedding.")
        query_embedding = query_response.data[0].embedding
        if len(query_embedding) != dimensionality:
            raise RuntimeError("The query embedding has a different dimensionality.")

        results = [
            (cosine_similarity(query_embedding, embedding), sentence)
            for sentence, embedding in zip(SENTENCES, embeddings)
        ]
        if any(not math.isfinite(score) for score, _ in results):
            raise RuntimeError("At least one cosine similarity score is not finite.")
        results.sort(key=lambda result: result[0], reverse=True)

        print(f"\nQuery:\n{QUERY}")
        print("\nSimilarity results:")
        for score, sentence in results:
            print(f"{score:.4f} | {sentence}")

        highest_score, highest_sentence = results[0]
        print(f"\nHighest-scoring sentence ({highest_score:.4f}):")
        print(highest_sentence)
    finally:
        model.unload()
        if model.is_loaded:
            raise RuntimeError("Model unload finished, but the model is still reported as loaded.")
        print("[OK] Embedding model unloaded.")


if __name__ == "__main__":
    main()
