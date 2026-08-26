from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "qwen2.5-0.5b"
PROMPT = "Explain retrieval-augmented generation in one sentence."


def show_download_progress(percent: float) -> None:
    print(f"\rDownloading {MODEL_ALIAS}: {percent:5.1f}%", end="", flush=True)


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

        client = model.get_chat_client()
        client.settings.temperature = 0.0
        client.settings.max_tokens = 80
        print(f"Prompt: {PROMPT}")
        print("Response: ", end="", flush=True)

        response_parts: list[str] = []
        for chunk in client.complete_streaming_chat(
            [{"role": "user", "content": PROMPT}]
        ):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                response_parts.append(content)
                print(content, end="", flush=True)
        print()

        if not "".join(response_parts).strip():
            raise RuntimeError("Inference completed without a non-empty response.")
        print("[OK] Inference produced a non-empty response.")
    finally:
        model.unload()
        if model.is_loaded:
            raise RuntimeError("Model unload finished, but the model is still reported as loaded.")
        print("[OK] Model unloaded.")


if __name__ == "__main__":
    main()
