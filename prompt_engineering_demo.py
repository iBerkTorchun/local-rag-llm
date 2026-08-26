from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "qwen2.5-0.5b"
CONTEXT = (
    "Employees may work remotely on Tuesdays and Thursdays. "
    "Remote work on other weekdays requires manager approval."
)
ANSWERABLE_QUESTION = (
    "Can employees work remotely on Wednesday without manager approval?"
)
UNANSWERABLE_QUESTION = "How many vacation days do employees receive?"
GROUNDING_INSTRUCTION = (
    "Use only the supplied context; never use outside knowledge or guess. "
    "Reason carefully about negation: if an action requires approval, it cannot be "
    "done without approval, so answer No. If the context does not explicitly state "
    "the requested fact, especially a number, reply exactly: The information is not "
    "available in the supplied context. Output only one short sentence."
)


def show_download_progress(percent: float) -> None:
    print(f"\rDownloading {MODEL_ALIAS}: {percent:5.1f}%", end="", flush=True)


def ask(client, messages: list[dict[str, str]]) -> str:
    response = client.complete_chat(messages)
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("The model returned an empty response.")
    return response.choices[0].message.content.strip()


def grounded_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GROUNDING_INSTRUCTION},
        {
            "role": "user",
            "content": f"Context:\n{CONTEXT}\n\nQuestion:\n{question}",
        },
    ]


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
        client.settings.max_tokens = 64
        client.settings.random_seed = 0

        print(f"\nFictional context:\n{CONTEXT}")

        print("\nExperiment A - Question without context")
        print(f"User message:\n{ANSWERABLE_QUESTION}")
        response_a = ask(
            client,
            [{"role": "user", "content": ANSWERABLE_QUESTION}],
        )
        print(f"Response:\n{response_a}")

        context_prompt = f"Context:\n{CONTEXT}\n\nQuestion:\n{ANSWERABLE_QUESTION}"
        print("\nExperiment B - Question with context")
        print(f"User message (context and question):\n{context_prompt}")
        response_b = ask(client, [{"role": "user", "content": context_prompt}])
        print(f"Response:\n{response_b}")

        print("\nExperiment C - Grounded Q&A")
        print(f"System message:\n{GROUNDING_INSTRUCTION}")

        print(f"\nAnswerable user question:\n{ANSWERABLE_QUESTION}")
        response_c_answerable = ask(
            client,
            grounded_messages(ANSWERABLE_QUESTION),
        )
        print(f"Response:\n{response_c_answerable}")

        print(f"\nUnanswerable user question:\n{UNANSWERABLE_QUESTION}")
        response_c_unanswerable = ask(
            client,
            grounded_messages(UNANSWERABLE_QUESTION),
        )
        print(f"Response:\n{response_c_unanswerable}")
    finally:
        model.unload()
        if model.is_loaded:
            raise RuntimeError("Model unload finished, but the model is still reported as loaded.")
        print("\n[OK] Model unloaded.")


if __name__ == "__main__":
    main()
