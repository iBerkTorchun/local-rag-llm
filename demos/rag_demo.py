import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_service import INSUFFICIENT_CONTEXT_RESPONSE, AnswerResult, RAGService


ANSWERABLE_QUESTIONS = [
    "Why is SQLite suitable for this local project?",
    "How does RAG help reduce hallucinations?",
    "What role does Foundry Local play in the application?",
    "What should the assistant do when retrieved information is insufficient?",
]
UNANSWERABLE_QUESTION = "How many vacation days do users of this application receive?"


def print_rag_result(question: str, result: AnswerResult) -> None:
    print(f"\nQuestion:\n{question}")
    print("\nRetrieved context:")

    for rank, source in enumerate(result["sources"], start=1):
        print(
            f"\n{rank}. {source['score']:.4f} | {source['source']} | "
            f"chunk {source['chunk_index']}"
        )
        print("   " + source["content"].replace("\n", "\n   "))

    print(f"\nGenerated answer:\n{result['answer']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    service = RAGService()
    try:
        service.start()
        print(f"[OK] Selected embedding model ID: {service.embedding_model_id}")
        print(f"[OK] Selected chat model ID: {service.chat_model_id}")

        for question in ANSWERABLE_QUESTIONS:
            print_rag_result(question, service.answer_query(question))

        result = service.answer_query(UNANSWERABLE_QUESTION)
        print_rag_result(UNANSWERABLE_QUESTION, result)
        if result["answer"] != INSUFFICIENT_CONTEXT_RESPONSE:
            raise RuntimeError(
                "The unanswerable question did not produce the required fallback."
            )

        print("\n[OK] Retrieval and non-empty-answer validations passed.")
        print("[OK] The unanswerable question produced the required fallback.")
    finally:
        service.close()


if __name__ == "__main__":
    main()
