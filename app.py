import atexit

from flask import Flask, jsonify, request

from rag_service import DEFAULT_TOP_K, RAGService


app = Flask(__name__)
rag_service = RAGService()
atexit.register(rag_service.close)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    if "question" not in payload:
        return jsonify({"error": "The 'question' field is required."}), 400

    question = payload["question"]
    if not isinstance(question, str):
        return jsonify({"error": "The 'question' field must be a string."}), 400

    question = question.strip()
    if not question:
        return jsonify({"error": "The 'question' field must not be empty."}), 400

    try:
        result = rag_service.answer_query(question, top_k=DEFAULT_TOP_K)
        return jsonify(result)
    except Exception:
        app.logger.exception("RAG request failed")
        return jsonify({"error": "An internal server error occurred."}), 500


def main() -> None:
    try:
        rag_service.start()
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=False,
        )
    finally:
        rag_service.close()


if __name__ == "__main__":
    main()
