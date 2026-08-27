# Architecture

All project components run on one machine.

A client interface accepts the user's question.

A server or pipeline layer orchestrates retrieval and generation.

SQLite stores the local document chunks and embeddings, while Foundry Local provides local model inference. During a query, relevant chunks are retrieved and supplied to the local LLM as augmented context.

