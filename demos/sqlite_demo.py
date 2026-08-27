import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "sqlite_demo.db"

SAMPLE_DOCUMENTS = [
    (
        "Learning Python",
        "Python is a programming language known for clear, readable syntax.",
        "software",
    ),
    (
        "SQLite Basics",
        "SQLite stores relational data in a single local database file.",
        "database",
    ),
    (
        "Version Control with Git",
        "Git records source-code changes and helps software teams collaborate.",
        "software",
    ),
    (
        "How Airplanes Fly",
        "Airplane wings generate lift as the aircraft moves through the air.",
        "aviation",
    ),
    (
        "Machine Learning Overview",
        "Machine learning models discover useful patterns in training data.",
        "machine learning",
    ),
]


def print_rows(heading: str, rows: list[tuple]) -> None:
    print(f"\n{heading}:")
    for document_id, title, content, category in rows:
        print(f"{document_id} | {title} | {category}")
        print(f"    {content}")


def main() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT
            )
            """
        )

        # Reset this fixed demo dataset so repeated runs do not create duplicates.
        connection.execute("DELETE FROM documents")
        connection.execute(
            "DELETE FROM sqlite_sequence WHERE name = ?",
            ("documents",),
        )
        connection.executemany(
            "INSERT INTO documents (title, content, category) VALUES (?, ?, ?)",
            SAMPLE_DOCUMENTS,
        )
        connection.commit()

        row_count = connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        if row_count != len(SAMPLE_DOCUMENTS):
            raise RuntimeError(
                f"Expected {len(SAMPLE_DOCUMENTS)} rows, but found {row_count}."
            )

        all_rows = connection.execute(
            "SELECT id, title, content, category FROM documents ORDER BY id"
        ).fetchall()

        selected_id = all_rows[0][0]
        row_by_id = connection.execute(
            "SELECT id, title, content, category FROM documents WHERE id = ?",
            (selected_id,),
        ).fetchone()

        category = "software"
        category_rows = connection.execute(
            """
            SELECT id, title, content, category
            FROM documents
            WHERE category = ?
            ORDER BY id
            """,
            (category,),
        ).fetchall()

        search_term = "data"
        search_rows = connection.execute(
            """
            SELECT id, title, content, category
            FROM documents
            WHERE content LIKE ?
            ORDER BY id
            """,
            (f"%{search_term}%",),
        ).fetchall()

        print(f"Database path: {DATABASE_PATH}")
        print(f"Stored rows: {row_count}")
        print_rows("All documents", all_rows)
        print_rows(f"Document with id {selected_id}", [row_by_id])
        print_rows(f"Documents in category {category!r}", category_rows)
        print_rows(f"Content containing {search_term!r}", search_rows)
    finally:
        connection.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
