import re


PARAGRAPHS_PER_CHUNK = 2


def chunk_text(document_text: str) -> list[str]:
    """Split text into ordered chunks of approximately two paragraphs each."""
    normalized_text = document_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized_text)
        if paragraph.strip()
    ]

    return [
        "\n\n".join(paragraphs[start : start + PARAGRAPHS_PER_CHUNK])
        for start in range(0, len(paragraphs), PARAGRAPHS_PER_CHUNK)
    ]
