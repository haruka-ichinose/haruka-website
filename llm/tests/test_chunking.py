from mecha_llm.chunking import chunk_document
from mecha_llm.documents import SourceDoc


def _count(text: str) -> int:
    return len(text.split())


def test_hash_comments_in_code_fences_are_not_headings():
    doc = SourceDoc(
        doc_id="repo-x",
        title="x",
        text="## Setup\n\nRun it:\n\n```bash\n# install deps\nuv sync\n```\n\n## Usage\n\nCall it.",
        source_type="repo",
        url="/",
    )
    chunks = chunk_document(doc, max_tokens=400, overlap_tokens=0, count_tokens=_count)
    assert [c.text.split("\n")[0] for c in chunks] == ["Setup", "Usage"]
    assert "# install deps" in chunks[0].text
