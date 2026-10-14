"""Text extraction bounds and source-preserving overlap without provider calls."""

from types import SimpleNamespace

import pytest

from app.knowledge.ingestion import IngestionService
from app.settings import Settings


def service(**options):
    config = Settings(
        _env_file=None,
        embedding_model="test",
        embedding_version="v1",
        embedding_dimension=3,
        document_chunk_chars=40,
        document_chunk_overlap=8,
        **options,
    )
    retrieval = SimpleNamespace(
        embedding_provider=SimpleNamespace(model="test", version="v1", dimension=3)
    )
    return IngestionService(None, None, retrieval, config)


def test_utf8_bom_and_line_endings(tmp_path):
    source = tmp_path / "policy.md"
    source.write_bytes(b"\xef\xbb\xbf# Policy\r\n14 days\rSupport")
    assert service().extract(source) == "# Policy\n14 days\nSupport"


@pytest.mark.parametrize("data", [b" ", b"\xff", b"bad\x00text"])
def test_invalid_text_rejected(data):
    with pytest.raises(ValueError):
        service().decode(data)


def test_format_and_size_limits(tmp_path):
    source = tmp_path / "policy.pdf"
    source.write_bytes(b"text")
    with pytest.raises(ValueError):
        service().extract(source)
    source = tmp_path / "policy.txt"
    source.write_bytes(b"x" * 11)
    with pytest.raises(ValueError):
        service(document_max_bytes=10).extract(source)


def test_chunk_overlap_preserves_all_content_and_locators():
    text = "0123456789" * 15
    chunks = service().chunk(text)
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert chunks[0].text + "".join(chunk.text[8:] for chunk in chunks[1:]) == text
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.locator == "lines:1-1" for chunk in chunks)
