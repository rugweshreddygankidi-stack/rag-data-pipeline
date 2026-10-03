import pytest

from rag_pipeline.chunking import chunk_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_empty_and_whitespace_text():
    assert chunk_text("") == [] and chunk_text("   \n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text(words(10), max_words=50, overlap=5) == [words(10)]


def test_exact_max_words_is_one_chunk():
    assert len(chunk_text(words(50), max_words=50, overlap=5)) == 1


def test_chunks_never_exceed_max_words():
    for chunk in chunk_text(words(1000), max_words=100, overlap=20):
        assert len(chunk.split()) <= 100


def test_consecutive_chunks_overlap():
    chunks = chunk_text(words(300), max_words=100, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_every_word_is_covered():
    text = words(537)
    covered = set()
    for chunk in chunk_text(text, max_words=100, overlap=10):
        covered.update(chunk.split())
    assert covered == set(text.split())


def test_last_chunk_reaches_end_of_text():
    assert chunk_text(words(250), max_words=100, overlap=20)[-1].split()[-1] == "w249"


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words, overlap)


def test_zero_overlap_partitions_text():
    chunks = chunk_text(words(30), max_words=10, overlap=0)
    assert len(chunks) == 3 and " ".join(chunks) == words(30)


def test_whitespace_is_normalised():
    assert chunk_text("a   b\n\nc\td", max_words=10, overlap=0) == ["a b c d"]
