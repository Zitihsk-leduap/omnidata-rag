from langchain_core.documents import Document

from rag.retrieval import HybridRetriever


class FakeDB:
    """Stands in for the Chroma client: just enough surface for HybridRetriever."""

    def __init__(self, chunks):
        self._chunks = chunks

    def get(self):
        return {
            "ids": [c.metadata["id"] for c in self._chunks],
            "documents": [c.page_content for c in self._chunks],
            "metadatas": [c.metadata for c in self._chunks],
        }

    def similarity_search_with_score(self, query, k=10):
        # Pretend every chunk is an equally good vector match.
        return [(c, 0.1) for c in self._chunks[:k]]

    def similarity_search(self, query, k=10):
        return self._chunks[:k]


def make_chunks():
    parent = Document(
        page_content="Parent section full text about founders.",
        metadata={"id": "parent_1", "type": "parent", "parent_id": "parent_1", "title": "Founders"},
    )
    child = Document(
        page_content="founders minimum seven",
        metadata={"id": "child_1", "type": "child", "parent_id": "parent_1"},
    )
    other = Document(
        page_content="unrelated clause about dividends",
        metadata={"id": "child_2", "type": "child", "parent_id": "parent_1"},
    )
    return [parent, child, other]


def test_build_index_creates_id_lookup_and_bm25():
    retriever = HybridRetriever(FakeDB(make_chunks()))
    assert retriever.id_to_idx["child_1"] == 1
    assert retriever.bm25 is not None


def test_retrieve_enriches_child_with_parent_context_without_mutating_cache():
    retriever = HybridRetriever(FakeDB(make_chunks()))

    results_1 = retriever.retrieve("founders minimum", k=2)
    results_2 = retriever.retrieve("founders minimum", k=2)

    # The cached Document must stay untouched across repeated queries —
    # it used to grow a new "[Parent Context]" block every time it was
    # retrieved again, since the old code mutated it in place.
    cached_child = retriever.chunks[retriever.id_to_idx["child_1"]]
    assert cached_child.page_content == "founders minimum seven"

    enriched_1 = next(doc for _, doc in results_1 if doc.metadata.get("id") == "child_1")
    enriched_2 = next(doc for _, doc in results_2 if doc.metadata.get("id") == "child_1")

    assert enriched_1.page_content.count("[Parent Context]") == 1
    assert enriched_2.page_content.count("[Parent Context]") == 1


def test_retrieve_falls_back_to_vector_search_when_corpus_empty():
    retriever = HybridRetriever(FakeDB([]))
    results = retriever.retrieve("anything", k=5)
    assert results == []


def test_normalize_scores_handles_equal_values():
    retriever = HybridRetriever(FakeDB(make_chunks()))
    assert retriever._normalize_scores([3, 3, 3]) == [0.5, 0.5, 0.5]


def test_normalize_scores_handles_empty_list():
    retriever = HybridRetriever(FakeDB(make_chunks()))
    assert retriever._normalize_scores([]) == []


def test_normalize_scores_scales_between_zero_and_one():
    retriever = HybridRetriever(FakeDB(make_chunks()))
    assert retriever._normalize_scores([1, 2, 3]) == [0.0, 0.5, 1.0]
