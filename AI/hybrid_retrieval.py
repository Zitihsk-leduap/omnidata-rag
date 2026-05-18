from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self, chroma_db):
        self.db = chroma_db
        self.bm25 = None
        self.chunks = []
        self.build_index()

    def build_index(self):
        all_docs = self.db.get()
        doc_ids = all_docs.get("ids", [])
        doc_contents = all_docs.get("documents", [])
        doc_metadatas = all_docs.get("metadatas", [])

        self.chunks = [
            Document(
                page_content=content,
                metadata=metadata or {}
            )
            for content, metadata in zip(doc_contents, doc_metadatas)
        ]

        if self.chunks:
            tokenized_docs = [doc.page_content.lower().split() for doc in self.chunks]
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"BM25 index built with {len(self.chunks)} chunks")
        else:
            print("Warning: No chunks found in database")

    def _normalize_scores(self, scores: List[float], min_val=None, max_val=None) -> List[float]:
        if min_val is None or max_val is None:
            if not scores or all(s == scores[0] for s in scores):
                return [0.5] * len(scores) if scores else []
            min_val = min(scores)
            max_val = max(scores)

        if min_val == max_val:
            return [0.5] * len(scores)

        return [(s - min_val) / (max_val - min_val) for s in scores]

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[float, Document]]:
        if not self.bm25 or not self.chunks:
            print("BM25 index not initialized, falling back to vector search")
            docs = self.db.similarity_search(query, k=k)
            return [(0.5, doc) for doc in docs]

        vector_docs = self.db.similarity_search_with_score(query, k=k)
        vector_scores = {doc.metadata.get('id'): 1 - score for doc, score in vector_docs}

        bm25_scores = self.bm25.get_scores(query.lower().split())
        sorted_bm25 = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        bm25_normalized = self._normalize_scores([score for _, score in sorted_bm25])
        bm25_by_idx = {idx: norm_score for (idx, _), norm_score in zip(sorted_bm25, bm25_normalized)}

        vector_scores_list = list(vector_scores.values())
        vector_normalized = self._normalize_scores(vector_scores_list) if vector_scores_list else []
        vector_by_id = {doc_id: norm_score for doc_id, norm_score in zip(vector_scores.keys(), vector_normalized)}

        hybrid_scores = {}
        all_indices = set(bm25_by_idx.keys())
        for i in range(len(self.chunks)):
            vector_id = self.chunks[i].metadata.get('id')
            bm25_score = bm25_by_idx.get(i, 0.0)
            vector_score = vector_by_id.get(vector_id, 0.0)
            hybrid_scores[i] = (bm25_score + vector_score) / 2

        top_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        result = []
        for idx, score in top_indices:
            result.append((score, self.chunks[idx]))

        return result
