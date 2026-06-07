import os

from .search_utils import min_max_normalize

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """
    Calculate the hybrid score as a weighted average of BM25 and semantic scores.
    The alpha parameter controls the weight of each component:
    - alpha = 1.0: Only BM25 score is considered.
    - alpha = 0.0: Only semantic score is considered.
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _semantic_search(self, query: str, limit: int):
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        weighted_map = (
            dict()
        )  # doc_id -> {"keyword_score": float, "semantic_score": float}
        keyword_search_results = self._bm25_search(query, 500 * 5)
        semantic_search_results = self._semantic_search(query, 500 * 5)
        max_keyword_search_score = max(
            map(lambda result: result[1], keyword_search_results)
        )
        min_keyword_search_score = min(
            map(lambda result: result[1], keyword_search_results)
        )
        max_semantic_search_score = max(
            map(lambda result: result["score"], semantic_search_results)
        )
        min_semantic_search_score = min(
            map(lambda result: result["score"], semantic_search_results)
        )
        for doc_id, keyword_score in keyword_search_results:
            weighted_map[doc_id] = {
                "keyword_score": min_max_normalize(
                    keyword_score, min_keyword_search_score, max_keyword_search_score
                ),
                "semantic_score": 0.0,
                "document": self.idx.docmap[doc_id],
            }
        for result in semantic_search_results:
            doc_id = result["id"]
            if doc_id not in weighted_map:
                weighted_map[doc_id] = {
                    "keyword_score": 0.0,
                    "document": self.semantic_search.document_map[doc_id],
                    "semantic_score": min_max_normalize(
                        result["score"],
                        min_semantic_search_score,
                        max_semantic_search_score,
                    ),
                }
            else:
                weighted_map[doc_id]["semantic_score"] = min_max_normalize(
                    result["score"],
                    min_semantic_search_score,
                    max_semantic_search_score,
                )
        for doc_id in weighted_map:
            keyword_score = weighted_map[doc_id]["keyword_score"]
            semantic_score = weighted_map[doc_id]["semantic_score"]
            weighted_map[doc_id]["hybrid_score"] = hybrid_score(
                keyword_score, semantic_score, alpha
            )
        return sorted(
            weighted_map.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
