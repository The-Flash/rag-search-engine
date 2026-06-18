import os
import time
from typing import Literal

from google import genai

from .search_utils import min_max_normalize

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

type SPELL = Literal["spell"]
type REWRITE = Literal["rewrite"]
type EXPAND = Literal["expand"]

type ENHANCE = SPELL | REWRITE | EXPAND

type INDIVIDUAL = Literal["individual"]
type RERANK = INDIVIDUAL


def spell_enhance_query(query: str) -> str:
    return f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """


def rewrite_enhance_query(query: str) -> str:
    return f"""Rewrite the user-provided movie search query below to be more specific and searchable.
    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """


def expand_enhance_query(query: str) -> str:
    return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""


def individual_rerank_query(query: str, doc_title: str, doc_description: str) -> str:
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc_title} - {doc_description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.
"""


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate the Reciprocal Rank Fusion (RRF) score for a given rank and k value.
    """
    return 1 / (k + rank)


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
        keyword_search_results = self._bm25_search(query, 500 * limit)
        semantic_search_results = self._semantic_search(query, 500 * limit)
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

    def rrf_search(
        self,
        query: str,
        k: int,
        limit: int = 10,
        enhance: ENHANCE | None = None,
        rerank: RERANK | None = None,
    ) -> list[dict]:
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        is_spell_enhance = enhance == "spell"
        is_rewrite_enhance = enhance == "rewrite"
        is_expand_enhance = enhance == "expand"
        is_individual_rerank = rerank == "individual"
        if is_individual_rerank:
            limit = limit * 5
        client = genai.Client(api_key=api_key)
        if is_spell_enhance:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=spell_enhance_query(query),
            )
            if response.text is not None:
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query

        if is_rewrite_enhance:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=rewrite_enhance_query(query),
            )
            if response.text is not None:
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
        if is_expand_enhance:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=expand_enhance_query(query),
            )
            if response.text is not None:
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query

        weighted_map = (
            dict()
        )  # doc_id -> {"keyword_rank": float, "semantic_rank": float}
        semantic_search_results = self._semantic_search(query, 500 * limit)
        keyword_search_results = self._bm25_search(query, 500 * limit)

        print("### keyword search ranking")
        for i, r in enumerate(keyword_search_results):
            (doc_id, _) = r
            weighted_map[doc_id] = {
                "keyword_rank": rrf_score(i, k),
                "semantic_rank": 0.0,
                "document": self.idx.docmap[doc_id],
                "llm_score": 0,
            }

        print("### semantic search ranking")
        for i, r in enumerate(semantic_search_results):
            doc_id = r["id"]
            if doc_id not in weighted_map:
                weighted_map[doc_id] = {
                    "keyword_rank": 0.0,
                    "document": self.semantic_search.document_map[doc_id],
                    "semantic_rank": rrf_score(
                        i,
                        k,
                    ),
                    "llm_score": 0,
                }
            else:
                weighted_map[doc_id]["semantic_rank"] = rrf_score(i, k)

        print("### hybrid search ranking")
        for doc_id in weighted_map:
            keyword_rank = weighted_map[doc_id]["keyword_rank"]
            semantic_rank = weighted_map[doc_id]["semantic_rank"]
            weighted_map[doc_id]["hybrid_score"] = keyword_rank + semantic_rank

        print("### llm ranking")
        for doc_id in weighted_map:
            time.sleep(3)
            document = weighted_map[doc_id]["document"]
            print(f"Ranking doc {document['id']}")
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=individual_rerank_query(
                    query, document["title"], document["description"]
                ),
            )
            if response.text is not None:
                weighted_map[doc_id]["llm_score"] = response.text

            print(f"Done ranking {document['id']}")

        return sorted(
            weighted_map.values(), key=lambda x: x["llm_score"], reverse=True
        )[:limit]
