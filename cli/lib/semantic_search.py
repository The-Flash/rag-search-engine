import json
import re

import numpy as np
from pathlib import Path
from collections.abc import Iterator

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    embeddings_path = "cache/embeddings.npy"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = list()
        self.document_map = dict()

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_strs = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_strs.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(doc_strs, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if Path(self.embeddings_path).exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        similarity_list = []
        for i, doc_embedding in enumerate(self.embeddings):
            document = self.documents[i]
            if document is None:
                continue
            similarity = cosine_similarity(embedding, doc_embedding)
            similarity_list.append((similarity, document))
        similarity_list = sorted(similarity_list, key=lambda x: x[0], reverse=True)
        return map(
            lambda x: dict(
                score=x[0],
                title=x[1]["title"],
                description=x[1]["description"],
            ),
            similarity_list[:limit],
        )


class ChunkedSemanticSearch(SemanticSearch):
    embeddings_path = "cache/chunk_embeddings.npy"
    chunk_metadata_path = "cache/chunk_metadata.json"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict] | None = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        doc_strs = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_strs.append(f"{document['title']}: {document['description']}")
        chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for document in documents:
            description = document["description"]
            if description.strip() == "":
                continue
            description_chunks = semantic_chunk(description)
            chunks.extend(description_chunks)
            for i, _ in enumerate(description_chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": document["id"],
                        "chunk_idx": i,
                        "total_chunks": len(description_chunks),
                    }
                )
        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        np.save(self.embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if (
            Path(self.embeddings_path).exists()
            and Path(self.chunk_metadata_path).exists()
        ):
            self.chunk_embeddings = np.load(self.embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> Iterator[dict]:
        query = query.strip()
        embedding = self.generate_embedding(query)
        if self.chunk_embeddings is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        if self.chunk_metadata is None:
            raise ValueError(
                "No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first."
            )
        chunk_scores: list[dict] = list()
        for chunk_embedding, chunks in zip(
            self.chunk_embeddings, self.chunk_metadata["chunks"]
        ):
            cosine_sim = cosine_similarity(embedding, chunk_embedding)
            chunk_scores.append(
                dict(
                    score=cosine_sim,
                    chunk_idx=chunks["chunk_idx"],
                    movie_idx=chunks["movie_idx"],
                )
            )
        movie_idx_to_score: dict[int, float] = dict()
        for chunk_score in chunk_scores:
            movie_score = chunk_score["score"]
            movie_idx = chunk_score["movie_idx"]
            current_score = movie_idx_to_score.get(movie_idx, float("-inf"))
            if movie_idx not in movie_idx_to_score or movie_score > current_score:
                movie_idx_to_score[movie_idx] = movie_score
        sorted_movies = sorted(
            movie_idx_to_score.items(), key=lambda x: x[1], reverse=True
        )
        result = sorted_movies[:limit]
        return map(
            lambda x: dict(
                id=x[0],
                title=self.document_map[x[0]]["title"],
                document=self.document_map[x[0]]["description"][:100],
                score=round(x[1], 4),
                metadata=self.chunk_metadata["chunks"][x[0]]
                if self.chunk_metadata["chunks"]
                else {},
            ),
            result,
        )


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list[str]:
    chunks: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    print(f"Semantically chunking {len(text)} characters")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    data_path = "data/movies.json"
    with open(data_path, "r") as f:
        data = json.load(f)
        documents = data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_query(query: str, limit=5):
    semantic_search = SemanticSearch()
    data_path = "data/movies.json"
    with open(data_path, "r") as f:
        data = json.load(f)
        documents = data["movies"]
    semantic_search.load_or_create_embeddings(documents)
    search_result = semantic_search.search(query, limit)
    for i, r in enumerate(search_result, start=1):
        print(f"{i}. {r['title']} (score: {r['score']})\n{r['description']}")
