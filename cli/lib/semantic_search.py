import json

import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer


class SemanticSearch:
    embeddings_path = "cache/embeddings.npy"

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
