import argparse
import math
import json
import os
import pickle
import string

from collections import Counter

from nltk.stem import PorterStemmer
from search_utils import BM25_B, BM25_K1


def tokenize(text: str) -> list[str]:
    stemmer = PorterStemmer()
    stop_words = get_stop_words()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if len(t) > 0]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


class InvertedIndex:
    index: dict[str, set[int]] = dict()
    docmap = dict()
    term_frequencies: dict[int, Counter] = dict()
    doc_lengths = dict()
    index_path = "cache/index.pkl"
    docmap_path = "cache/docmap.pkl"
    tf_path = "cache/term_frequencies.pkl"
    doc_lengths_path = "cache/doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        count = 0
        for token in tokens:
            doc_ids = self.index.get(token, set())
            doc_ids.add(doc_id)
            self.index[token] = doc_ids
            self.term_frequencies[doc_id] = self.term_frequencies.get(doc_id, Counter())
            self.term_frequencies[doc_id][token] += 1
            count += 1
        self.doc_lengths[doc_id] = count

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        avg_length = total_length / len(self.docmap) if self.docmap else 0
        return avg_length

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) == 0:
            raise ValueError("Term must contain at least one valid token.")
        token = tokens[0]
        self.term_frequencies[doc_id] = self.term_frequencies.get(doc_id, Counter())
        return self.term_frequencies[doc_id][token]

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize(term)
        if len(tokens) == 0:
            raise ValueError("Term must contain at least one valid token.")
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        doc_ids = sorted(doc_ids)
        return doc_ids

    def get_bm25_idf(self, term: str) -> float:
        # log((N - df + 0.5) / (df + 0.5) + 1)
        N = len(self.docmap)
        df = len(self.get_documents(term))
        bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        # length_norm = 1 - b + b * (doc_length / avg_doc_length)       length_norm
        length_norm = (
            1 - b + b * (self.doc_lengths.get(doc_id, 0) / self.__get_avg_doc_length())
        )
        raw_tf = self.get_tf(doc_id, term)
        # BM25 saturation formala - (tf * (k1 + 1)) / (tf + k1 * length_norm) where tf is term frequency, k1 is a tunable parameter and length_norm is the document length normalization factor
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term) -> float:
        # BM25 = bm25_tf * bm25_idf
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int):
        scores = dict()
        tokens = tokenize(query)
        for token in tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                score = self.bm25(doc_id, token)
                scores[doc_id] = scores.get(doc_id, 0.0) + round(score, 2)
        # Sort the documents by score in descending order.
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:limit]

    def build(self, data):
        for d in data:
            doc_id = d["id"]
            self.docmap[doc_id] = d
            text = f"{d['title']} {d['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open(self.index_path, "wb") as index_f:
            pickle.dump(self.index, index_f)
        with open(self.docmap_path, "wb") as docmap_f:
            pickle.dump(self.docmap, docmap_f)
        with open(self.tf_path, "wb") as tf_f:
            pickle.dump(self.term_frequencies, tf_f)
        with open(self.doc_lengths_path, "wb") as doc_lengths_f:
            pickle.dump(self.doc_lengths, doc_lengths_f)

    def load(self):
        index_files = [
            self.index_path,
            self.docmap_path,
            self.tf_path,
            self.doc_lengths_path,
        ]
        if any(not os.path.exists(path) for path in index_files):
            raise FileNotFoundError(
                "Index files not found. Please run the 'build' command to create the index."
            )
        with open(self.index_path, "rb") as index_f:
            self.index = pickle.load(index_f)
        with open(self.docmap_path, "rb") as docmap_f:
            self.docmap = pickle.load(docmap_f)
        with open(self.tf_path, "rb") as tf_f:
            self.term_frequencies = pickle.load(tf_f)
        with open(self.doc_lengths_path, "rb") as doc_lengths_f:
            self.doc_lengths = pickle.load(doc_lengths_f)


def get_stop_words() -> list[str]:
    with open("data/stopwords.txt", "r") as f:
        stop_words = f.read().splitlines()
    return stop_words


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize(term)
    if len(token) == 0:
        raise Exception("Term must contain at least one valid token.")
    token = token[0]
    idf = inverted_index.get_bm25_idf(token)
    return idf


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize(term)
    if len(token) == 0:
        raise Exception("Term must contain at least one valid token.")
    token = token[0]
    tf = inverted_index.get_bm25_tf(doc_id, token, k1, b)
    return tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a document and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a document and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    data_path = "data/movies.json"
    match args.command:
        case "search":
            inverted_index = InvertedIndex()
            inverted_index.load()
            with open(data_path, "r") as f:
                data = json.load(f)
            movies = data["movies"]
            print("Searching for:", args.query)
            num_of_results = 0
            for query_token in tokenize(args.query):
                doc_ids = inverted_index.get_documents(query_token)
                for doc_id in doc_ids:
                    if num_of_results == 5:
                        break
                    movie = inverted_index.docmap[doc_id]
                    print(f"{doc_id}. {movie['title']}")
                    num_of_results += 1
        case "build":
            with open(data_path, "r") as f:
                data = json.load(f)
            movies = data["movies"]
            index = InvertedIndex()
            index.build(movies)
            index.save()
        case "tf":
            doc_id = args.doc_id
            term = args.term
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf = inverted_index.get_tf(doc_id, term)
            print(f"Term frequency of '{term}' in document {doc_id}: {tf}")
        case "idf":
            term = args.term
            inverted_index = InvertedIndex()
            inverted_index.load()
            ## IDF formula - log((N + 1) / (1 + df)) where N is total number of documents and df is document frequency of the term
            N = len(inverted_index.docmap)
            df = len(inverted_index.get_documents(term))
            idf = math.log((N + 1) / (df + 1))
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf = inverted_index.get_tf(doc_id, term)
            idf = math.log(
                (len(inverted_index.docmap) + 1)
                / (1 + len(inverted_index.get_documents(term)))
            )
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25":
            idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            inverted_index = InvertedIndex()
            inverted_index.load()
            print("Searching for:", args.query)
            results = inverted_index.bm25_search(args.query, limit=5)
            for index, (doc_id, score) in enumerate(results, start=1):
                movie = inverted_index.docmap[doc_id]
                print(f"{index}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
