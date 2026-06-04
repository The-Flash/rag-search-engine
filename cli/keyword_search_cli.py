import argparse
import json
import math

from lib.keyword_search import InvertedIndex, tokenize
from lib.search_utils import BM25_B, BM25_K1


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
