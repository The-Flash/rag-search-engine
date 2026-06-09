import argparse
import json

from lib.search_utils import min_max_normalize
from lib.hybrid_search import HybridSearch


def normalize_command(scores: list[float]) -> list[float]:
    result: list[float] = []
    for score in scores:
        result.append(min_max_normalize(score, min(scores), max(scores)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of values"
    )
    normalize_parser.add_argument(
        "nargs",
        type=float,
        nargs="*",
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Weighted score"
    )

    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, help="Alpha value", default=0.5
    )

    weighted_search_parser.add_argument("--limit", type=int, help="Limit", default=5)

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF score")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, help="K value", default=60)
    rrf_search_parser.add_argument("--limit", type=int, help="Limit", default=5)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_command(args.nargs):
                print(f"* {score:.4f}")
        case "weighted-search":
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            # 1. Paddington
            #  Hybrid Score: 1.000
            #  BM25: 1.000, Semantic: 1.000
            #  Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...
            # 2. The Indian in the Cupboard
            #  Hybrid Score: 0.943
            #  BM25: 0.966, Semantic: 0.850
            print("Search results:")
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['document']['title']}")
                print(f" Hybrid Score: {result['hybrid_score']:.4f}")
                print(
                    f" BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
                )
                print(f" {result['document']['description'][:100]}...")
        case "rrf-search":
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['document']['title']}")
                print(f" RRF Score: {result['hybrid_score']:.4f}")
                print(
                    f" BM25: {result['keyword_rank']:.4f}, Semantic: {result['semantic_rank']:.4f}"
                )
                print(f" {result['document']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
