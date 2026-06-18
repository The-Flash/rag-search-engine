import argparse
import json

from dotenv import load_dotenv
from lib.search_utils import min_max_normalize
from lib.hybrid_search import HybridSearch

load_dotenv()


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
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
    )

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
            print("Search results:")
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['document']['title']}")
                print(f" Hybrid Score: {result['hybrid_score']:.4f}")
                print(
                    f" BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
                )
                print(f" {result['document']['description'][:100]}...")
        case "rrf-search":
            print("rerank - ", args.rerank_method)
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['document']['title']}")
                print(f" RRF Score: {result['hybrid_score']:.4f}")
                print(f"Re-rank Score: {result['llm_score']:.4f}/10")
                print(
                    f"""
                    BM25: {result["keyword_rank"]:.4f}, Semantic: {result["semantic_rank"]:.4f}
                    """
                )
                print(f" {result['document']['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
