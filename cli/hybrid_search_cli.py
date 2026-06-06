import argparse

from lib.search_utils import min_max_normalize


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_command(args.nargs):
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
