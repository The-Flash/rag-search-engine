#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_query,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="Verify embeddings")
    embed_query_parser.add_argument("query", type=str, help="Embed query")

    search_parser = subparsers.add_parser("search", help="Search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Max results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_query(args.query, args.limit)
        case "chunk":
            chunks: list[str] = []
            text: str = args.text
            words = text.split()
            chunk_size: int = args.chunk_size
            n = 1
            text_chunk = ""
            for word in words:
                text_chunk += f"{word} "
                if n == chunk_size:
                    chunks.append(text_chunk[:-1])
                    text_chunk = ""
                    n = 1
                else:
                    n += 1
            if text_chunk:
                chunks.append(text_chunk.strip())
            print(f"Chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
