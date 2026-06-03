#!/usr/bin/env python3

import argparse
import json
import re
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_query,
    ChunkedSemanticSearch,
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
    chunk_parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Text to chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Chunk overlap"
    )
    subparsers.add_parser(
        "embed_chunks", help="Embed chunks of text from the movies dataset"
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
            overlap: int = args.overlap
            words = text.split()
            chunk_size: int = args.chunk_size
            n = 1
            text_chunk = ""
            for i, word in enumerate(words):
                text_chunk += f"{word} "
                if n == chunk_size:
                    chunks.append(text_chunk[:-1])  # remove trailing space
                    text_chunk = ""
                    n = 1
                    if overlap > 0:  # start overlap
                        last_chunk = chunks[-1] if len(chunks) > 0 else None
                        if last_chunk:
                            overlap_words = last_chunk.split()[-overlap:]
                            text_chunk = " ".join(overlap_words) + " " + text_chunk
                else:
                    n += 1
            if text_chunk:
                chunks.append(text_chunk.strip())
            print(f"Chunking {len(text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            chunks: list[str] = []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            overlap: int = args.overlap
            n = 1
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
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                documents = data["movies"]
            chunked_semantic_search = ChunkedSemanticSearch()
            chunk_embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(
                documents
            )
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
