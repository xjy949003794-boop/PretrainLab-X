"""Compatibility entry point for the Day 4 preprocessing CLI."""

from .preprocessing import build_corpus, clean_text, main

__all__ = ["build_corpus", "clean_text", "main"]


if __name__ == "__main__":
    main()
