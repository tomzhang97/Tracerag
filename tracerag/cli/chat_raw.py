"""
Raw RAG query CLI - Direct access to TraceRAG pipeline.

This is a simplified wrapper around demo_search for command-line RAG queries.

Usage:
    python -m tracerag.cli.chat_raw --index_root /data/tracerag/index --query "Your question here"

Or use the alias:
    tracerag-search --index_root /data/tracerag/index --query "Your question"
"""

# This is just a re-export of demo_search functionality
from tracerag.cli.demo_search import main, app

if __name__ == "__main__":
    app()
