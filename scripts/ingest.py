#!/usr/bin/env python3
"""Index a directory of text files.

    python scripts/ingest.py data/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.store import ingest_paths  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/ingest.py <dir-or-files>")
        raise SystemExit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.md")) + sorted(p.rglob("*.txt")))
        elif p.is_file():
            paths.append(p)

    if not paths:
        print("no .md or .txt files found")
        raise SystemExit(1)

    n = ingest_paths(paths)
    print(f"indexed {n} chunks from {len(paths)} files")


if __name__ == "__main__":
    main()
