from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


PathLike = Union[str, Path]


def ensure_parent_dir(path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: PathLike, indent: int = 2, ensure_ascii: bool = False) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(path: PathLike) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# ADDED METHOD: LOAD JSON LINES (.jsonl)
# =========================================================
def load_jsonl(path: PathLike) -> List[Any]:
    """Reads a .jsonl file line by line into a list of objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# =========================================================
# ADDED METHOD: SAVE JSON LINES (.jsonl)
# =========================================================
def save_jsonl(data: List[Any], path: PathLike, ensure_ascii: bool = False) -> None:
    """Saves a list of objects line by line into a .jsonl file."""
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


def save_text(text: str, path: PathLike) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_text(path: PathLike) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def list_files(root: PathLike, suffixes: List[str] | None = None, recursive: bool = True) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []

    if recursive:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = [p for p in root.glob("*") if p.is_file()]

    if suffixes:
        suffixes_lower = {s.lower() for s in suffixes}
        files = [p for p in files if p.suffix.lower() in suffixes_lower]

    return sorted(files)


def read_raw_text_documents(root: PathLike, suffixes: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Reads all raw text documents from a folder and returns a list of dicts:
    [
      {
        "doc_id": "doc_000001",
        "source_file": "example.txt",
        "relative_path": "subdir/example.txt",
        "text": "..."
      },
      ...
    ]
    """
    root = Path(root)
    files = list_files(root, suffixes=suffixes, recursive=True)

    docs: List[Dict[str, Any]] = []
    for idx, fp in enumerate(files, start=1):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        docs.append({
            "doc_id": f"doc_{idx:06d}",
            "source_file": fp.name,
            "relative_path": str(fp.relative_to(root)),
            "text": text
        })

    return docs