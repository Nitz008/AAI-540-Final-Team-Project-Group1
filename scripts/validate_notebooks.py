"""Check committed notebooks without executing AWS calls or requiring a kernel."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def validate(path: Path) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: expected notebook format 4")
    try:
        import nbformat
    except ImportError:
        pass  # Syntax/structure checks remain available with standard-library Python.
    else:
        nbformat.validate(nbformat.from_dict(notebook))
    count = 0
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython line magics are valid notebook syntax, outside Python's AST.
        source = "\n".join(
            "pass # IPython magic" if line.lstrip().startswith(("%", "!")) else line
            for line in source.splitlines()
        )
        ast.parse(source, filename=f"{path.name}:cell-{index}")
        count += 1
    print(f"{path.name}: valid structure and {count} compilable code cells")
    return count


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No notebooks found")
    for path in notebooks:
        validate(path)


if __name__ == "__main__":
    main()
