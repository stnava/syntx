#!/usr/bin/env python3
"""
Prove that changes to Python files are documentation-only: for every modified .py file, the
syntax tree with all docstrings (and comments) removed must equal the committed version's.

    python scripts/check_docs_only.py [REV]      # default REV = HEAD; exit 1 on a code change
"""
import ast
import subprocess
import sys


def _strip(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    files = subprocess.check_output(["git", "diff", "--name-only", rev, "--", "*.py"], text=True).split()
    bad = []
    for f in files:
        old = subprocess.run(["git", "show", f"{rev}:{f}"], capture_output=True, text=True)
        if old.returncode != 0:
            print(f"new file (not checked): {f}")
            continue
        if _strip(ast.parse(old.stdout)) != _strip(ast.parse(open(f).read())):
            bad.append(f)
    for f in files:
        print(("CODE CHANGED: " if f in bad else "docs only:    ") + f)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
