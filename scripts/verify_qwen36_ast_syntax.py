"""批量验证 Qwen3.6 Python 文件语法。

验证项：
1. tilert/models/qwen3_6 下所有 .py 文件可通过 AST 解析。
2. 无 import 级语法错误。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_ast_syntax.py
"""
import ast
import os
import sys


def main():
    root = "/public/home/dinggy/yiny/projects/TileRT/tilert/models/qwen3_6"
    failed = []
    passed = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ast.parse(f.read())
                passed.append(path)
            except SyntaxError as exc:
                failed.append((path, exc))

    print(f"Passed: {len(passed)}, Failed: {len(failed)}")
    for path in passed[:5]:
        print(f"  OK  {path}")
    if len(passed) > 5:
        print(f"  ... and {len(passed) - 5} more")
    for path, exc in failed:
        print(f"  FAIL {path}: {exc}")

    if failed:
        print("\n=== AST syntax check FAILED ===", file=sys.stderr)
        return 1
    print("\n=== AST syntax check PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
