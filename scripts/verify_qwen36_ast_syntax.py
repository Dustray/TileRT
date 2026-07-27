"""批量验证 Qwen3.6 Python 文件语法。

验证项：
1. tilert/models/qwen3_6 下所有 .py 文件可通过 AST 解析。
2. 无 import 级语法错误。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_ast_syntax.py
"""
import ast
import logging
import os
import sys

from tilert import logger

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)


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

    logger.info("Passed: %d, Failed: %d", len(passed), len(failed))
    for path in passed[:5]:
        logger.info("  OK  %s", path)
    if len(passed) > 5:
        logger.info("  ... and %d more", len(passed) - 5)
    for path, exc in failed:
        logger.error("  FAIL %s: %s", path, exc)

    if failed:
        logger.error("\n=== AST syntax check FAILED ===")
        return 1
    logger.info("\n=== AST syntax check PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
