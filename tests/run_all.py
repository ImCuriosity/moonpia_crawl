# -*- coding: utf-8 -*-
"""전체 테스트 실행.

    python -m tests.run_all
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["test_preprocess", "test_storage", "test_auth", "test_wizard"]


def main() -> int:
    import importlib

    total_failed = 0
    for name in MODULES:
        print("\n" + "=" * 60)
        print("  %s" % name)
        print("=" * 60)
        mod = importlib.import_module("tests." + name)
        total_failed += mod.run()

    print("\n" + "=" * 60)
    if total_failed:
        print("  실패한 모듈 %d개" % total_failed)
    else:
        print("  전체 통과")
    print("=" * 60)
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
