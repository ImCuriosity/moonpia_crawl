# -*- coding: utf-8 -*-
"""마법사 입력 처리 테스트.

대화형 입력은 BOM·공백·따옴표가 섞여 들어오기 쉽고, 그것 때문에 멀쩡한 답이
거부되면 질문 순서가 통째로 어긋난다. 실제로 겪은 버그라 회귀 테스트로 고정한다.

    python -m tests.test_wizard
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munpia import wizard  # noqa: E402


class _FakeInput:
    """input()을 대체해 미리 정한 답을 순서대로 돌려준다."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def _with_input(answers, fn):
    original = wizard.input if hasattr(wizard, "input") else None
    fake = _FakeInput(answers)
    import builtins
    saved = builtins.input
    builtins.input = fake
    try:
        return fn(), fake
    finally:
        builtins.input = saved
        if original is not None:
            wizard.input = original


def test_clean_input_strips_bom():
    """BOM이 붙은 'n'이 그대로 'n'으로 인식되어야 한다 (실제 발생한 버그)."""
    assert wizard._clean_input("﻿n") == "n"
    assert wizard._clean_input("  y  ") == "y"
    assert wizard._clean_input('"587273"') == "587273"
    assert wizard._clean_input("​n​") == "n"
    assert wizard._clean_input("") == ""


def test_ask_yes_accepts_bom_prefixed_answer():
    (result, _) = _with_input(["﻿n"], lambda: wizard.ask_yes("계속?", default=True))
    assert result is False


def test_ask_yes_variants():
    for answer, expected in [("y", True), ("Y", True), ("yes", True), ("예", True),
                             ("ㅇ", True), ("1", True),
                             ("n", False), ("N", False), ("no", False),
                             ("아니오", False), ("0", False)]:
        (result, _) = _with_input([answer], lambda: wizard.ask_yes("q", default=True))
        assert result is expected, "%s -> %s" % (answer, result)


def test_ask_yes_empty_uses_default():
    (r1, _) = _with_input([""], lambda: wizard.ask_yes("q", default=True))
    (r2, _) = _with_input([""], lambda: wizard.ask_yes("q", default=False))
    assert r1 is True and r2 is False


def test_ask_yes_gives_up_after_repeated_garbage():
    """인식 불가 입력이 계속돼도 무한 루프에 빠지지 않아야 한다."""
    answers = ["zzz"] * 10
    (result, fake) = _with_input(answers, lambda: wizard.ask_yes("q", default=True))
    assert result is True
    # 5회까지만 되묻고 포기한다 (입력을 전부 소진하지 않는다)
    assert len(fake.prompts) <= 5


def test_ask_yes_eof_uses_default():
    (result, _) = _with_input([], lambda: wizard.ask_yes("q", default=False))
    assert result is False


def test_ask_int():
    (r, _) = _with_input(["25"], lambda: wizard.ask_int("n", 20))
    assert r == 25
    (r, _) = _with_input([""], lambda: wizard.ask_int("n", 20))
    assert r == 20
    (r, _) = _with_input([""], lambda: wizard.ask_int("n", None))
    assert r is None
    # 쉼표와 '개' 접미사 허용
    (r, _) = _with_input(["1,500"], lambda: wizard.ask_int("n", 1))
    assert r == 1500
    (r, _) = _with_input(["30개"], lambda: wizard.ask_int("n", 1))
    assert r == 30
    # 숫자가 아니면 되묻고, 계속 실패하면 기본값
    (r, fake) = _with_input(["abc"] * 10, lambda: wizard.ask_int("n", 7))
    assert r == 7 and len(fake.prompts) <= 5


def test_ask_int_rejects_below_minimum():
    (r, _) = _with_input(["0", "5"], lambda: wizard.ask_int("n", None, minimum=1))
    assert r == 5


def test_ask_uses_default_on_empty():
    (r, _) = _with_input([""], lambda: wizard.ask("경로", "novel_ids.txt"))
    assert r == "novel_ids.txt"


def test_mask_hides_id():
    assert wizard._mask("munpiauser").startswith("mu")
    assert "user" not in wizard._mask("munpiauser")


def test_explain_scope():
    assert wizard.explain_scope(False) == "free"
    assert wizard.explain_scope(True) == "all"


def run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    buf = io.StringIO()
    for name, fn in tests:
        saved_stdout = sys.stdout
        try:
            sys.stdout = buf  # 마법사 출력이 테스트 결과를 가리지 않게 한다
            fn()
            sys.stdout = saved_stdout
            print("  PASS  %s" % name)
        except AssertionError as exc:
            sys.stdout = saved_stdout
            failed += 1
            import traceback
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            print("  FAIL  %s (line %d): %s" % (name, tb.lineno, tb.line))
            if str(exc):
                print("        %s" % exc)
        except Exception as exc:
            sys.stdout = saved_stdout
            failed += 1
            print("  ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
        finally:
            sys.stdout = saved_stdout
    print("\n%d개 중 %d개 통과" % (len(tests), len(tests) - failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
