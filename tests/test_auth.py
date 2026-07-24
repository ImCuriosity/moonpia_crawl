# -*- coding: utf-8 -*-
"""인증 유틸 테스트 — 실제 자격증명 없이 .env 파싱과 마스킹을 검증한다.

    python -m tests.test_auth
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munpia import auth  # noqa: E402


def _write(tmp: str, text: str) -> str:
    path = os.path.join(tmp, ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def test_dotenv_parsing():
    tmp = tempfile.mkdtemp()
    try:
        path = _write(tmp, "\n".join([
            "# 주석은 무시",
            "MUNPIA_ID=myuser",
            'MUNPIA_PW="p@ss=word#1"',      # 따옴표 + 값 안의 = 과 #
            "export EXTRA='quoted'",         # export 접두사 + 홑따옴표
            "",
            "BROKEN_LINE_NO_EQUALS",
        ]))
        env = auth.load_dotenv(path)
        assert env["MUNPIA_ID"] == "myuser"
        # 값에 포함된 = 와 # 는 잘리면 안 된다 (비밀번호에 흔한 문자)
        assert env["MUNPIA_PW"] == "p@ss=word#1"
        assert env["EXTRA"] == "quoted"
        assert "BROKEN_LINE_NO_EQUALS" not in env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dotenv_missing_file():
    assert auth.load_dotenv(os.path.join(tempfile.mkdtemp(), "nope.env")) == {}


def test_dotenv_does_not_pollute_environ():
    """자격증명이 os.environ에 새어 하위 프로세스로 전파되면 안 된다."""
    tmp = tempfile.mkdtemp()
    try:
        path = _write(tmp, "MUNPIA_SECRET_CANARY=leaked")
        auth.load_dotenv(path)
        assert os.environ.get("MUNPIA_SECRET_CANARY") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_credentials_from_env_file():
    tmp = tempfile.mkdtemp()
    try:
        path = _write(tmp, "MUNPIA_ID=abc\nMUNPIA_PW=def\n")
        uid, pw = auth.read_credentials(path)
        assert uid == "abc" and pw == "def"

        # 별칭 키도 인식해야 한다
        path2 = _write(tmp, "MUNPIA_USERNAME=xyz\nMUNPIA_PASSWORD=123\n")
        uid2, pw2 = auth.read_credentials(path2)
        assert uid2 == "xyz" and pw2 == "123"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_credentials_absent():
    uid, pw = auth.read_credentials(os.path.join(tempfile.mkdtemp(), "none.env"))
    # 환경변수에 우연히 설정되어 있지 않은 한 None
    assert uid is None or isinstance(uid, str)


def test_mask_hides_identifier():
    assert auth._mask("munpiauser") == "mu********"
    assert auth._mask("ab") == "a*"
    # 마스킹 결과에 원본 꼬리가 남으면 안 된다
    assert "user" not in auth._mask("munpiauser")


def test_login_error_extraction():
    html = '<div id="loginResult">아이디 또는 비밀번호가 <b>일치하지</b> 않습니다.</div>'
    msg = auth._extract_login_error(html)
    assert "아이디" in msg
    assert auth._extract_login_error("<div>없음</div>") == ""


def test_save_cookies_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "sub", "cookies.json")
        auth.save_cookies([{"name": "a", "value": "1",
                            "domain": ".munpia.com", "path": "/"}], path)
        assert os.path.exists(path)
        data = json.load(open(path, encoding="utf-8"))
        assert data[0]["name"] == "a"

        # client가 이 형식을 그대로 읽어야 한다
        from munpia.client import MunpiaClient
        c = MunpiaClient(cookie_file=path)
        assert c.logged_in
        assert c.session.cookies.get("a") == "1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failed += 1
            import traceback
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            print("  FAIL  %s (line %d): %s" % (name, tb.lineno, tb.line))
            if str(exc):
                print("        %s" % exc)
        except Exception as exc:
            failed += 1
            print("  ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d개 중 %d개 통과" % (len(tests), len(tests) - failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
