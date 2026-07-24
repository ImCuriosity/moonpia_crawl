# -*- coding: utf-8 -*-
"""저장 계층 테스트 — 네트워크 없이 CSV/JSONL 출력과 재개 로직을 검증한다.

    python -m tests.test_storage
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munpia.crawler import NovelResult  # noqa: E402
from munpia.schema import CommentRecord, EpisodeRecord, NovelRecord  # noqa: E402
from munpia.storage import DatasetWriter  # noqa: E402


def _sample(novel_id: int = 100) -> NovelResult:
    novel = NovelRecord(novel_id=novel_id, title="테스트작품",
                        introduction="여러 줄\n소개, 쉼표 포함")
    ep = EpisodeRecord(episode_uid="%d_1" % novel_id, novel_id=novel_id,
                       entry_id=1, episode_num=1, title="1화", view_count=10)
    cm = CommentRecord(comment_uid="%d_1_5" % novel_id, episode_uid="%d_1" % novel_id,
                       novel_id=novel_id, entry_id=1, episode_num=1, comment_id=5,
                       user_key="u_tester", body="재밌어요\n다음화 기대", like_count=3)
    return NovelResult(novel, [ep], [cm], [])


def test_csv_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        with DatasetWriter(tmp, fmt="csv") as w:
            w.write_result(_sample())
            assert w.summary()["novels"] == 1

        path = os.path.join(tmp, "novels.csv")
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["title"] == "테스트작품"
        # 개행과 쉼표가 든 필드가 따옴표 처리되어 컬럼이 밀리지 않아야 한다
        assert rows[0]["introduction"] == "여러 줄\n소개, 쉼표 포함"
        assert rows[0]["novel_id"] == "100"

        with open(os.path.join(tmp, "comments.csv"), "r",
                  encoding="utf-8-sig", newline="") as f:
            crows = list(csv.DictReader(f))
        assert crows[0]["body"] == "재밌어요\n다음화 기대"
        assert crows[0]["episode_uid"] == "100_1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_jsonl_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        with DatasetWriter(tmp, fmt="jsonl") as w:
            w.write_result(_sample(200))

        path = os.path.join(tmp, "novels.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        assert len(lines) == 1
        assert lines[0]["novel_id"] == 200
        assert lines[0]["title"] == "테스트작품"
        # ensure_ascii=False 로 한글이 이스케이프되지 않아야 한다
        assert "테스트작품" in open(path, encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_append_does_not_duplicate_header():
    tmp = tempfile.mkdtemp()
    try:
        with DatasetWriter(tmp, fmt="csv") as w:
            w.write_result(_sample(1))
        with DatasetWriter(tmp, fmt="csv") as w:
            w.write_result(_sample(2))

        with open(os.path.join(tmp, "novels.csv"), "r",
                  encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        # 헤더 1줄 + 데이터 2줄
        assert len(rows) == 3, rows
        assert rows[0][0] == "novel_id"
        assert rows[1][0] == "1" and rows[2][0] == "2"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_skips_completed():
    tmp = tempfile.mkdtemp()
    try:
        with DatasetWriter(tmp, fmt="csv") as w:
            w.write_result(_sample(42))
            assert w.is_done(42)

        # 새 세션에서도 완료 목록이 살아 있어야 한다
        with DatasetWriter(tmp, fmt="csv", resume=True) as w2:
            assert w2.is_done(42)
            assert not w2.is_done(43)

        # resume=False면 무시한다
        with DatasetWriter(tmp, fmt="csv", resume=False) as w3:
            assert not w3.is_done(42)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_errors_are_logged_without_novel():
    tmp = tempfile.mkdtemp()
    try:
        with DatasetWriter(tmp, fmt="csv") as w:
            # 작품 메타 수집 자체가 실패한 경우 — 오류만 남고 행은 안 쓰인다
            w.write_result(NovelResult(None, [], [], ["비공개 작품"]))
            assert w.summary()["novels"] == 0
        text = open(os.path.join(tmp, "_errors.log"), encoding="utf-8").read()
        assert "비공개 작품" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_format_rejected():
    try:
        DatasetWriter(tempfile.mkdtemp(), fmt="parquet")
    except ValueError:
        return
    raise AssertionError("잘못된 fmt를 거부해야 합니다")


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
