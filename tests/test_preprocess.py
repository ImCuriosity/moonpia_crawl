# -*- coding: utf-8 -*-
"""전처리 규칙 회귀 테스트.

    python -m tests.test_preprocess
    (또는 pytest tests/)
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munpia.preprocess import (  # noqa: E402
    char_length, clean_text, epoch, iso, parse_datetime, stable_user_key,
    to_bool_int, to_int,
)
from munpia.schema import (  # noqa: E402
    fill_reply_counts, parse_comment, parse_episode, parse_novel,
)


def test_clean_text_basic():
    assert clean_text("  안녕\t하세요  ") == "안녕 하세요"
    assert clean_text("줄1\r\n\r\n\r\n줄2") == "줄1\n\n줄2"
    assert clean_text(None) == ""
    assert clean_text("<b>굵게</b>") == "굵게"
    assert clean_text("한<br>줄") == "한\n줄"
    assert clean_text("&lt;태그&gt;") == "<태그>"
    # 개행을 접는 모드
    assert clean_text("a\nb", keep_newlines=False) == "a b"


def test_clean_text_keeps_emotion_signal():
    """감정 신호(자모 반복, 이모지)는 살리고 과도한 반복만 축약한다."""
    assert clean_text("ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅋㅋㅋㅋ"
    assert clean_text("ㅠㅠ") == "ㅠㅠ"          # 짧은 건 그대로
    assert clean_text("대박!!!!!!!") == "대박!!!!"
    assert "😭" in clean_text("슬프다 😭")


def test_clean_text_control_chars():
    assert clean_text("a\x00\x07b") == "a b"
    assert clean_text("보​이​지​않음") == "보이지않음"


def test_to_int():
    assert to_int("1,234") == 1234
    assert to_int("조회 12,345회") == 12345
    assert to_int(None) == 0
    assert to_int("") == 0
    assert to_int("없음") == 0
    assert to_int(42) == 42
    assert to_int(42.9) == 42
    assert to_int("1.2만") == 12000
    assert to_int("3천") == 3000
    assert to_int("-5") == -5
    assert to_int(None, default=-1) == -1


def test_to_bool_int():
    assert to_bool_int(True) == 1
    assert to_bool_int(False) == 0
    assert to_bool_int("true") == 1
    assert to_bool_int(None) == 0


def test_datetime():
    dt = parse_datetime("2026-07-23T09:12:16")
    assert dt is not None and dt.year == 2026 and dt.minute == 12
    assert iso("2026-07-23T09:12:16") == "2026-07-23 09:12:16"
    assert iso("깨진값") == ""
    assert epoch("깨진값") == 0
    assert epoch("2026-07-23T09:12:16") > 0


def test_stable_user_key():
    assert stable_user_key("ptsforu", "따스한봄날") == "u_ptsforu"
    # blogUrl이 없으면 닉네임 해시로 대체하되 접두사로 구분된다
    key = stable_user_key("", "익명독자")
    assert key.startswith("n_") and len(key) == 18
    assert stable_user_key("", "") == ""
    # 같은 닉네임은 항상 같은 키
    assert stable_user_key(None, "홍길동") == stable_user_key("", "홍길동")


def test_char_length():
    assert char_length("안녕 하세요") == 5
    assert char_length("") == 0


def test_parse_novel():
    detail = {"novelInfo": {
        "id": 1, "title": " 테스트작품 ", "authorName": "작가",
        "genres": ["판타지", "무협"], "genreBestName": "판타지",
        "introduction": "소개\t글\n\n\n입니다", "groupName": "작가연재",
        "viewCount": 132, "likeCount": 9, "preferenceCount": 24,
        "chapterCount": 4, "freeChapterCount": 4, "characters": 21609,
        "free": True, "adult": False, "finish": False, "pause": False,
        "createdAt": "2026-07-23T09:10:39", "updatedAt": "2026-07-24T18:30:00",
    }}
    stats = {"maleCount": 101, "femaleCount": 16, "age30sPercent": 31.5}
    rec = parse_novel(1, detail, stats, "2026-07-24 18:48:05")

    assert rec.title == "테스트작품"
    assert rec.genres == "판타지|무협"
    assert rec.introduction == "소개 글\n\n입니다"
    assert rec.preference_count == 24
    assert rec.serial_status == "연재중" and rec.status_code == 0
    assert rec.is_free == 1 and rec.is_adult == 0
    assert rec.reader_male_count == 101
    assert rec.reader_age30s_pct == 31.5
    assert rec.created_ts > 0

    # 완결/휴재 분기
    detail["novelInfo"]["finish"] = True
    assert parse_novel(1, detail).status_code == 1
    detail["novelInfo"]["finish"] = False
    detail["novelInfo"]["pause"] = True
    assert parse_novel(1, detail).status_code == 2


def test_parse_episode():
    item = {"id": 8587264, "num": 1, "title": " 각성 그리고 뽑기 ",
            "commentCount": 1, "createdAt": "2026-07-23T09:12:16",
            "viewCount": 54, "likeCount": 3, "free": True, "pages": 11}
    rec = parse_episode(587273, item)
    assert rec.episode_uid == "587273_8587264"
    assert rec.title == "각성 그리고 뽑기"
    assert rec.view_count == 54 and rec.comment_count == 1
    assert rec.char_estimate == 11 * 900
    assert rec.is_free == 1


def test_parse_comment_text_and_sticker():
    text_item = {"id": 1, "parentId": 0, "nickName": "고추냉이",
                 "blogUrl": "n1551_delafos", "createdAt": "2026-07-23T13:25:02",
                 "contentType": "TEXT", "content": "오셨다ㅋㅋㅋㅋㅋㅋ",
                 "likeCount": 2, "dislikeCount": 1, "replyLevel": 0, "secret": False}
    rec = parse_comment(587273, 8587264, 1, text_item)
    assert rec.comment_uid == "587273_8587264_1"
    assert rec.episode_uid == "587273_8587264"
    assert rec.user_key == "u_n1551_delafos"
    assert rec.body == "오셨다ㅋㅋㅋㅋ"
    assert rec.sticker_url == ""
    assert rec.like_count == 2 and rec.dislike_count == 1
    assert rec.is_reply == 0

    sticker_item = dict(text_item, id=2, contentType="STICKER",
                        content="https://cdn1.munpia.com/x.png", replyLevel=1)
    rec2 = parse_comment(587273, 8587264, 1, sticker_item)
    # 스티커는 본문이 텍스트가 아니므로 body에 URL이 새어들어가면 안 된다
    assert rec2.body == ""
    assert rec2.sticker_url.startswith("https://")
    assert rec2.is_reply == 1 and rec2.reply_level == 1


def test_fill_reply_counts():
    base = {"nickName": "a", "blogUrl": "a", "createdAt": "2026-07-23T00:00:00",
            "contentType": "TEXT", "content": "x", "replyLevel": 0}
    parent = parse_comment(1, 2, 1, dict(base, id=100, parentId=0))
    child1 = parse_comment(1, 2, 1, dict(base, id=101, parentId=100, replyLevel=1))
    child2 = parse_comment(1, 2, 1, dict(base, id=102, parentId=100, replyLevel=1))
    rows = fill_reply_counts([parent, child1, child2])
    assert rows[0].reply_count == 2
    assert rows[1].reply_count == 0


def test_missing_fields_do_not_raise():
    """필드가 통째로 비어도 예외 없이 기본값으로 채워져야 한다."""
    assert parse_novel(9, {}).novel_id == 9
    assert parse_episode(9, {}).novel_id == 9
    assert parse_comment(9, 8, 1, {}).novel_id == 9


def run() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
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
