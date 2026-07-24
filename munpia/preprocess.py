# -*- coding: utf-8 -*-
"""텍스트 정제 및 숫자/일시 정규화 유틸 (요구사항 3.4).

수집 단계가 아니라 여기서만 값을 가공한다. 파서는 원본 필드를 꺼내오고
정규화는 전부 이 모듈을 거치게 해서, 학습 데이터의 표현이 한 곳에서 결정되도록 한다.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Any, Optional

# 제어문자(개행/탭 제외)와 제로폭 문자 — 크롤링 텍스트에 섞여 토크나이저를 망가뜨린다
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[​-‏  ﻿­]")
# 3회 이상 반복되는 동일 문자 (ㅋㅋㅋㅋㅋ, ㅠㅠㅠㅠ, !!!!!) → 감정 신호는 살리되 길이는 억제
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_WS_RE = re.compile(r"[ \t 　]+")
# 줄 앞뒤의 가로 공백만 제거한다. \s를 쓰면 연속 개행까지 삼켜 문단 구분이 사라진다.
_NEWLINE_RE = re.compile(r"[^\S\n]*\n[^\S\n]*")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NUM_RE = re.compile(r"-?\d+")

# 문피아 댓글에 흔한 노이즈
_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_BR_RE = re.compile(r"(?i)<br\s*/?>")


def clean_text(
    value: Any,
    *,
    keep_newlines: bool = True,
    collapse_repeats: bool = True,
    strip_urls: bool = False,
) -> str:
    """탭·개행·제어문자·중복 공백을 정리한 문자열을 돌려준다.

    감정 분석(KOTE 등)에 넣을 댓글은 이모지와 자모(ㅋㅋ, ㅠㅠ)가 그 자체로 라벨 신호이므로
    지우지 않는다. 대신 과도한 반복만 4자로 잘라 길이 폭주를 막는다.

    Args:
        keep_newlines: False면 모든 개행을 공백 하나로 접는다 (CSV 한 셀에 넣을 때 유용).
        collapse_repeats: 4회 이상 반복 문자를 4회로 축약.
        strip_urls: 본문 내 URL 제거.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)

    # NFC 정규화: 한글 자모 분리(NFD) 상태로 오는 닉네임을 합성형으로 통일
    text = unicodedata.normalize("NFC", text)

    text = _HTML_BR_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')

    if strip_urls:
        text = _URL_RE.sub(" ", text)

    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if collapse_repeats:
        text = _REPEAT_RE.sub(lambda m: m.group(1) * 4, text)

    if keep_newlines:
        text = _NEWLINE_RE.sub("\n", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        text = "\n".join(_WS_RE.sub(" ", line).strip() for line in text.split("\n"))
    else:
        text = _WS_RE.sub(" ", text.replace("\n", " "))

    return text.strip()


def to_int(value: Any, default: int = 0) -> int:
    """'1,234', '조회 1,234회', 1234.0, None 등을 정수로 변환한다.

    파싱 실패 시 예외 대신 default를 돌려준다 — 한 필드 때문에 회차 전체를
    버리는 것보다 결측을 0으로 두고 수집 성공률을 지키는 쪽이 낫다.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return default if value != value else int(value)  # NaN 방어

    text = str(value).strip()
    if not text:
        return default
    text = text.replace(",", "").replace(" ", "")

    # '1.2만', '3천' 같은 축약 표기 대응
    m = re.match(r"^(-?\d+(?:\.\d+)?)(만|천|억)$", text)
    if m:
        scale = {"천": 1_000, "만": 10_000, "억": 100_000_000}[m.group(2)]
        return int(float(m.group(1)) * scale)

    m = _NUM_RE.search(text)
    return int(m.group()) if m else default


def to_bool_int(value: Any) -> int:
    """불리언을 0/1 정수로. 모델 입력은 전부 수치여야 하므로 True/False를 쓰지 않는다."""
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("true", "1", "y", "yes") else 0
    return 1 if bool(value) else 0


def parse_datetime(value: Any) -> Optional[datetime]:
    """문피아 API의 ISO8601(초 단위, 타임존 없음 = KST)을 datetime으로 변환."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def iso(value: Any) -> str:
    """저장용 ISO 문자열. 파싱 실패면 빈 문자열."""
    dt = parse_datetime(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def epoch(value: Any) -> int:
    """시계열 모델용 유닉스 타임스탬프(초). 파싱 실패면 0."""
    dt = parse_datetime(value)
    return int(dt.timestamp()) if dt else 0


def stable_user_key(blog_url: Any, nickname: Any) -> str:
    """댓글 작성자의 안정적 식별자.

    문피아는 blogUrl(회원 블로그 슬러그)이 회원당 고정이라 유지율 추적의 기본 키로 쓴다.
    비어 있을 때만 닉네임 해시로 대체한다 — 닉네임은 변경 가능하므로 신뢰도가 낮고,
    그래서 어느 쪽에서 왔는지 접두사로 구분해 둔다.
    """
    slug = clean_text(blog_url, keep_newlines=False, collapse_repeats=False)
    if slug:
        return "u_" + slug
    nick = clean_text(nickname, keep_newlines=False, collapse_repeats=False)
    if not nick:
        return ""
    return "n_" + hashlib.sha1(nick.encode("utf-8")).hexdigest()[:16]


def char_length(text: Any) -> int:
    """공백을 제외한 글자 수 — 회차 분량 지표."""
    cleaned = clean_text(text, keep_newlines=False)
    return len(re.sub(r"\s", "", cleaned))
