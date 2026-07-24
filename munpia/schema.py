# -*- coding: utf-8 -*-
"""수집 레코드 정의와 API 응답 → 레코드 정규화 (요구사항 3.1~3.4).

세 테이블은 다음 키로 조인한다:
    novels.novel_id  1---N  episodes.novel_id
    episodes.episode_uid  1---N  comments.episode_uid
불리언은 전부 0/1 정수, 카운트는 int로 내보내 모델에 바로 넣을 수 있게 한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from .preprocess import (
    char_length, clean_text, epoch, iso, stable_user_key, to_bool_int, to_int,
)


def _fieldnames(cls) -> List[str]:
    return [f.name for f in fields(cls)]


# --------------------------------------------------------------------- 작품
@dataclass
class NovelRecord:
    """작품 기본 정보(메타데이터) — 요구사항 3.1."""

    novel_id: int
    title: str = ""
    author_name: str = ""
    genre_main: str = ""
    genres: str = ""                 # 복수 장르는 '|'로 연결
    introduction: str = ""           # 시놉시스 본문
    group_name: str = ""             # 유료 / 일반연재 / 자유연재 / 작가연재
    serial_status: str = ""          # 연재중 / 완결 / 휴재
    status_code: int = 0             # 0=연재중, 1=완결, 2=휴재 (모델 입력용)

    total_view_count: int = 0        # 총 조회수
    total_like_count: int = 0        # 총 추천수
    preference_count: int = 0        # 총 선작수(선호작품)
    chapter_count: int = 0
    free_chapter_count: int = 0
    total_characters: int = 0        # 작품 전체 글자 수

    is_free: int = 0
    is_adult: int = 0
    is_paid_serial: int = 0
    is_exclusive: int = 0
    is_contest: int = 0
    is_ebook: int = 0

    created_at: str = ""
    updated_at: str = ""
    created_ts: int = 0
    updated_ts: int = 0

    # 독자 인구통계 (read-statistics)
    reader_male_count: int = 0
    reader_female_count: int = 0
    reader_age10s_pct: float = 0.0
    reader_age20s_pct: float = 0.0
    reader_age30s_pct: float = 0.0
    reader_age40s_pct: float = 0.0
    reader_age50s_pct: float = 0.0

    collected_at: str = ""

    FIELDS = None  # 클래스 정의 후 채움

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


NovelRecord.FIELDS = _fieldnames(NovelRecord)


def parse_novel(novel_id: int, detail: Dict[str, Any],
                stats: Optional[Dict[str, Any]] = None,
                collected_at: str = "") -> NovelRecord:
    """`/novel-detail/{id}` + `/read-statistics` 응답을 NovelRecord로 정규화."""
    info = detail.get("novelInfo") or {}
    genres = [clean_text(g, keep_newlines=False) for g in (info.get("genres") or [])]

    finished = to_bool_int(info.get("finish"))
    paused = to_bool_int(info.get("pause"))
    if finished:
        status, code = "완결", 1
    elif paused:
        status, code = "휴재", 2
    else:
        status, code = "연재중", 0

    rec = NovelRecord(
        novel_id=to_int(novel_id),
        title=clean_text(info.get("title"), keep_newlines=False),
        author_name=clean_text(info.get("authorName"), keep_newlines=False),
        genre_main=clean_text(info.get("genreBestName") or (genres[0] if genres else ""),
                              keep_newlines=False),
        genres="|".join(g for g in genres if g),
        introduction=clean_text(info.get("introduction")),
        group_name=clean_text(info.get("groupName"), keep_newlines=False),
        serial_status=status,
        status_code=code,
        total_view_count=to_int(info.get("viewCount")),
        total_like_count=to_int(info.get("likeCount")),
        preference_count=to_int(info.get("preferenceCount")),
        chapter_count=to_int(info.get("chapterCount")),
        free_chapter_count=to_int(info.get("freeChapterCount")),
        total_characters=to_int(info.get("characters")),
        is_free=to_bool_int(info.get("free")),
        is_adult=to_bool_int(info.get("adult")),
        is_paid_serial=to_bool_int(info.get("paidSerial")),
        is_exclusive=to_bool_int(info.get("exclusive")),
        is_contest=to_bool_int(info.get("contest")),
        is_ebook=to_bool_int(info.get("ebook")),
        created_at=iso(info.get("createdAt")),
        updated_at=iso(info.get("updatedAt")),
        created_ts=epoch(info.get("createdAt")),
        updated_ts=epoch(info.get("updatedAt")),
        collected_at=collected_at,
    )

    if stats:
        rec.reader_male_count = to_int(stats.get("maleCount"))
        rec.reader_female_count = to_int(stats.get("femaleCount"))
        for age in (10, 20, 30, 40, 50):
            key = "age%dsPercent" % age
            try:
                value = float(stats.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            setattr(rec, "reader_age%ds_pct" % age, value)
    return rec


# --------------------------------------------------------------------- 회차
@dataclass
class EpisodeRecord:
    """회차별 정량 시계열 지표 — 요구사항 3.2."""

    episode_uid: str = ""            # "{novel_id}_{entry_id}" — 댓글 조인 키
    novel_id: int = 0
    entry_id: int = 0
    episode_num: int = 0
    title: str = ""

    published_at: str = ""
    published_ts: int = 0

    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0           # 목록 기준 댓글 수
    comment_collected: int = 0       # 실제로 수집에 성공한 댓글 수
    pages: int = 0                   # 문피아 표기 분량(쪽)
    char_estimate: int = 0           # pages 기반 추정 글자 수

    is_free: int = 0
    is_adult: int = 0
    is_notice: int = 0
    # 로그인 상태에서만 의미가 있다. 유료 회차 댓글은 구매/대여한 회차만 열리므로
    # 수집 대상을 고를 때 이 값으로 헛된 요청을 걸러낸다.
    is_purchased: int = 0
    is_rented: int = 0

    # 회차 상세(entry_info)에서만 얻는 독자 반응 버튼 — 고정 팬층 신호
    reaction_total: int = 0
    reaction_best: int = 0
    reaction_funny: int = 0
    reaction_amazing: int = 0
    reaction_cheer: int = 0
    reaction_impressed: int = 0
    author_comment: str = ""

    comment_status: str = "ok"       # ok / permission / error / skipped
    collected_at: str = ""

    FIELDS = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EpisodeRecord.FIELDS = _fieldnames(EpisodeRecord)

# 문피아 뷰어 1쪽 ≈ 900자. 본문 전체는 robots.txt가 막은 /novel/viewer/ 뒤에 있으므로
# 분량은 pages로 추정한다. 절대값보다 회차 간 상대 비교에 쓰라는 의미의 근사치다.
CHARS_PER_PAGE = 900


def parse_episode(novel_id: int, item: Dict[str, Any],
                  collected_at: str = "") -> EpisodeRecord:
    """`/chapters` 목록의 한 항목을 EpisodeRecord로 정규화."""
    entry_id = to_int(item.get("id"))
    pages = to_int(item.get("pages"))
    return EpisodeRecord(
        episode_uid="%d_%d" % (to_int(novel_id), entry_id),
        novel_id=to_int(novel_id),
        entry_id=entry_id,
        episode_num=to_int(item.get("num")),
        title=clean_text(item.get("title"), keep_newlines=False),
        published_at=iso(item.get("createdAt")),
        published_ts=epoch(item.get("createdAt")),
        view_count=to_int(item.get("viewCount")),
        like_count=to_int(item.get("likeCount")),
        comment_count=to_int(item.get("commentCount")),
        pages=pages,
        char_estimate=pages * CHARS_PER_PAGE,
        is_free=to_bool_int(item.get("free")),
        is_adult=to_bool_int(item.get("adult")),
        is_notice=to_bool_int(item.get("notice")),
        is_purchased=to_bool_int(item.get("purchased")),
        is_rented=to_bool_int(item.get("rented")),
        collected_at=collected_at,
    )


def enrich_episode(rec: EpisodeRecord, info: Dict[str, Any]) -> EpisodeRecord:
    """`/entries/{id}/info` 응답으로 반응 버튼 집계를 채운다."""
    entry = info.get("entry") or {}
    reaction = entry.get("reaction") or {}
    rec.reaction_total = to_int(reaction.get("totalCount"))
    rec.reaction_best = to_int(reaction.get("bestCount"))
    rec.reaction_funny = to_int(reaction.get("funnyCount"))
    rec.reaction_amazing = to_int(reaction.get("amazingCount"))
    rec.reaction_cheer = to_int(reaction.get("cheerCount"))
    rec.reaction_impressed = to_int(reaction.get("impressedCount"))
    rec.author_comment = clean_text(entry.get("authorComment"))
    # 상세가 목록보다 최신이므로 조회/추천은 상세 값으로 덮어쓴다
    if entry.get("viewCount") is not None:
        rec.view_count = to_int(entry.get("viewCount"))
    if entry.get("likeCount") is not None:
        rec.like_count = to_int(entry.get("likeCount"))
    return rec


# --------------------------------------------------------------------- 댓글
@dataclass
class CommentRecord:
    """회차별 독자 반응(댓글) — 요구사항 3.3."""

    comment_uid: str = ""            # "{novel_id}_{entry_id}_{comment_id}"
    episode_uid: str = ""            # episodes 조인 키
    novel_id: int = 0
    entry_id: int = 0
    episode_num: int = 0
    comment_id: int = 0
    parent_id: int = 0               # 0이면 최상위 댓글

    user_key: str = ""               # 유저별 연속 작성/유지율 추적용 안정 식별자
    nickname: str = ""
    blog_url: str = ""

    content_type: str = "TEXT"       # TEXT / STICKER
    body: str = ""                   # 감정 분석 입력 텍스트
    sticker_url: str = ""
    body_char_len: int = 0

    created_at: str = ""
    created_ts: int = 0

    like_count: int = 0
    dislike_count: int = 0

    reply_level: int = 0             # 0=원댓글, 1 이상=대댓글
    is_reply: int = 0
    reply_count: int = 0             # 이 댓글에 달린 대댓글 수 (수집 후 집계)
    is_secret: int = 0
    is_blocked: int = 0

    collected_at: str = ""

    FIELDS = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CommentRecord.FIELDS = _fieldnames(CommentRecord)


def parse_comment(novel_id: int, entry_id: int, episode_num: int,
                  item: Dict[str, Any], collected_at: str = "") -> CommentRecord:
    """댓글 API의 한 항목을 CommentRecord로 정규화.

    STICKER 타입은 content가 텍스트가 아니라 이미지 URL이므로 body에 넣지 않는다.
    감정 분석 학습 시 body가 빈 행은 제외하면 된다.
    """
    comment_id = to_int(item.get("id"))
    content_type = clean_text(item.get("contentType"), keep_newlines=False).upper() or "TEXT"
    raw_content = item.get("content")

    if content_type == "STICKER":
        body, sticker = "", clean_text(raw_content, keep_newlines=False)
    else:
        body, sticker = clean_text(raw_content), ""

    level = to_int(item.get("replyLevel"))
    return CommentRecord(
        comment_uid="%d_%d_%d" % (to_int(novel_id), to_int(entry_id), comment_id),
        episode_uid="%d_%d" % (to_int(novel_id), to_int(entry_id)),
        novel_id=to_int(novel_id),
        entry_id=to_int(entry_id),
        episode_num=to_int(episode_num),
        comment_id=comment_id,
        parent_id=to_int(item.get("parentId")),
        user_key=stable_user_key(item.get("blogUrl"), item.get("nickName")),
        nickname=clean_text(item.get("nickName"), keep_newlines=False),
        blog_url=clean_text(item.get("blogUrl"), keep_newlines=False),
        content_type=content_type,
        body=body,
        sticker_url=sticker,
        body_char_len=char_length(body),
        created_at=iso(item.get("createdAt")),
        created_ts=epoch(item.get("createdAt")),
        like_count=to_int(item.get("likeCount")),
        dislike_count=to_int(item.get("dislikeCount")),
        reply_level=level,
        is_reply=1 if level > 0 else 0,
        is_secret=to_bool_int(item.get("secret")),
        is_blocked=to_bool_int(item.get("block")),
        collected_at=collected_at,
    )


def fill_reply_counts(comments: List[CommentRecord]) -> List[CommentRecord]:
    """parent_id를 집계해 각 댓글의 대댓글 개수를 채운다 (요구사항 3.3)."""
    counts: Dict[int, int] = {}
    for c in comments:
        if c.parent_id:
            counts[c.parent_id] = counts.get(c.parent_id, 0) + 1
    for c in comments:
        c.reply_count = counts.get(c.comment_id, 0)
    return comments
