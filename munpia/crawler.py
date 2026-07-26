# -*- coding: utf-8 -*-
"""작품 → 회차 → 댓글 순회 오케스트레이션.

설계 원칙: 한 회차의 파싱 실패가 전체 크롤을 중단시키지 않는다 (요구사항 4).
모든 단위 작업을 try/except로 감싸고 실패는 로그에 남긴 뒤 다음 대상으로 넘어간다.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .client import (
    MunpiaAPIError, MunpiaClient, NovelUnavailable, PermissionRequired,
)
from .schema import (
    CommentRecord, EpisodeRecord, NovelRecord,
    enrich_episode, fill_reply_counts, parse_comment, parse_episode, parse_novel,
)

log = logging.getLogger(__name__)

CHAPTER_PAGE_SIZE = 100
COMMENT_PAGE_SIZE = 20
# total이 0으로 오거나 API가 같은 페이지를 반복 반환할 때의 무한 루프 방지.
# 100 * 500 = 5만 회차로, 실제 최장 연재작보다 한참 크다.
MAX_CHAPTER_PAGES = 500
# 권한 오류가 이만큼 연속되면 그 작품의 댓글 수집을 포기한다.
# 유료 연재는 회차 대부분이 비공개라 계속 두드려봐야 요청만 낭비된다.
PERMISSION_FAILURE_LIMIT = 5


class NovelResult(object):
    """작품 하나의 수집 결과 묶음."""

    def __init__(self, novel: Optional[NovelRecord],
                 episodes: List[EpisodeRecord],
                 comments: List[CommentRecord],
                 errors: List[str]):
        self.novel = novel
        self.episodes = episodes
        self.comments = comments
        self.errors = errors

    def __repr__(self) -> str:
        title = self.novel.title if self.novel else "?"
        return "<NovelResult %s ep=%d cmt=%d err=%d>" % (
            title, len(self.episodes), len(self.comments), len(self.errors))


class MunpiaCrawler:
    """문피아 작품 데이터 수집기."""

    def __init__(
        self,
        client: Optional[MunpiaClient] = None,
        *,
        collect_comments: bool = True,
        comment_scope: str = "free",   # free | all | none
        comment_order: str = "OLDEST",
        max_comment_pages: int = 50,
        fetch_entry_detail: bool = False,
        max_episodes: Optional[int] = None,
    ):
        if comment_scope not in ("free", "all", "none"):
            raise ValueError("comment_scope는 free/all/none 중 하나여야 합니다")
        self.client = client or MunpiaClient()
        self.collect_comments = collect_comments and comment_scope != "none"
        self.comment_scope = comment_scope
        self.comment_order = comment_order
        self.max_comment_pages = max_comment_pages
        # 회차 상세는 회차당 요청이 1회 더 든다. 반응 버튼 지표가 필요할 때만 켠다.
        self.fetch_entry_detail = fetch_entry_detail
        self.max_episodes = max_episodes

    # ------------------------------------------------------------------ 작품
    def crawl_novel(self, novel_id: int) -> NovelResult:
        """작품 하나의 메타 + 회차 + 댓글을 수집한다. 예외를 밖으로 던지지 않는다."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        errors: List[str] = []

        novel = self._fetch_novel(novel_id, stamp, errors)
        if novel is None:
            return NovelResult(None, [], [], errors)

        log.info("[%d] %s / %s (%s, %d화)", novel_id, novel.title,
                 novel.author_name, novel.serial_status, novel.chapter_count)

        # 자르기는 _fetch_episodes 안에서 상세 요청 전에 이미 끝났다
        episodes = self._fetch_episodes(novel_id, stamp, errors)

        comments: List[CommentRecord] = []
        if self.collect_comments and episodes:
            comments = self._fetch_all_comments(novel_id, episodes, stamp, errors)

        log.info("[%d] 완료 — 회차 %d건, 댓글 %d건, 오류 %d건",
                 novel_id, len(episodes), len(comments), len(errors))
        return NovelResult(novel, episodes, comments, errors)

    def _fetch_novel(self, novel_id: int, stamp: str,
                     errors: List[str]) -> Optional[NovelRecord]:
        try:
            detail = self.client.novel_detail(novel_id)
        except NovelUnavailable as exc:
            msg = "작품 %d 사용 불가: %s" % (novel_id, exc.message)
            log.warning(msg)
            errors.append(msg)
            return None
        except PermissionRequired as exc:
            msg = "작품 %d 접근 권한 없음: %s" % (novel_id, exc.message)
            log.warning(msg)
            errors.append(msg)
            return None
        except Exception as exc:
            msg = "작품 %d 메타 수집 실패: %s" % (novel_id, exc)
            log.error(msg)
            errors.append(msg)
            return None

        # 독자 통계는 부가 정보 — 실패해도 작품 수집은 계속한다
        stats: Dict[str, Any] = {}
        try:
            stats = self.client.read_statistics(novel_id)
        except Exception as exc:
            log.debug("작품 %d 독자통계 생략: %s", novel_id, exc)

        try:
            return parse_novel(novel_id, detail, stats, stamp)
        except Exception as exc:
            msg = "작품 %d 메타 파싱 실패: %s" % (novel_id, exc)
            log.error(msg)
            errors.append(msg)
            return None

    # ------------------------------------------------------------------ 회차
    def _fetch_episodes(self, novel_id: int, stamp: str,
                        errors: List[str]) -> List[EpisodeRecord]:
        """회차 목록 전체를 페이지네이션으로 수집한다."""
        episodes: List[EpisodeRecord] = []
        seen = set()
        page, total = 1, None

        while page <= MAX_CHAPTER_PAGES:
            try:
                payload = self.client.chapters_page(novel_id, page, CHAPTER_PAGE_SIZE)
            except Exception as exc:
                msg = "작품 %d 회차목록 %d페이지 실패: %s" % (novel_id, page, exc)
                log.error(msg)
                errors.append(msg)
                break

            items = payload.get("list") or []
            if total is None:
                total = payload.get("total", 0)
            if not items:
                break

            before = len(episodes)
            for item in items:
                try:
                    rec = parse_episode(novel_id, item, stamp)
                except Exception as exc:
                    msg = "작품 %d 회차 파싱 실패(%s): %s" % (novel_id, item.get("id"), exc)
                    log.error(msg)
                    errors.append(msg)
                    continue
                if rec.entry_id in seen:
                    continue
                seen.add(rec.entry_id)
                episodes.append(rec)

            # 새로 얻은 회차가 없다 = 같은 페이지를 다시 받았다. 더 진행해도 소득이 없다.
            if len(episodes) == before:
                break
            if total and len(episodes) >= total:
                break
            if len(items) < CHAPTER_PAGE_SIZE:
                break
            page += 1
        else:
            log.warning("작품 %d — 회차 페이지 상한(%d)에 도달했습니다",
                        novel_id, MAX_CHAPTER_PAGES)

        # 회차 번호 오름차순 = 연재 시간 순. 이탈률 계산이 이 순서를 전제한다.
        episodes.sort(key=lambda e: (e.episode_num, e.published_ts))

        # 상세 요청 **전에** 자른다. 뒤에서 자르면 200화짜리 작품에서 --max-episodes 30을
        # 줘도 상세를 200번 부르고 170개를 버린다 — 요청의 85%가 낭비되고 서버에도 그만큼
        # 부담이다. 목록은 페이지당 100건이라 어차피 몇 번이면 끝나므로 여기서만 자르면 된다.
        if self.max_episodes:
            episodes = episodes[: self.max_episodes]

        if self.fetch_entry_detail:
            for ep in episodes:
                try:
                    info = self.client.entry_info(novel_id, ep.entry_id)
                    enrich_episode(ep, info)
                except PermissionRequired:
                    log.debug("회차 %s 상세 권한 없음", ep.episode_uid)
                except Exception as exc:
                    msg = "회차 %s 상세 수집 실패: %s" % (ep.episode_uid, exc)
                    log.warning(msg)
                    errors.append(msg)
        return episodes

    # ------------------------------------------------------------------ 댓글
    def _should_collect(self, ep: EpisodeRecord) -> bool:
        """이 회차의 댓글을 실제로 요청해 볼 가치가 있는지 판단한다.

        문피아 유료 회차는 회차당 과금(100골드)이고, 댓글은 구매/대여한 회차만 열린다.
        구매하지 않은 회차를 두드려 봐야 권한 오류만 돌아오므로 미리 걸러낸다.
        """
        if ep.comment_count <= 0:
            return False
        if ep.is_free == 1:
            return True
        if self.comment_scope != "all":
            return False
        # 유료 회차는 구매·대여한 것만 시도한다.
        # (플래그는 로그인 상태에서만 채워지므로 비로그인이면 자연히 전부 걸러진다)
        return ep.is_purchased == 1 or ep.is_rented == 1

    def _fetch_all_comments(self, novel_id: int, episodes: List[EpisodeRecord],
                            stamp: str, errors: List[str]) -> List[CommentRecord]:
        collected: List[CommentRecord] = []
        permission_failures = 0

        for ep in episodes:
            if not self._should_collect(ep):
                ep.comment_status = "skipped"
                continue
            if permission_failures >= PERMISSION_FAILURE_LIMIT:
                ep.comment_status = "permission"
                continue

            try:
                rows = self._fetch_comments_for_episode(novel_id, ep, stamp)
            except PermissionRequired as exc:
                permission_failures += 1
                ep.comment_status = "permission"
                log.debug("회차 %s 댓글 권한 없음: %s", ep.episode_uid, exc.message)
                if permission_failures == PERMISSION_FAILURE_LIMIT:
                    log.warning("작품 %d — 권한 오류 %d회 연속, 댓글 수집 중단 "
                                "(유료 회차는 로그인/구매 필요)",
                                novel_id, permission_failures)
                continue
            except Exception as exc:
                msg = "회차 %s 댓글 수집 실패: %s" % (ep.episode_uid, exc)
                log.error(msg)
                errors.append(msg)
                ep.comment_status = "error"
                continue

            permission_failures = 0
            ep.comment_status = "ok"
            ep.comment_collected = len(rows)
            collected.extend(rows)

        return fill_reply_counts(collected)

    def _fetch_comments_for_episode(self, novel_id: int, ep: EpisodeRecord,
                                    stamp: str) -> List[CommentRecord]:
        """한 회차의 댓글을 전 페이지 수집한다. 대댓글은 같은 목록에 replyLevel로 섞여 온다."""
        rows: List[CommentRecord] = []
        seen = set()
        page = 1

        while page <= self.max_comment_pages:
            payload = self.client.comments_page(
                novel_id, ep.entry_id, page, COMMENT_PAGE_SIZE, self.comment_order
            )
            items = payload.get("list") or []
            if not items:
                break

            for item in items:
                try:
                    rec = parse_comment(novel_id, ep.entry_id, ep.episode_num, item, stamp)
                except Exception as exc:
                    log.warning("댓글 파싱 실패(%s/%s): %s",
                                ep.episode_uid, item.get("id"), exc)
                    continue
                if rec.comment_id in seen:
                    continue
                seen.add(rec.comment_id)
                rows.append(rec)

            total_pages = payload.get("totalPages") or 0
            if page >= total_pages or len(items) < COMMENT_PAGE_SIZE:
                break
            page += 1

        return rows

    # ------------------------------------------------------------------ 배치
    def crawl_many(self, novel_ids: Iterable[int]) -> Iterable[NovelResult]:
        """여러 작품을 순차 수집한다. 결과를 하나씩 흘려보내 메모리 사용을 억제한다."""
        for novel_id in novel_ids:
            try:
                yield self.crawl_novel(int(novel_id))
            except Exception as exc:  # 크롤러 자체 버그가 배치를 죽이지 않도록
                log.exception("작품 %s 처리 중 예기치 못한 오류: %s", novel_id, exc)
                yield NovelResult(None, [], [], ["작품 %s 치명적 오류: %s" % (novel_id, exc)])

    # ------------------------------------------------------------------ 시드
    def discover_novel_ids(self, genres: Optional[List[str]] = None,
                           include_paid: bool = True) -> List[int]:
        """크롤 대상 후보 작품 ID를 수집한다 (최신 업데이트 + 장르별 목록)."""
        found: List[int] = []
        seen = set()

        def add(items: Any) -> None:
            if not isinstance(items, list):
                return
            for it in items:
                if isinstance(it, dict):
                    nid = it.get("novelId") or it.get("id")
                    if nid and nid not in seen:
                        seen.add(nid)
                        found.append(int(nid))

        try:
            payload = self.client.latest_updates()
            add(payload.get("latestUpdates"))
        except Exception as exc:
            log.warning("최신 업데이트 목록 실패: %s", exc)

        for genre in (genres or ["FANTASY", "NEWFANTASY", "HEROISM", "HISTORY", "ROMANCE"]):
            for paid in ([False, True] if include_paid else [False]):
                try:
                    payload = self.client.genre_novels(genre, paid=paid)
                    for value in payload.values():
                        add(value)
                except Exception as exc:
                    log.warning("장르 목록 실패 (%s, paid=%s): %s", genre, paid, exc)

        log.info("시드 작품 %d개 발견", len(found))
        return found
