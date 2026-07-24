# -*- coding: utf-8 -*-
"""문피아 JSON API 클라이언트.

문피아는 React SPA로 전환되어 HTML에는 <div id="root">만 있고 데이터가 없다.
따라서 DOM 파싱 대신 SPA가 호출하는 내부 JSON API를 그대로 호출한다.
엔드포인트/파라미터는 pc-novel 번들에서 확인한 실제 정의를 따른다.

접속 차단 방지를 위해 요청마다 랜덤 딜레이를 넣고, 5xx/네트워크 오류는
지수 백오프로 재시도한다 (요구사항 4).
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.munpia.com"

# 번들(pc-novel/index.js)에서 추출한 실제 엔드포인트 정의
EP_NOVEL_DETAIL = "/api/v1/pc/novel-detail/{novel_id}"
EP_READ_STATS = "/api/v1/pc/novel-detail/{novel_id}/read-statistics"
EP_CHAPTERS = "/api/v1/pc/novel-detail/{novel_id}/chapters"
EP_ENTRY_INFO = "/api/v1/pc/novel-detail/{novel_id}/entries/{entry_id}/info"
EP_ENTRY_COMMENTS = "/api/v1/pc/novel-detail/{novel_id}/entries/{entry_id}/comments"
EP_LATEST_UPDATE = "/api/novel/latest-update"
EP_GENRE_FREE = "/api/v1/main/genre/free-novel"
EP_GENRE_PAID = "/api/v1/main/genre/paid-novel"

OK_CODE = "M000_00000"

# 번들의 에러코드 → 의미 매핑 (useAxios 청크에서 확인)
CODE_NOVEL_PRIVATE = {"A002_12001", "A002_12002", "A002_12003"}
CODE_CONTRACT_ENDED = {"A002_12004", "A002_12005"}
CODE_ADULT_REQUIRED = {"A002_12006", "A002_12007"}
CODE_APP_ONLY = {"A002_12201"}
CODE_NO_PERMISSION = {"A002_14003"}

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class MunpiaAPIError(RuntimeError):
    """API가 비정상 code를 반환했을 때."""

    def __init__(self, code: str, message: str, url: str = ""):
        super().__init__("[%s] %s (%s)" % (code, message, url))
        self.code = code
        self.message = message
        self.url = url


class NovelUnavailable(MunpiaAPIError):
    """작품이 삭제/비공개/계약종료 상태 — 재시도해도 소용없다."""


class PermissionRequired(MunpiaAPIError):
    """로그인 또는 구매/성인인증이 필요한 리소스."""


class MunpiaClient:
    """레이트리밋과 재시도를 담당하는 얇은 HTTP 래퍼."""

    def __init__(
        self,
        *,
        min_delay: float = 0.8,
        max_delay: float = 1.8,
        timeout: float = 20.0,
        max_retries: int = 3,
        backoff: float = 2.0,
        cookie_file: Optional[str] = None,
        user_agent: str = DEFAULT_UA,
    ):
        if min_delay > max_delay:
            raise ValueError("min_delay는 max_delay보다 클 수 없습니다")
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ko-KR,ko;q=0.9",
            "Origin": BASE_URL,
            "Referer": BASE_URL + "/",
        })
        self._last_request_at = 0.0
        self.logged_in = False
        if cookie_file:
            self.load_cookies(cookie_file)

    # ------------------------------------------------------------------ 인증
    def load_cookies(self, path: str) -> bool:
        """auth.py가 저장한 쿠키 JSON을 세션에 주입한다.

        유료 회차 댓글처럼 권한이 필요한 리소스에만 필요하다.
        비밀번호는 다루지 않는다 — 브라우저에서 직접 로그인한 결과만 재사용한다.
        """
        if not os.path.exists(path):
            log.warning("쿠키 파일이 없습니다: %s (비로그인으로 진행)", path)
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            for c in cookies:
                name, value = c.get("name"), c.get("value")
                if not name or value is None:
                    continue
                self.session.cookies.set(
                    name, value,
                    domain=c.get("domain", ".munpia.com").lstrip("."),
                    path=c.get("path", "/"),
                )
            self.logged_in = True
            log.info("쿠키 %d개 로드 완료: %s", len(cookies), path)
            return True
        except Exception as exc:
            log.error("쿠키 로드 실패 (%s): %s", path, exc)
            return False

    # ------------------------------------------------------------------ 내부
    def _sleep(self) -> None:
        """요청 간 최소 간격을 랜덤하게 유지한다 (요구사항 4: 랜덤 딜레이)."""
        elapsed = time.time() - self._last_request_at
        wait = random.uniform(self.min_delay, self.max_delay) - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        referer_novel_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """API를 호출하고 result 부분을 돌려준다.

        Raises:
            NovelUnavailable: 비공개/삭제/계약종료 — 상위에서 스킵해야 한다.
            PermissionRequired: 로그인/구매/성인인증 필요.
            MunpiaAPIError: 그 밖의 비정상 응답.
            requests.RequestException: 재시도 후에도 실패한 네트워크 오류.
        """
        url = BASE_URL + path
        headers = {}
        if referer_novel_id is not None:
            # SPA와 동일한 Referer를 보내지 않으면 일부 엔드포인트가 400을 낸다
            headers["Referer"] = "%s/novel/detail/%d" % (BASE_URL, referer_novel_id)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._sleep()
            try:
                resp = self.session.get(
                    url, params=params, headers=headers or None, timeout=self.timeout
                )
                self._last_request_at = time.time()

                # 429/5xx는 일시적 — 백오프 후 재시도
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = self.backoff ** attempt + random.uniform(0, 1)
                    log.warning("HTTP %d (%s) — %.1fs 후 재시도 %d/%d",
                                resp.status_code, url, wait, attempt, self.max_retries)
                    time.sleep(wait)
                    last_exc = requests.HTTPError("HTTP %d" % resp.status_code)
                    continue

                try:
                    payload = resp.json()
                except ValueError:
                    raise MunpiaAPIError("PARSE_ERROR", resp.text[:200], url)

                code = payload.get("code", "")
                if code == OK_CODE:
                    return payload.get("result") or {}

                message = payload.get("message", "")
                if code in CODE_NOVEL_PRIVATE or code in CODE_CONTRACT_ENDED:
                    raise NovelUnavailable(code, message, url)
                if (code in CODE_NO_PERMISSION or code in CODE_ADULT_REQUIRED
                        or code in CODE_APP_ONLY):
                    raise PermissionRequired(code, message, url)
                raise MunpiaAPIError(code, message, url)

            except (MunpiaAPIError, NovelUnavailable, PermissionRequired):
                raise  # 애플리케이션 레벨 오류는 재시도해도 결과가 같다
            except requests.RequestException as exc:
                last_exc = exc
                wait = self.backoff ** attempt + random.uniform(0, 1)
                log.warning("네트워크 오류 (%s): %s — %.1fs 후 재시도 %d/%d",
                            url, exc, wait, attempt, self.max_retries)
                time.sleep(wait)

        raise last_exc if last_exc else RuntimeError("요청 실패: %s" % url)

    # --------------------------------------------------------------- 엔드포인트
    def novel_detail(self, novel_id: int) -> Dict[str, Any]:
        return self.get(EP_NOVEL_DETAIL.format(novel_id=novel_id),
                        referer_novel_id=novel_id)

    def read_statistics(self, novel_id: int) -> Dict[str, Any]:
        return self.get(EP_READ_STATS.format(novel_id=novel_id),
                        referer_novel_id=novel_id)

    def chapters_page(self, novel_id: int, page: int = 1, size: int = 100) -> Dict[str, Any]:
        return self.get(EP_CHAPTERS.format(novel_id=novel_id),
                        {"page": page, "size": size},
                        referer_novel_id=novel_id)

    def entry_info(self, novel_id: int, entry_id: int) -> Dict[str, Any]:
        return self.get(EP_ENTRY_INFO.format(novel_id=novel_id, entry_id=entry_id),
                        referer_novel_id=novel_id)

    def comments_page(
        self, novel_id: int, entry_id: int,
        page: int = 1, size: int = 20, order: str = "OLDEST",
    ) -> Dict[str, Any]:
        """회차 댓글 한 페이지.

        order: OLDEST | LATEST | EMPATHY. 시계열 분석에는 OLDEST가 안정적이다
        (수집 중 새 댓글이 달려도 앞 페이지가 밀리지 않는다).
        """
        return self.get(
            EP_ENTRY_COMMENTS.format(novel_id=novel_id, entry_id=entry_id),
            {"order": order, "page": page, "size": size},
            referer_novel_id=novel_id,
        )

    def latest_updates(self) -> Dict[str, Any]:
        """최근 업데이트된 작품 목록 — 크롤 대상 시드 확보용."""
        return self.get(EP_LATEST_UPDATE)

    def genre_novels(self, genre: str = "FANTASY", paid: bool = False) -> Dict[str, Any]:
        """장르별 작품 목록 — 크롤 대상 시드 확보용."""
        ep = EP_GENRE_PAID if paid else EP_GENRE_FREE
        return self.get(ep, {"genreType": genre, "adultMode": "false"})

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "MunpiaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
