# -*- coding: utf-8 -*-
"""로그인 세션 쿠키 확보.

유료 회차 댓글은 비로그인 상태에서 A002_14003(권한 없음)이 뜬다. 그 경우에만 필요하다.

두 가지 경로를 제공한다.
  1. .env의 자격증명으로 로그인 폼을 직접 POST (기본, 의존성 없음)
  2. Playwright로 브라우저를 띄워 수동 로그인 (캡차·SNS 로그인·2단계 인증 대응)

어느 쪽이든 결과물은 동일한 형식의 쿠키 JSON이고, 비밀번호는 저장하지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

LOGIN_PAGE = "https://nssl.munpia.com/login"
LOGIN_POST = "https://nssl.munpia.com/login"
HOME_URL = "https://www.munpia.com/"
ME_URL = "https://www.munpia.com/api/member/my-info-simple"

DEFAULT_COOKIE_PATH = "data/cookies.json"
DEFAULT_ENV_PATH = ".env"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

ENV_ID_KEYS = ("MUNPIA_ID", "MUNPIA_USERNAME", "MUNPIA_USER")
ENV_PW_KEYS = ("MUNPIA_PW", "MUNPIA_PASSWORD", "MUNPIA_PASS")


# ------------------------------------------------------------------ .env 로딩
def load_dotenv(path: str = DEFAULT_ENV_PATH) -> Dict[str, str]:
    """.env를 파싱해 dict로 돌려준다. 외부 의존성을 쓰지 않는다.

    `KEY=value` 형식만 지원하며 `#` 주석과 따옴표를 처리한다.
    os.environ에 주입하지 않는다 — 자격증명이 하위 프로세스로 새는 걸 막기 위해서다.
    """
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.lower().startswith("export "):
                    key = key[7:].strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key] = value
    except Exception as exc:
        log.error(".env 파싱 실패 (%s): %s", path, exc)
    return values


def read_credentials(env_path: str = DEFAULT_ENV_PATH
                     ) -> Tuple[Optional[str], Optional[str]]:
    """.env → 환경변수 순으로 아이디/비밀번호를 찾는다.

    반환값을 로그에 남기지 않는다. 호출부도 그래야 한다.
    """
    env = load_dotenv(env_path)

    def pick(keys) -> Optional[str]:
        for k in keys:
            v = env.get(k) or os.environ.get(k)
            if v:
                return v
        return None

    return pick(ENV_ID_KEYS), pick(ENV_PW_KEYS)


# ------------------------------------------------------------------ 쿠키 저장
def _session_cookies(session: requests.Session) -> List[dict]:
    """requests 쿠키를 Playwright와 동일한 형식으로 직렬화한다."""
    out = []
    for c in session.cookies:
        out.append({
            "name": c.name,
            "value": c.value,
            "domain": c.domain or ".munpia.com",
            "path": c.path or "/",
        })
    return out


def save_cookies(cookies: List[dict], cookie_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(cookie_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cookie_path, "w", encoding="utf-8") as f:
        json.dump(cookies, f, ensure_ascii=False, indent=1)
    log.info("쿠키 %d개를 저장했습니다: %s", len(cookies), cookie_path)


# ------------------------------------------------------------------ 세션 검증
def session_is_logged_in(session: requests.Session) -> Tuple[bool, dict]:
    """`my-info-simple`의 login 플래그로 로그인 여부를 확인한다."""
    try:
        r = session.get(ME_URL, timeout=20,
                        headers={"Accept": "application/json", "Referer": HOME_URL})
        info = r.json()
        return bool(info.get("login")), info
    except Exception as exc:
        log.debug("로그인 상태 확인 실패: %s", exc)
        return False, {}


# --------------------------------------------------------- 폼 로그인 (.env 방식)
def login_with_credentials(username: str, password: str,
                           cookie_path: str = DEFAULT_COOKIE_PATH) -> bool:
    """로그인 폼을 직접 POST 해서 세션 쿠키를 확보한다.

    문피아 로그인 페이지의 reCAPTCHA는 조건부로만 렌더된다(로그인 실패 누적 등).
    캡차가 켜진 상태가 감지되면 자동 로그인을 포기하고 브라우저 방식을 안내한다.

    비밀번호는 요청 본문에만 쓰이고 저장·로깅되지 않는다.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        "Accept-Language": "ko-KR,ko;q=0.9",
    })

    # 1) 로그인 페이지에서 CSRF 토큰과 세션 쿠키를 받는다
    try:
        page = session.get(LOGIN_PAGE, timeout=25)
        page.raise_for_status()
    except Exception as exc:
        log.error("로그인 페이지 요청 실패: %s", exc)
        return False

    html = page.text
    m = (re.search(r'<input[^>]*name="_csrf"[^>]*value="([^"]+)"', html)
         or re.search(r'<meta[^>]*name="_csrf"[^>]*content="([^"]+)"', html))
    if not m:
        log.error("CSRF 토큰을 찾지 못했습니다. 로그인 페이지 구조가 바뀐 것 같습니다.")
        return False
    csrf = m.group(1)

    # 캡차 위젯이 실제로 렌더된 상태인지 확인 (평소에는 주석만 있고 비어 있다)
    if re.search(r'class="[^"]*g-recaptcha[^"]*"', html) or "data-sitekey" in html:
        log.error("로그인 페이지에 reCAPTCHA가 활성화되어 있습니다. "
                  "자동 로그인을 진행할 수 없습니다.")
        log.error("→ python -m munpia.cli login --browser 로 직접 로그인해 주세요.")
        return False

    # 2) 자격증명 제출
    payload = {
        "pageType": "PC",
        "redirectUrl": "",
        "_csrf": csrf,
        "username": username,
        "password": password,
        "autoLogin": "Y",
    }
    try:
        resp = session.post(
            LOGIN_POST, data=payload, timeout=25, allow_redirects=True,
            headers={"Referer": LOGIN_PAGE,
                     "Origin": "https://nssl.munpia.com",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
    except Exception as exc:
        log.error("로그인 요청 실패: %s", exc)
        return False

    # 3) 홈을 한 번 거쳐 .munpia.com 전역 쿠키를 확보한다
    try:
        session.get(HOME_URL, timeout=25)
    except Exception:
        pass

    # 4) 실제로 로그인됐는지 API로 확인한다 (HTTP 200만으로는 판정할 수 없다)
    ok, info = session_is_logged_in(session)
    if not ok:
        reason = _extract_login_error(resp.text)
        log.error("로그인 실패%s", (": " + reason) if reason else "")
        log.error("→ 아이디/비밀번호를 확인하거나 "
                  "python -m munpia.cli login --browser 를 사용하세요.")
        return False

    save_cookies(_session_cookies(session), cookie_path)
    log.info("로그인 성공 (level=%s, 성인인증=%s)",
             info.get("level"), info.get("adultVerification"))
    return True


def _extract_login_error(html: str) -> str:
    """로그인 결과 모달에 담긴 실패 사유를 뽑아낸다."""
    m = re.search(r'id="loginResult"[^>]*>(.*?)</', html, re.S)
    if not m:
        return ""
    text = re.sub(r"<[^>]+>", " ", m.group(1))
    return re.sub(r"\s+", " ", text).strip()[:200]


# ------------------------------------------------------ 브라우저 로그인 (fallback)
def login_with_browser(cookie_path: str = DEFAULT_COOKIE_PATH,
                       timeout_sec: int = 300) -> bool:
    """브라우저 창을 띄워 수동 로그인을 받고 쿠키를 저장한다.

    캡차·SNS 로그인·2단계 인증처럼 폼 POST로 넘길 수 없는 경우에 쓴다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("playwright가 설치되지 않았습니다. "
                  "pip install playwright && playwright install chromium")
        return False

    import time

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(locale="ko-KR", user_agent=UA)
        page = context.new_page()
        page.goto(LOGIN_PAGE, wait_until="domcontentloaded")

        print("=" * 68)
        print(" 열린 브라우저 창에서 직접 로그인해 주세요.")
        print(" 로그인이 완료되면 자동으로 쿠키를 저장합니다. (최대 %d초 대기)" % timeout_sec)
        print("=" * 68)

        deadline = time.time() + timeout_sec
        logged_in = False
        while time.time() < deadline:
            try:
                # 페이지가 아니라 API로 판정한다 — URL 패턴보다 확실하다
                result = page.evaluate(
                    "() => fetch('%s', {credentials:'include'})"
                    ".then(r => r.json()).then(j => j.login).catch(() => false)" % ME_URL
                )
                if result:
                    logged_in = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not logged_in:
            print("로그인이 감지되지 않았습니다. 현재 쿠키를 그대로 저장합니다.")

        try:
            page.goto(HOME_URL, wait_until="domcontentloaded")
            time.sleep(1.5)
        except Exception:
            pass

        cookies = context.cookies()
        browser.close()

    if not cookies:
        log.error("쿠키를 가져오지 못했습니다")
        return False

    save_cookies(cookies, cookie_path)
    return logged_in


# ------------------------------------------------------------------- 진입점
def login(cookie_path: str = DEFAULT_COOKIE_PATH,
          env_path: str = DEFAULT_ENV_PATH,
          force_browser: bool = False) -> bool:
    """.env에 자격증명이 있으면 자동 로그인, 없거나 실패하면 브라우저로 넘긴다."""
    if force_browser:
        return login_with_browser(cookie_path)

    username, password = read_credentials(env_path)
    if username and password:
        log.info("%s의 자격증명으로 로그인을 시도합니다 (아이디: %s)",
                 env_path, _mask(username))
        if login_with_credentials(username, password, cookie_path):
            return True
        log.info("자동 로그인에 실패했습니다. 브라우저 방식으로 전환합니다.")
        return login_with_browser(cookie_path)

    log.info("%s에서 자격증명을 찾지 못했습니다 (MUNPIA_ID / MUNPIA_PW). "
             "브라우저 로그인으로 진행합니다.", env_path)
    return login_with_browser(cookie_path)


def _mask(value: str) -> str:
    """아이디를 부분 마스킹해 로그에 남긴다. 비밀번호는 절대 로그에 넣지 않는다."""
    if len(value) <= 2:
        return value[0] + "*"
    return value[:2] + "*" * (len(value) - 2)


# ------------------------------------------------------------------- 진단
def check_login(cookie_path: str = DEFAULT_COOKIE_PATH,
                novel_id: Optional[int] = None,
                entry_id: Optional[int] = None) -> None:
    """저장한 쿠키로 유료 회차 댓글이 실제로 열리는지 확인한다.

    로그인만으로 되는지, 회차 구매까지 필요한지를 판별하는 용도다.
    """
    from .client import MunpiaClient, PermissionRequired

    client = MunpiaClient(cookie_file=cookie_path)
    if not client.logged_in:
        print("쿠키를 로드하지 못했습니다: %s" % cookie_path)
        print("→ python -m munpia.cli login 을 먼저 실행하세요.")
        return

    ok, info = session_is_logged_in(client.session)
    print("세션 로그인 상태: %s" % ("로그인됨" if ok else "비로그인"))
    if ok:
        print("  회원 등급 level=%s / 성인인증=%s"
              % (info.get("level"), info.get("adultVerification")))
    else:
        print("→ 쿠키가 만료되었을 수 있습니다. 다시 로그인하세요.")
        return

    if novel_id is None:
        print("\n유료 회차 댓글 접근을 확인하려면 --novel-id 를 지정하세요.")
        return

    if entry_id is None:
        try:
            payload = client.chapters_page(novel_id, 1, 100)
            paid = [c for c in (payload.get("list") or [])
                    if not c.get("free") and (c.get("commentCount") or 0) > 0]
            if not paid:
                print("댓글이 달린 유료 회차를 찾지 못했습니다.")
                return
            entry_id = paid[0]["id"]
            print("\n테스트 대상 유료 회차: entry_id=%s (댓글 %s개)"
                  % (entry_id, paid[0].get("commentCount")))
        except Exception as exc:
            print("회차 목록 조회 실패: %s" % exc)
            return

    try:
        payload = client.comments_page(novel_id, entry_id, 1, 20)
        print("성공 — 유료 회차 댓글 %s건 접근 가능 (전체 %s건)"
              % (len(payload.get("list") or []), payload.get("total")))
        print("→ --comment-scope all --cookies %s 로 수집하세요." % cookie_path)
    except PermissionRequired as exc:
        print("여전히 권한 없음: %s" % exc.message)
        print("→ 로그인만으로는 부족하고 해당 회차 구매/대여가 필요합니다.")
        print("→ --comment-scope free 로 무료 회차만 수집하세요.")
    except Exception as exc:
        print("확인 실패: %s" % exc)
