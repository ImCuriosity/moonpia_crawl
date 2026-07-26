# -*- coding: utf-8 -*-
"""대화형 실행 마법사.

`run.bat`(Windows) 또는 `run.sh`(macOS/Linux)가 이 모듈을 실행한다.
명령줄 옵션을 모르는 사용자도 질문에 답하기만 하면 수집이 끝나도록 하는 게 목적이다.

세부 제어가 필요하면 `python -m munpia.cli` 쪽을 쓰면 된다.
"""
from __future__ import annotations

import getpass
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

from .auth import (
    DEFAULT_COOKIE_PATH, DEFAULT_ENV_PATH, login_with_browser,
    login_with_credentials, read_credentials, session_is_logged_in,
)
from .client import MunpiaClient, PermissionRequired
from .cli import parse_novel_ids
from .crawler import MunpiaCrawler
from .storage import DatasetWriter

RAW_DIR = os.path.join("data", "raw")
FEATURE_DIR = os.path.join("data", "features")

BAR = "=" * 66
LINE = "-" * 66


# ------------------------------------------------------------------ 출력 유틸
def title(text: str) -> None:
    print("\n" + BAR)
    print("  " + text)
    print(BAR)


def step(n: int, total: int, text: str) -> None:
    print("\n[%d/%d] %s" % (n, total, text))
    print(LINE)


def _clean_input(value: str) -> str:
    """입력에서 BOM·제로폭·따옴표 같은 눈에 안 보이는 문자를 제거한다.

    복사·붙여넣기나 파이프 입력에는 BOM이 섞여 들어오는 일이 흔하고,
    그대로 두면 'n'이 '﻿n'이 되어 멀쩡한 답이 거부된다.
    """
    return value.replace("﻿", "").replace("​", "").strip().strip('"\'')


def ask(prompt: str, default: str = "") -> str:
    suffix = " [%s]" % default if default else ""
    try:
        value = _clean_input(input("  %s%s: " % (prompt, suffix)))
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


YES_WORDS = ("y", "yes", "ye", "예", "네", "ㅇ", "ㅛ", "1", "o", "ok")
NO_WORDS = ("n", "no", "아니", "아니오", "아니요", "ㄴ", "ㅜ", "0", "x")


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    for _ in range(5):
        value = ask("%s (%s)" % (prompt, hint)).lower()
        if not value:
            return default
        if value in YES_WORDS:
            return True
        if value in NO_WORDS:
            return False
        print("     y(예) 또는 n(아니오) 으로 답해 주세요.")
    # 다섯 번 연속 인식 실패면 입력이 정상이 아니다. 기본값으로 넘어간다.
    print("     기본값(%s)으로 진행합니다." % ("예" if default else "아니오"))
    return default


def ask_int(prompt: str, default: Optional[int] = None,
            minimum: int = 1) -> Optional[int]:
    for _ in range(5):
        raw = ask(prompt, str(default) if default is not None else "")
        if not raw:
            return default
        try:
            value = int(raw.replace(",", "").replace("개", "").strip())
        except ValueError:
            print("     숫자를 입력해 주세요.")
            continue
        if value < minimum:
            print("     %d 이상이어야 합니다." % minimum)
            continue
        return value
    return default


# --------------------------------------------------------------------- 1. 로그인
def do_login(cookie_path: str) -> Tuple[MunpiaClient, bool]:
    """로그인 세션을 확보한다. 반환값은 (클라이언트, 로그인여부)."""
    step(1, 4, "로그인")

    # 이미 저장된 세션이 살아 있으면 재사용한다 (재로그인은 실패 카운트만 쌓는다)
    if os.path.exists(cookie_path):
        client = MunpiaClient(cookie_file=cookie_path)
        ok, info = session_is_logged_in(client.session)
        if ok:
            print("  저장된 로그인 세션을 찾았습니다 (등급 level=%s)." % info.get("level"))
            if ask_yes("  이 세션을 그대로 사용할까요?"):
                return client, True
        else:
            print("  저장된 세션이 만료되었습니다. 다시 로그인합니다.")

    print("  유료 회차 댓글을 수집하려면 로그인이 필요합니다.")
    print("  무료 회차만 수집한다면 로그인 없이도 됩니다.")
    if not ask_yes("  로그인하시겠습니까?"):
        return MunpiaClient(), False

    # .env에 자격증명이 있으면 우선 사용
    env_id, env_pw = read_credentials(DEFAULT_ENV_PATH)
    if env_id and env_pw:
        print("  .env에서 자격증명을 찾았습니다 (아이디: %s)." % _mask(env_id))
        if ask_yes("  이 계정으로 로그인할까요?"):
            if login_with_credentials(env_id, env_pw, cookie_path):
                return MunpiaClient(cookie_file=cookie_path), True
            print("  자동 로그인에 실패했습니다.")

    for attempt in range(3):
        username = ask("문피아 아이디")
        if not username:
            print("  아이디를 입력하지 않아 비로그인으로 진행합니다.")
            return MunpiaClient(), False
        # getpass: 입력한 비밀번호가 화면에 표시되지 않는다
        try:
            password = getpass.getpass("  비밀번호 (입력해도 화면에 보이지 않습니다): ")
        except Exception:
            password = ask("비밀번호")
        if not password:
            print("  비밀번호가 비어 있습니다.")
            continue

        print("  로그인 중...")
        if login_with_credentials(username, password, cookie_path):
            if ask_yes("  다음 실행을 위해 .env에 저장할까요?", default=False):
                _save_env(username, password)
            del password
            return MunpiaClient(cookie_file=cookie_path), True

        del password
        print("  로그인 실패 (%d/3)" % (attempt + 1))
        if attempt < 2 and not ask_yes("  다시 시도할까요?"):
            break

    print("\n  아이디/비밀번호 로그인에 실패했습니다.")
    if ask_yes("  브라우저 창을 띄워 직접 로그인할까요?", default=False):
        if login_with_browser(cookie_path):
            return MunpiaClient(cookie_file=cookie_path), True

    print("  비로그인 상태로 진행합니다 (무료 회차 댓글만 수집됩니다).")
    return MunpiaClient(), False


def _mask(value: str) -> str:
    if len(value) <= 2:
        return value[0] + "*"
    return value[:2] + "*" * (len(value) - 2)


def _save_env(username: str, password: str) -> None:
    try:
        with open(DEFAULT_ENV_PATH, "w", encoding="utf-8") as f:
            f.write("MUNPIA_ID=%s\nMUNPIA_PW=%s\n" % (username, password))
        print("  .env에 저장했습니다. (이 파일은 .gitignore에 등록되어 있습니다)")
    except Exception as exc:
        print("  .env 저장 실패: %s" % exc)


# ------------------------------------------------------- 2. 유료 댓글 접근 판정
def explain_scope(logged_in: bool) -> str:
    """수집 범위를 정하고, 유료 회차가 어디까지 열리는지 사용자에게 설명한다."""
    if not logged_in:
        print("\n  비로그인 상태입니다 → 무료 회차 댓글만 수집합니다.")
        return "free"

    print("\n  로그인됨 → 무료 회차 + '구매·대여한 유료 회차'의 댓글을 수집합니다.")
    print()
    print("  참고: 문피아는 회차당 결제(100골드 ≈ 100원) 방식입니다.")
    print("        무제한 정액제가 아니라서, 유료 회차 댓글은 회원님이 실제로")
    print("        구매·대여한 회차만 열립니다. 구매하지 않은 회차는 자동으로")
    print("        건너뛰므로 헛된 요청이 발생하지 않습니다.")
    print()
    print("  이 프로그램은 조회만 하며 결제 API는 호출하지 않습니다.")
    print("  수집 도중 돈이 빠져나가는 일은 없습니다.")
    return "all"


# --------------------------------------------------------------- 3. 대상 선택
def choose_targets(crawler: MunpiaCrawler) -> List[int]:
    step(2, 4, "수집할 작품 선택")
    print("  1) 인기·최신 작품을 자동으로 찾기")
    print("  2) 작품 주소(URL) 또는 번호 직접 입력")
    print("  3) 파일에서 목록 불러오기 (한 줄에 하나씩)")

    choice = ask("선택", "1")

    if choice == "2":
        print("\n  작품 주소나 번호를 입력하세요. 여러 개면 띄어쓰기로 구분합니다.")
        print("  예: https://www.munpia.com/novel/detail/479065  또는  479065")
        raw = ask("작품")
        ids = parse_novel_ids(raw.split())
        if not ids:
            print("  인식된 작품이 없습니다.")
        return ids

    if choice == "3":
        path = ask("파일 경로", "novel_ids.txt")
        if not os.path.exists(path):
            print("  파일을 찾을 수 없습니다: %s" % path)
            return []
        with open(path, "r", encoding="utf-8") as f:
            return parse_novel_ids(f.read().splitlines())

    limit = ask_int("몇 개 작품을 수집할까요?", 20)
    print("  작품 목록을 찾는 중...")
    ids = crawler.discover_novel_ids()
    if not ids:
        print("  작품을 찾지 못했습니다. 네트워크를 확인해 주세요.")
        return []
    print("  %d개 발견 → 상위 %d개를 수집합니다." % (len(ids), min(limit, len(ids))))
    return ids[:limit]


# --------------------------------------------------------------- 4. 수집 옵션
def choose_options(default_scope: str) -> Tuple[str, Optional[int], bool]:
    step(3, 4, "수집 옵션")

    scope = default_scope
    if default_scope == "all":
        print("  댓글 수집 범위: 무료 회차 + 구매·대여한 유료 회차")
        if not ask_yes("  이대로 진행할까요? (아니오 = 무료 회차만)"):
            scope = "free"
    else:
        print("  댓글 수집 범위: 무료 회차만")

    max_episodes = ask_int("작품당 최대 회차 수 (엔터=제한 없음)", None)

    detail = ask_yes("  회차별 반응 버튼(최고/웃김/감동 등)도 수집할까요? "
                     "(시간이 약 2배)", default=False)
    return scope, max_episodes, detail


# ------------------------------------------------------------------ 5. 수집 실행
def run_crawl(crawler: MunpiaCrawler, novel_ids: List[int]) -> dict:
    step(4, 4, "수집 시작")
    print("  중단하려면 Ctrl+C 를 누르세요. 그때까지 모은 데이터는 저장됩니다.\n")

    started = time.time()
    with DatasetWriter(RAW_DIR, fmt="csv", resume=True) as writer:
        targets = [i for i in novel_ids if not writer.is_done(i)]
        skipped = len(novel_ids) - len(targets)
        if skipped:
            print("  이미 수집한 작품 %d개는 건너뜁니다." % skipped)
        if not targets:
            print("  새로 수집할 작품이 없습니다.")
            return writer.summary()

        done = 0
        try:
            for result in crawler.crawl_many(targets):
                done += 1
                writer.write_result(result)
                if result.novel is None:
                    print("  [%d/%d] 건너뜀 (비공개이거나 접근할 수 없는 작품)"
                          % (done, len(targets)))
                    continue
                print("  [%d/%d] %s — 회차 %d건, 댓글 %d건%s"
                      % (done, len(targets), result.novel.title[:28],
                         len(result.episodes), len(result.comments),
                         "  (오류 %d)" % len(result.errors) if result.errors else ""))
        except KeyboardInterrupt:
            print("\n  사용자가 중단했습니다. 여기까지의 데이터는 저장되었습니다.")

        summary = writer.summary()

    elapsed = time.time() - started
    print("\n  소요 시간: %d분 %d초" % (elapsed // 60, elapsed % 60))
    return summary


# --------------------------------------------------------------- 6. 피처 생성
def build_features() -> bool:
    print("\n  학습용 피처를 만드는 중...")
    try:
        import pandas as pd
    except ImportError:
        print("  pandas가 없어 피처 생성을 건너뜁니다. (pip install pandas)")
        return False

    from .features import build_dataset

    def read(name: str):
        path = os.path.join(RAW_DIR, "%s.csv" % name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    novels, episodes, comments = read("novels"), read("episodes"), read("comments")
    if episodes.empty:
        print("  수집된 회차가 없어 피처를 만들지 않았습니다.")
        return False

    try:
        ep_feat, users = build_dataset(novels, episodes, comments)
    except Exception as exc:
        print("  피처 생성 실패: %s" % exc)
        return False

    os.makedirs(FEATURE_DIR, exist_ok=True)
    ep_feat.to_csv(os.path.join(FEATURE_DIR, "episode_features.csv"),
                   index=False, encoding="utf-8-sig")
    if not users.empty:
        users.to_csv(os.path.join(FEATURE_DIR, "user_features.csv"),
                     index=False, encoding="utf-8-sig")
        print("  회차 피처 %d행, 독자 피처 %d행 (고정팬 %d명)"
              % (len(ep_feat), len(users), int(users["is_core_fan"].sum())))
    else:
        print("  회차 피처 %d행 (댓글이 없어 독자 피처는 생략)" % len(ep_feat))
    return True


MODEL_DIR = os.path.join("data", "models")


def run_sentiment() -> bool:
    """댓글 감정 분석 (선택). torch/transformers가 있어야 한다.

    수백 MB짜리 모델을 받으므로 묻지 않고 돌리지 않는다. 의존성이 없으면
    설치 방법만 알려주고 넘어간다 — 여기서 멈추면 이미 모은 데이터가 아깝다.
    """
    try:
        import pandas as pd
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("\n  댓글 감정 분석은 건너뜁니다 (torch·transformers 미설치).")
        print("    쓰려면: pip install torch transformers")
        return False

    comments_path = os.path.join(RAW_DIR, "comments.csv")
    if not os.path.exists(comments_path):
        return False
    comments = pd.read_csv(comments_path)
    if comments.empty:
        return False

    print("\n  댓글 %d건에 감정 분석(KOTE)을 돌릴 수 있습니다." % len(comments))
    print("  처음 한 번은 모델을 내려받느라 몇 분 걸립니다.")
    if not ask_yes("  감정 분석을 할까요?", default=False):
        return False

    from .sentiment import run as run_kote
    novels_path = os.path.join(RAW_DIR, "novels.csv")
    novels = pd.read_csv(novels_path) if os.path.exists(novels_path) else pd.DataFrame()
    try:
        out = run_kote(comments, novels, out_dir=FEATURE_DIR)
    except Exception as exc:
        print("  감정 분석 실패: %s" % exc)
        return False
    print("  회차 감정 %d행을 만들었습니다." % len(out))
    return True


def train_models() -> bool:
    """이탈 예측 모델 학습 (선택)."""
    try:
        import pandas as pd
        import sklearn  # noqa: F401
    except ImportError:
        print("\n  모델 학습은 건너뜁니다 (scikit-learn 미설치).")
        print("    쓰려면: pip install scikit-learn")
        return False

    ep_path = os.path.join(FEATURE_DIR, "episode_features.csv")
    if not os.path.exists(ep_path):
        return False

    print("\n  수집한 데이터로 이탈 예측 모델을 학습할 수 있습니다.")
    if not ask_yes("  모델을 학습할까요?", default=True):
        return False

    from .model import run as run_model

    def read(d: str, name: str):
        path = os.path.join(d, "%s.csv" % name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    ep_feat = pd.read_csv(ep_path)
    sent = read(FEATURE_DIR, "episode_sentiment")
    if not sent.empty:
        ep_feat = ep_feat.merge(sent, on=["novel_id", "episode_num"], how="left")

    episodes, comments = read(RAW_DIR, "episodes"), read(RAW_DIR, "comments")
    novels = read(RAW_DIR, "novels")

    trained = 0
    for task in ("episode", "user", "novel"):
        if task == "user" and comments.empty:
            continue
        try:
            # 유료 구간은 조회수가 세는 대상이 바뀌어 이탈률이 성립하지 않는다.
            # 마법사 기본값은 무료 구간(1~25화)으로 고정한다.
            run_model(ep_feat, comments, episodes, out_dir=MODEL_DIR, task=task,
                      max_episode=25, novels=novels)
            trained += 1
        except ValueError as exc:
            # 작품 수가 적으면 작품 간 비교는 애초에 성립하지 않는다. 정상이다.
            print("  [%s] 건너뜀 — %s" % (task, exc))
    return trained > 0


# ------------------------------------------------------------------- 진입점
def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="  ! %(levelname)s | %(message)s")

    title("문피아 웹소설 데이터 수집기")
    print("  독자 이탈률 · 고정 팬층 분석용 데이터를 모읍니다.")
    print("  질문에 답하기만 하면 됩니다. 엔터를 누르면 [기본값]이 선택됩니다.")

    cookie_path = os.path.join("data", "cookies.json")
    os.makedirs("data", exist_ok=True)

    try:
        client, logged_in = do_login(cookie_path)
        scope = explain_scope(logged_in)

        crawler = MunpiaCrawler(client, comment_scope=scope)
        novel_ids = choose_targets(crawler)
        if not novel_ids:
            print("\n  수집할 작품이 없어 종료합니다.")
            return 1

        scope, max_episodes, detail = choose_options(scope)
        crawler.comment_scope = scope
        crawler.collect_comments = scope != "none"
        crawler.max_episodes = max_episodes
        crawler.fetch_entry_detail = detail

        summary = run_crawl(crawler, novel_ids)
        made_features = build_features()
        made_sentiment = run_sentiment() if made_features else False
        trained = train_models() if made_features else False

        title("완료")
        print("  작품 %d건 · 회차 %d건 · 댓글 %d건을 수집했습니다."
              % (summary["novels"], summary["episodes"], summary["comments"]))
        print("\n  원본 데이터   : %s" % os.path.abspath(RAW_DIR))
        print("    novels.csv     작품 정보")
        print("    episodes.csv   회차별 조회수·추천수·댓글수")
        print("    comments.csv   댓글 본문 (감정 분석용)")
        if made_features:
            print("\n  학습용 피처   : %s" % os.path.abspath(FEATURE_DIR))
            print("    episode_features.csv  이탈률 지표")
            print("    user_features.csv     독자별 충성도")
            if made_sentiment:
                print("    episode_sentiment.csv 회차별 댓글 감정 (KOTE)")
        if trained:
            print("\n  학습 결과     : %s" % os.path.abspath(MODEL_DIR))
            print("    *_factor_rank.csv     이탈 요인 순위")
            print("    *_cv_scores.csv       모델별 성능")
            print("    *_predictions.csv     위험 회차·독자 랭킹")
        print("\n  자세한 설명은 README.md 를 참고하세요.")
        return 0

    except KeyboardInterrupt:
        print("\n\n  중단했습니다.")
        return 130
    except Exception as exc:
        print("\n  예기치 못한 오류가 발생했습니다: %s" % exc)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
