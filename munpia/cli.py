# -*- coding: utf-8 -*-
"""명령줄 진입점.

    python -m munpia.cli crawl --novel-ids 587273 --out data/raw
    python -m munpia.cli discover --limit 50 --out data/raw
    python -m munpia.cli features --in data/raw --out data/features
    python -m munpia.cli train --features data/features --raw data/raw
    python -m munpia.cli login
    python -m munpia.cli check-login --novel-id 479065
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from typing import List

from .client import MunpiaClient
from .crawler import MunpiaCrawler
from .storage import DatasetWriter


def setup_logging(out_dir: str, verbose: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(out_dir, "crawl.log"), encoding="utf-8")]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_novel_ids(values: List[str]) -> List[int]:
    """숫자 ID와 작품 URL을 모두 받아 ID 리스트로 만든다."""
    ids: List[int] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        m = re.search(r"(\d{2,})", value)
        if m:
            ids.append(int(m.group(1)))
        else:
            logging.warning("작품 ID를 인식하지 못했습니다: %s", value)
    return ids


def load_ids_from_file(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_novel_ids(f.read().splitlines())


def _build_crawler(args) -> MunpiaCrawler:
    client = MunpiaClient(
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        max_retries=args.retries,
        cookie_file=args.cookies,
    )
    return MunpiaCrawler(
        client,
        comment_scope=args.comment_scope,
        max_comment_pages=args.max_comment_pages,
        fetch_entry_detail=args.entry_detail,
        max_episodes=args.max_episodes,
    )


def cmd_crawl(args) -> int:
    setup_logging(args.out, args.verbose)
    log = logging.getLogger("munpia.cli")

    ids: List[int] = []
    if args.novel_ids:
        ids += parse_novel_ids(args.novel_ids)
    if args.id_file:
        ids += load_ids_from_file(args.id_file)
    if not ids:
        log.error("수집할 작품 ID가 없습니다 (--novel-ids 또는 --id-file)")
        return 2

    seen, unique = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)

    crawler = _build_crawler(args)
    log.info("작품 %d개 수집 시작 (댓글 범위=%s)", len(unique), args.comment_scope)

    with DatasetWriter(args.out, fmt=args.format, resume=not args.no_resume) as writer:
        targets = [i for i in unique if not writer.is_done(i)]
        skipped = len(unique) - len(targets)
        if skipped:
            log.info("이미 수집된 %d개 건너뜀", skipped)

        for result in crawler.crawl_many(targets):
            writer.write_result(result)

        summary = writer.summary()
    log.info("수집 완료 — 작품 %d, 회차 %d, 댓글 %d (누적 완료 %d)",
             summary["novels"], summary["episodes"],
             summary["comments"], summary["completed_total"])
    return 0


def cmd_discover(args) -> int:
    setup_logging(args.out, args.verbose)
    log = logging.getLogger("munpia.cli")

    crawler = _build_crawler(args)
    ids = crawler.discover_novel_ids(
        genres=args.genres.split(",") if args.genres else None,
        include_paid=not args.free_only,
    )
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        log.error("작품을 찾지 못했습니다")
        return 1

    os.makedirs(args.out, exist_ok=True)
    id_path = os.path.join(args.out, "novel_ids.txt")
    with open(id_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(i) for i in ids))
    log.info("작품 ID %d개 저장: %s", len(ids), id_path)

    if args.crawl:
        with DatasetWriter(args.out, fmt=args.format, resume=not args.no_resume) as writer:
            targets = [i for i in ids if not writer.is_done(i)]
            for result in crawler.crawl_many(targets):
                writer.write_result(result)
            summary = writer.summary()
        log.info("수집 완료 — 작품 %d, 회차 %d, 댓글 %d",
                 summary["novels"], summary["episodes"], summary["comments"])
    return 0


def cmd_features(args) -> int:
    setup_logging(args.out, args.verbose)
    log = logging.getLogger("munpia.cli")
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas가 필요합니다: pip install pandas")
        return 2
    from .features import build_dataset

    def read(name: str) -> "pd.DataFrame":
        for ext, reader in (("csv", pd.read_csv), ("jsonl", pd.read_json)):
            path = os.path.join(args.inp, "%s.%s" % (name, ext))
            if os.path.exists(path):
                if ext == "jsonl":
                    return pd.read_json(path, lines=True)
                return pd.read_csv(path)
        log.warning("%s 테이블을 찾지 못했습니다", name)
        return pd.DataFrame()

    novels, episodes, comments = read("novels"), read("episodes"), read("comments")
    if episodes.empty:
        log.error("episodes 데이터가 없습니다")
        return 1

    ep_feat, users = build_dataset(novels, episodes, comments)
    os.makedirs(args.out, exist_ok=True)

    ep_path = os.path.join(args.out, "episode_features.csv")
    ep_feat.to_csv(ep_path, index=False, encoding="utf-8-sig")
    log.info("회차 피처 %d행 저장: %s", len(ep_feat), ep_path)

    if not users.empty:
        user_path = os.path.join(args.out, "user_features.csv")
        users.to_csv(user_path, index=False, encoding="utf-8-sig")
        log.info("유저 피처 %d행 저장: %s (고정팬 %d명)",
                 len(users), user_path, int(users["is_core_fan"].sum()))
    return 0


def cmd_sentiment(args) -> int:
    setup_logging(args.out, args.verbose)
    log = logging.getLogger("munpia.cli")
    try:
        import pandas as pd
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        log.error("KOTE 감정 분석에는 torch와 transformers가 필요합니다: "
                  "pip install torch transformers (%s)", exc)
        return 2
    from .sentiment import run as run_sentiment

    def read(name: str) -> "pd.DataFrame":
        path = os.path.join(args.raw, "%s.csv" % name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    comments, novels = read("comments"), read("novels")
    if comments.empty:
        log.error("%s 에 comments.csv 가 없습니다", args.raw)
        return 1

    run_sentiment(comments, novels, out_dir=args.out, model_name=args.model,
                  device=args.device, batch_size=args.batch_size,
                  min_comments=args.min_comments)
    return 0


def cmd_train(args) -> int:
    setup_logging(args.out, args.verbose)
    log = logging.getLogger("munpia.cli")
    try:
        import pandas as pd
        import sklearn  # noqa: F401
    except ImportError as exc:
        log.error("scikit-learn과 pandas가 필요합니다: pip install scikit-learn pandas (%s)",
                  exc)
        return 2
    from .model import run as run_model

    def read(path: str) -> "pd.DataFrame":
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    ep_feat = read(os.path.join(args.features, "episode_features.csv"))
    episodes = read(os.path.join(args.raw, "episodes.csv"))
    comments = read(os.path.join(args.raw, "comments.csv"))
    novels = read(os.path.join(args.raw, "novels.csv"))

    if ep_feat.empty:
        log.error("%s 에 episode_features.csv 가 없습니다. 먼저 `features` 명령을 실행하세요",
                  args.features)
        return 1

    # 감정 피처가 있으면 붙인다. 없으면 그냥 없는 채로 돌아간다 —
    # KOTE는 선택 단계이고, 이것 때문에 학습이 막히면 안 된다.
    sent = read(os.path.join(args.features, "episode_sentiment.csv"))
    if sent.empty:
        log.info("episode_sentiment.csv 가 없어 감정 피처 없이 학습합니다 "
                 "(`sentiment` 명령으로 생성)")
    else:
        ep_feat = ep_feat.merge(sent, on=["novel_id", "episode_num"], how="left")
        covered = sent["sent_churn_index"].notna().sum() if "sent_churn_index" in sent else 0
        log.info("감정 피처 결합: 회차 %d행 중 유효 %d행 (%.1f%%)",
                 len(ep_feat), covered, 100.0 * covered / max(len(ep_feat), 1))

    tasks = (["episode", "user", "novel"] if args.task == "all"
             else ["episode", "user"] if args.task == "both" else [args.task])
    failed = 0
    for task in tasks:
        if task == "user" and comments.empty:
            log.warning("comments.csv 가 없어 독자 단위 학습을 건너뜁니다")
            continue
        try:
            run_model(ep_feat, comments, episodes, out_dir=args.out, task=task,
                      model=args.model, label_mode=args.label_mode,
                      threshold=args.threshold, n_splits=args.folds,
                      class_weight=(None if args.class_weight == "none"
                                    else args.class_weight),
                      min_episode=args.min_episode, max_episode=args.max_episode,
                      fixed_effect=args.fixed_effect, novels=novels,
                      save_model=not args.no_save_model)
        except ValueError as exc:
            # 한 태스크가 데이터 부족으로 실패해도 다른 태스크는 끝까지 돌린다
            log.error("[%s] 학습 실패: %s", task, exc)
            failed += 1
    return 1 if failed == len(tasks) else 0


def cmd_login(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from .auth import login
    return 0 if login(args.cookies, args.env, args.browser) else 1


def cmd_check_login(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from .auth import check_login
    check_login(args.cookies, args.novel_id, args.entry_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="munpia", description="문피아 웹소설 데이터 수집/전처리 파이프라인")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--out", default="data/raw", help="출력 디렉터리")
        sp.add_argument("--format", choices=["csv", "jsonl"], default="csv")
        sp.add_argument("--min-delay", type=float, default=0.8, help="요청 간 최소 딜레이(초)")
        sp.add_argument("--max-delay", type=float, default=1.8, help="요청 간 최대 딜레이(초)")
        sp.add_argument("--retries", type=int, default=3)
        sp.add_argument("--cookies", default=None, help="로그인 쿠키 JSON 경로")
        sp.add_argument("--comment-scope", choices=["free", "all", "none"],
                        default="free",
                        help="댓글 수집 범위 (free=무료회차만, all=전체[로그인 필요])")
        sp.add_argument("--max-comment-pages", type=int, default=50)
        sp.add_argument("--max-episodes", type=int, default=None,
                        help="작품당 수집할 최대 회차 수 (테스트용)")
        sp.add_argument("--entry-detail", action="store_true",
                        help="회차 상세를 추가 호출해 반응 버튼 지표까지 수집 (요청 2배)")
        sp.add_argument("--no-resume", action="store_true", help="완료 목록을 무시하고 재수집")
        sp.add_argument("--verbose", "-v", action="store_true")

    sp = sub.add_parser("crawl", help="지정한 작품을 수집")
    sp.add_argument("--novel-ids", nargs="*", default=[], help="작품 ID 또는 URL")
    sp.add_argument("--id-file", default=None, help="작품 ID 목록 파일")
    add_common(sp)
    sp.set_defaults(func=cmd_crawl)

    sp = sub.add_parser("discover", help="크롤 대상 작품 ID를 탐색")
    sp.add_argument("--genres", default=None, help="쉼표 구분 장르코드 (예: FANTASY,HEROISM)")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--free-only", action="store_true", help="무료연재만")
    sp.add_argument("--crawl", action="store_true", help="탐색 후 바로 수집")
    add_common(sp)
    sp.set_defaults(func=cmd_discover)

    sp = sub.add_parser("features", help="원본 테이블에서 학습용 피처 생성")
    sp.add_argument("--in", dest="inp", default="data/raw")
    sp.add_argument("--out", default="data/features")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("sentiment", help="댓글 감정 분석 (KOTE) → 회차별 감정 피처")
    sp.add_argument("--raw", default="data/raw", help="comments.csv 가 있는 디렉터리")
    sp.add_argument("--out", default="data/features")
    sp.add_argument("--model", default="searle-j/kote_for_easygoing_people")
    sp.add_argument("--device", type=int, default=-1, help="-1=CPU, 0 이상=GPU 번호")
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--min-comments", type=int, default=3,
                    help="이 수 미만 댓글로 계산된 회차 감정은 NaN 처리")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(func=cmd_sentiment)

    sp = sub.add_parser("train", help="이탈 예측 모델 학습·교차검증 (scikit-learn)")
    sp.add_argument("--features", default="data/features",
                    help="episode_features.csv 가 있는 디렉터리")
    sp.add_argument("--raw", default="data/raw",
                    help="comments.csv / episodes.csv 가 있는 디렉터리")
    sp.add_argument("--out", default="data/models")
    sp.add_argument("--task", choices=["episode", "user", "novel", "both", "all"],
                    default="both",
                    help="episode=회차 이탈, user=독자 이탈, novel=작품 간 비교, all=전부")
    sp.add_argument("--model", default="logistic",
                    choices=["logistic", "logistic_l1", "random_forest",
                             "hist_gbm", "dummy"],
                    help="최종 저장할 모델 (교차검증은 항상 전부 비교)")
    sp.add_argument("--label-mode",
                    choices=["quantile", "within_novel", "absolute"],
                    default="quantile",
                    help="within_novel=작품 내 분위수. 요인 순위를 볼 때 권장")
    sp.add_argument("--min-episode", type=int, default=2,
                    help="이 회차 번호 미만 제외 (1화는 직전 회차가 없어 라벨 없음)")
    sp.add_argument("--max-episode", type=int, default=None,
                    help="이 회차 번호 초과 제외. 25면 무료 구간만 (권장)")
    sp.add_argument("--fixed-effect", action="store_true",
                    help="작품 고정효과. 작품 간 차이를 흡수해 회차 변동만 본다. "
                         "켜면 교차검증이 회차 단위 층화 분할로 바뀐다")
    sp.add_argument("--threshold", type=float, default=0.75,
                    help="quantile이면 분위수(0~1), absolute면 이탈률 임계값")
    sp.add_argument("--folds", type=int, default=5, help="GroupKFold 폴드 수")
    sp.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="balanced는 확률 보정을 깨뜨린다. 순위만 쓸 때만 사용")
    sp.add_argument("--no-save-model", action="store_true",
                    help=".joblib 저장 생략")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("login", help="로그인해서 세션 쿠키 저장")
    sp.add_argument("--cookies", default="data/cookies.json")
    sp.add_argument("--env", default=".env",
                    help="MUNPIA_ID / MUNPIA_PW 가 든 .env 경로")
    sp.add_argument("--browser", action="store_true",
                    help=".env를 무시하고 브라우저에서 직접 로그인")
    sp.set_defaults(func=cmd_login)

    sp = sub.add_parser("check-login", help="쿠키로 유료 회차 댓글 접근 가능한지 확인")
    sp.add_argument("--cookies", default="data/cookies.json")
    sp.add_argument("--novel-id", type=int, default=None)
    sp.add_argument("--entry-id", type=int, default=None)
    sp.set_defaults(func=cmd_check_login)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
