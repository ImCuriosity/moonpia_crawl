# -*- coding: utf-8 -*-
"""수집 결과 요약 — 데이터가 쓸 만한지 먼저 눈으로 확인하는 용도.

    python scripts/explore.py --in data/raw --features data/features
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def load(directory: str, name: str) -> pd.DataFrame:
    path = os.path.join(directory, "%s.csv" % name)
    if not os.path.exists(path):
        print("없음: %s" % path)
        return pd.DataFrame()
    return pd.read_csv(path)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw")
    ap.add_argument("--features", default="data/features")
    args = ap.parse_args()

    novels = load(args.inp, "novels")
    episodes = load(args.inp, "episodes")
    comments = load(args.inp, "comments")

    section("수집 규모")
    print("작품 %d개 / 회차 %d건 / 댓글 %d건"
          % (len(novels), len(episodes), len(comments)))
    if not comments.empty:
        print("고유 댓글 작성자: %d명" % comments["user_key"].nunique())
        print("텍스트 댓글: %d건 (스티커 %d건)"
              % ((comments.content_type == "TEXT").sum(),
                 (comments.content_type == "STICKER").sum()))

    if not episodes.empty:
        section("회차 댓글 수집 상태")
        print(episodes["comment_status"].value_counts().to_string())
        got = episodes[episodes.comment_status == "ok"]
        if not got.empty:
            # 목록의 commentCount와 실제 수집량이 어긋나면 페이지네이션을 의심해야 한다
            diff = (got.comment_count - got.comment_collected)
            print("\n목록 댓글수 vs 실제 수집량 차이: 평균 %.2f, 최대 %d"
                  % (diff.mean(), diff.max()))

    if not novels.empty:
        section("작품 목록")
        cols = ["novel_id", "title", "author_name", "genre_main", "serial_status",
                "chapter_count", "total_view_count", "preference_count"]
        print(novels[[c for c in cols if c in novels.columns]]
              .to_string(index=False, max_colwidth=30))

    ep_feat = load(args.features, "episode_features")
    if not ep_feat.empty:
        section("이탈률 노이즈 제거 단계별 평균")
        for label, col in (("원본            ", "churn_step"),
                           ("페이월 경계 제외 ", "churn_step_ex_paywall"),
                           ("+ 신선회차 제외  ", "churn_step_clean")):
            if col in ep_feat.columns:
                print("%s %.4f  (표본 %d)"
                      % (label, ep_feat[col].mean(), ep_feat[col].notna().sum()))

        if "is_paywall_boundary" in ep_feat.columns:
            section("무료→유료 전환 경계 (이탈이 아니라 결제 장벽)")
            b = ep_feat[ep_feat.is_paywall_boundary == 1]
            cols = ["novel_id", "episode_num", "view_count", "churn_step",
                    "last_free_episode"]
            print(b[[c for c in cols if c in b.columns]].to_string(index=False))

        churn_col = ("churn_step_clean" if "churn_step_clean" in ep_feat.columns
                     else "churn_step")
        section("실질 이탈률 상위 회차 (%s)" % churn_col)
        cols = ["novel_id", "episode_num", "title", "view_count",
                churn_col, "days_since_prev", "age_days"]
        top = ep_feat.dropna(subset=[churn_col]).nlargest(12, churn_col)
        print(top[[c for c in cols if c in top.columns]]
              .to_string(index=False, max_colwidth=26))

        section("작품별 평균 잔존율 / 고정팬 비율")
        agg = {"retention_step": "mean", churn_col: "mean"}
        if "returning_commenter_ratio" in ep_feat.columns:
            agg["returning_commenter_ratio"] = "mean"
        print(ep_feat.groupby("novel_id").agg(agg).round(4).to_string())

    users = load(args.features, "user_features")
    if not users.empty:
        section("고정 팬 상위 20명")
        cols = ["novel_id", "nickname", "comment_count", "episodes_commented",
                "max_consecutive_episodes", "engagement_density",
                "loyalty_ratio", "is_core_fan"]
        print(users.nlargest(20, "comment_count")[[c for c in cols if c in users.columns]]
              .to_string(index=False, max_colwidth=20))
        print("\n전체 %d명 중 고정팬(is_core_fan=1) %d명 (%.1f%%)"
              % (len(users), int(users.is_core_fan.sum()),
                 100.0 * users.is_core_fan.mean()))

    if not comments.empty:
        section("감정 분석 입력 샘플 (TEXT, 길이>0)")
        texts = comments[(comments.content_type == "TEXT")
                         & (comments.body_char_len > 0)]
        print("학습 가능 텍스트 %d건, 평균 길이 %.1f자"
              % (len(texts), texts.body_char_len.mean()))
        print()
        print(texts[["episode_num", "nickname", "body", "like_count"]]
              .head(15).to_string(index=False, max_colwidth=48))

    return 0


if __name__ == "__main__":
    sys.exit(main())
