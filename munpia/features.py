# -*- coding: utf-8 -*-
"""수집한 원본 테이블에서 학습용 파생 지표를 만든다.

목표 두 가지를 그대로 지표로 옮긴 모듈이다.
  - 이탈률 추정  → 회차 간 조회수 감소율, 첫 회차 대비 잔존율
  - 고정 팬층    → 유저별 댓글 연속성/유지율, 회차별 재방문 댓글러 비율
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------- 회차 시계열
def build_episode_features(episodes: pd.DataFrame) -> pd.DataFrame:
    """회차 테이블에 이탈률 관련 파생 컬럼을 추가한다.

    조회수는 누적값이라 회차가 뒤로 갈수록 자연 감소한다. 그 감소폭이
    독자 이탈의 대리 지표(proxy)다.
    """
    if episodes.empty:
        return episodes.copy()

    df = episodes.copy()
    df = df.sort_values(["novel_id", "episode_num"]).reset_index(drop=True)
    g = df.groupby("novel_id", sort=False)

    df["episode_index"] = g.cumcount()                       # 0부터 시작하는 회차 순번
    df["first_view_count"] = g["view_count"].transform("first")
    df["prev_view_count"] = g["view_count"].shift(1)

    # 첫 회차 대비 잔존율 — 작품 간 비교가 가능하도록 정규화한 값
    df["retention_from_first"] = _safe_div(df["view_count"], df["first_view_count"])
    # 직전 회차 대비 잔존율. 1 - 이 값이 해당 회차의 이탈률이다.
    df["retention_step"] = _safe_div(df["view_count"], df["prev_view_count"])
    df["churn_step"] = (1.0 - df["retention_step"]).clip(lower=-1.0, upper=1.0)
    df.loc[df["prev_view_count"].isna(), ["retention_step", "churn_step"]] = np.nan

    # 참여 강도 — 조회 대비 얼마나 반응했는가
    df["like_per_view"] = _safe_div(df["like_count"], df["view_count"])
    df["comment_per_view"] = _safe_div(df["comment_count"], df["view_count"])
    df["reaction_per_view"] = _safe_div(df.get("reaction_total", 0), df["view_count"])

    # 연재 간격 — 공백이 길수록 이탈이 커진다는 가설을 검증할 수 있게 남긴다
    ts = pd.to_numeric(df["published_ts"], errors="coerce")
    df["days_since_prev"] = (ts - g["published_ts"].shift(1)) / 86400.0
    df["days_since_first"] = (ts - g["published_ts"].transform("first")) / 86400.0

    # 이동평균 대비 편차 — 특정 회차에서 튀는 이탈을 잡아낸다
    df["view_ma5"] = g["view_count"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    df["view_vs_ma5"] = _safe_div(df["view_count"], df["view_ma5"])

    df = _mark_paywall(df)
    df = _mark_maturity(df)
    return df


def _to_epoch_seconds(values) -> pd.Series:
    """datetime 문자열을 유닉스 초로 바꾼다. 파싱 실패는 NaN.

    `astype("int64") // 10**9` 을 쓰면 안 된다. pandas 2.0부터 to_datetime이 입력
    정밀도에 맞춰 datetime64[s]/[us]/[ns] 중 하나를 돌려주는데, 우리 collected_at은
    초 단위 문자열이라 datetime64[s]로 파싱된다. 그 상태에서 10**9로 나누면 값이
    10억분의 1로 뭉개져 age_days가 -19000일 같은 값이 되고 is_mature가 전부 0이 된다.
    Timedelta로 나누면 해상도와 무관하게 항상 초가 나온다.
    """
    ts = pd.to_datetime(values, errors="coerce")
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    return (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)


def _mark_maturity(df: pd.DataFrame, mature_days: float = 7.0) -> pd.DataFrame:
    """회차가 '조회수가 충분히 쌓인' 상태인지 표시한다.

    조회수는 누적값이라 방금 올라온 회차는 필연적으로 낮다. 그대로 두면 모든 작품의
    최신 회차가 이탈률 최상위를 차지해 신호를 덮어버린다. 수집 시점 기준 경과일을
    계산해 학습 전에 걸러낼 수 있게 한다.
    """
    # collected_at이 없거나 비었으면 그 작품에서 관측된 최신 게시시각을 기준으로 삼는다
    fallback = pd.to_numeric(
        df.groupby("novel_id")["published_ts"].transform("max"), errors="coerce")
    if "collected_at" in df.columns:
        collected_epoch = _to_epoch_seconds(df["collected_at"]).fillna(fallback)
    else:
        collected_epoch = fallback

    df["age_days"] = (collected_epoch - pd.to_numeric(df["published_ts"],
                                                      errors="coerce")) / 86400.0
    df["is_mature"] = (df["age_days"] >= mature_days).astype(int)

    # 페이월과 신선도를 모두 걸러낸 이탈률 — 순수 콘텐츠 이탈 신호
    df["churn_step_clean"] = df["churn_step_ex_paywall"].where(df["is_mature"] == 1, np.nan)
    return df


def _mark_paywall(df: pd.DataFrame) -> pd.DataFrame:
    """무료 → 유료 전환 경계를 표시한다.

    문피아 유료 연재는 앞 20~25화만 무료다. 그 경계에서 조회수가 급락하는데,
    이것은 '작품이 재미없어서 떠난 이탈'이 아니라 '결제 장벽'이다.
    두 신호를 섞어서 학습시키면 모델이 페이월 위치를 이탈 원인으로 오학습한다.
    구분해서 쓸 수 있도록 플래그와 상대 위치를 남긴다.

    경계가 실제로 존재하는 작품에만 적용한다. 수집 범위가 전부 무료(무료연재)거나
    전부 유료면 걸러낼 경계 자체가 없으므로 원본 이탈률을 그대로 통과시킨다.
    이 구분이 없으면 무료연재는 마지막 두 회차가, 전유료작은 전 회차가 근거 없이
    NaN 처리되어 학습 데이터에서 통째로 사라진다.
    """
    is_free = pd.to_numeric(df["is_free"], errors="coerce").fillna(0)
    g = df.groupby("novel_id", sort=False)
    prev_free = g["is_free"].shift(1)

    # 직전 회차는 무료였는데 이번 회차가 유료 = 페이월 첫 회차
    df["is_paywall_boundary"] = (
        (prev_free == 1) & (df["is_free"] == 0)
    ).astype(int)

    # 무료 회차와 유료 회차가 둘 다 있어야 경계가 존재한다
    by_novel = is_free.groupby(df["novel_id"], sort=False)
    has_paywall = (by_novel.transform("max") == 1) & (by_novel.transform("min") == 0)
    df["has_paywall"] = has_paywall.astype(int)

    # 작품별 마지막 무료 회차 번호 — 경계까지의 거리를 재는 기준점
    free_max = (df[is_free == 1]
                .groupby("novel_id")["episode_num"].max()
                .rename("last_free_episode"))
    df = df.merge(free_max, on="novel_id", how="left")
    df["last_free_episode"] = df["last_free_episode"].where(has_paywall.to_numpy(), np.nan)
    df["episodes_from_paywall"] = df["episode_num"] - df["last_free_episode"]

    # 페이월 경계 ±1화를 제외한 이탈률. 순수 콘텐츠 이탈만 보고 싶을 때 쓴다.
    # 경계가 없는 작품은 제외할 것이 없으므로 원본을 그대로 쓴다.
    near_boundary = df["episodes_from_paywall"].abs() <= 1
    df["churn_step_ex_paywall"] = df["churn_step"].where(~near_boundary, np.nan)
    return df


def _safe_div(num, den) -> pd.Series:
    """0 나누기를 NaN으로 흘려보내는 나눗셈.

    분모가 Series인 것을 기준으로 길이를 맞춘다. 분자가 스칼라(컬럼 부재 시 기본값)로
    들어와도 브로드캐스트되도록 해서, 호출부가 컬럼 존재 여부를 신경 쓰지 않아도 되게 한다.
    """
    den = pd.to_numeric(pd.Series(den), errors="coerce")
    if np.isscalar(num):
        num = pd.Series(num, index=den.index, dtype="float64")
    else:
        num = pd.to_numeric(pd.Series(num), errors="coerce")
        num.index = den.index  # 정렬된 동일 길이 전제 — 인덱스만 맞춰준다
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den.replace(0, np.nan)


def _as_int(value, default: int = 0) -> int:
    """NaN/None을 default로 흘려보내는 정수 변환."""
    if value is None or (isinstance(value, float) and value != value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ 팬층 분석
def build_user_features(comments: pd.DataFrame,
                        episodes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """댓글 작성자별 충성도 지표.

    문피아 blogUrl 기반 user_key로 동일인을 추적한다 (닉네임은 변경 가능).
    """
    if comments.empty:
        return pd.DataFrame()

    df = comments[comments["user_key"].astype(str) != ""].copy()
    if df.empty:
        return pd.DataFrame()

    df["episode_num"] = pd.to_numeric(df["episode_num"], errors="coerce")
    df = df.sort_values(["novel_id", "user_key", "episode_num"])

    rows = []
    for (novel_id, user_key), grp in df.groupby(["novel_id", "user_key"], sort=False):
        eps = grp["episode_num"].dropna().astype(int).unique()
        eps.sort()
        ts = pd.to_numeric(grp["created_ts"], errors="coerce")
        rows.append({
            "novel_id": novel_id,
            "user_key": user_key,
            "nickname": grp["nickname"].iloc[-1],
            "comment_count": len(grp),
            "episodes_commented": len(eps),
            "first_episode": int(eps[0]) if len(eps) else 0,
            "last_episode": int(eps[-1]) if len(eps) else 0,
            "episode_span": int(eps[-1] - eps[0] + 1) if len(eps) else 0,
            "max_consecutive_episodes": _max_consecutive(eps),
            "reply_ratio": float(grp["is_reply"].mean()),
            "avg_like_received": float(pd.to_numeric(grp["like_count"],
                                                     errors="coerce").fillna(0).mean()),
            "avg_body_len": float(pd.to_numeric(grp["body_char_len"],
                                                errors="coerce").fillna(0).mean()),
            # min()/max()가 NaN이면 `nan or 0`은 NaN(truthy)이라 int(NaN)에서 죽는다.
            # 댓글 시각이 통째로 결측인 작품 하나 때문에 전체 실행이 멈추면 안 된다.
            "first_ts": _as_int(ts.min()),
            "last_ts": _as_int(ts.max()),
        })

    users = pd.DataFrame(rows)
    users["active_days"] = (users["last_ts"] - users["first_ts"]) / 86400.0

    # 참여 밀도: 자기가 걸쳐 있는 구간 중 실제로 댓글을 단 회차 비율.
    # 1에 가까울수록 매 회차 따라오는 고정 독자다.
    users["engagement_density"] = _safe_div(users["episodes_commented"],
                                            users["episode_span"]).fillna(0.0)

    if episodes is not None and not episodes.empty:
        totals = (episodes.groupby("novel_id")["episode_num"]
                  .max().rename("novel_total_episodes"))
        users = users.merge(totals, on="novel_id", how="left")
        users["loyalty_ratio"] = _safe_div(users["episodes_commented"],
                                           users["novel_total_episodes"]).fillna(0.0)

    # 고정 팬 라벨: 3회차 이상 참여 + 연속 2회 이상. 학습 시 시작 기준선으로 쓰라는 값이다.
    users["is_core_fan"] = (
        (users["episodes_commented"] >= 3) & (users["max_consecutive_episodes"] >= 2)
    ).astype(int)

    return users.sort_values(["novel_id", "comment_count"], ascending=[True, False])


def _max_consecutive(sorted_eps: np.ndarray) -> int:
    """연속된 회차 번호의 최대 길이 (유저별 연속 작성 여부 추적)."""
    if len(sorted_eps) == 0:
        return 0
    best = run = 1
    for i in range(1, len(sorted_eps)):
        run = run + 1 if sorted_eps[i] == sorted_eps[i - 1] + 1 else 1
        best = max(best, run)
    return best


def build_episode_audience_features(comments: pd.DataFrame) -> pd.DataFrame:
    """회차별 댓글러 구성 — 신규 유입 대 재방문(고정팬) 비율."""
    if comments.empty:
        return pd.DataFrame()

    df = comments[comments["user_key"].astype(str) != ""].copy()
    if df.empty:
        return pd.DataFrame()
    df["episode_num"] = pd.to_numeric(df["episode_num"], errors="coerce")
    df = df.dropna(subset=["episode_num"])

    out = []
    for novel_id, novel_grp in df.groupby("novel_id", sort=False):
        seen: set = set()
        prev_users: set = set()
        for ep_num, ep_grp in sorted(novel_grp.groupby("episode_num"),
                                     key=lambda kv: kv[0]):
            users = set(ep_grp["user_key"])
            new_users = users - seen
            returning = users & prev_users
            out.append({
                "novel_id": novel_id,
                "episode_num": int(ep_num),
                "commenter_count": len(users),
                "new_commenter_count": len(new_users),
                "returning_commenter_count": len(returning),
                # 직전 회차에도 댓글을 단 사람의 비율 = 고정 팬층 두께
                "returning_commenter_ratio": len(returning) / len(users) if users else 0.0,
                "new_commenter_ratio": len(new_users) / len(users) if users else 0.0,
                "avg_comment_like": float(pd.to_numeric(
                    ep_grp["like_count"], errors="coerce").fillna(0).mean()),
                "reply_ratio": float(pd.to_numeric(
                    ep_grp["is_reply"], errors="coerce").fillna(0).mean()),
            })
            seen |= users
            prev_users = users
    return pd.DataFrame(out)


def build_dataset(novels: pd.DataFrame, episodes: pd.DataFrame,
                  comments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """학습에 바로 넣을 수 있는 회차 단위 테이블과 유저 단위 테이블을 만든다."""
    ep_feat = build_episode_features(episodes)

    if not comments.empty:
        audience = build_episode_audience_features(comments)
        if not audience.empty:
            ep_feat = ep_feat.merge(audience, on=["novel_id", "episode_num"], how="left")
    users = build_user_features(comments, episodes)

    if not novels.empty:
        novel_cols = [c for c in ("novel_id", "genre_main", "status_code", "group_name",
                                  "is_free", "is_adult", "preference_count",
                                  "total_view_count", "total_like_count")
                      if c in novels.columns]
        ep_feat = ep_feat.merge(novels[novel_cols], on="novel_id",
                                how="left", suffixes=("", "_novel"))

    return ep_feat, users
