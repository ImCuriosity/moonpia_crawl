# -*- coding: utf-8 -*-
"""이탈 추론 — scikit-learn 모델 학습·평가·예측.

`features.py`가 만드는 `churn_step_clean`은 **계산된 관측값**이지 추론이 아니다.
조회수 두 개를 나눈 산술 결과라, 이미 일어난 이탈을 사후에 기술할 뿐
"이 회차는 왜 떨어졌나", "다음 회차는 위험한가"에 답하지 못한다.
이 모듈이 그 위를 덮는 추론 레이어다.

두 가지 이탈을 서로 다른 문제로 푼다.

  episode  — 회차 단위. "이 회차에서 큰 이탈이 날 것인가" 이진 분류.
             라벨은 churn_step_clean, 피처는 **직전 회차까지의 정보 + 이번 회차의
             정적 속성(분량·무료여부·연재간격)** 만 쓴다. 사전 예측 문제다.

  user     — 독자 단위. "이 독자가 이 회차를 끝으로 떠나는가" 이진 분류.
             조회수 기반이 아니라 개인의 실제 재등장 여부라서, 이쪽이 이탈의
             정의로는 더 곧다. 대신 댓글을 쓴 독자만 관측된다는 한계가 있다.

## 누수(leakage)에 대해

이 모듈에서 가장 신경 쓴 부분이다. `episode_features.csv`를 그대로
LogisticRegression에 넣으면 ROC-AUC가 0.99를 넘는데, 전부 가짜다.
라벨 `churn_step_clean = 1 - view_count/prev_view_count` 이므로 피처에
`view_count`나 `retention_step`, `view_vs_ma5`가 남아 있으면 모델이 라벨을
그냥 역산한다. `like_per_view`·`comment_per_view`도 분모가 `view_count`라
같은 경로로 새어든다.

그래서 피처를 블랙리스트가 아니라 **화이트리스트**로 고정했다
(`EPISODE_FEATURES`). 새 컬럼이 생겨도 자동으로 딸려 들어가지 않는다.
검증도 무작위 분할이 아니라 작품 단위 `GroupKFold`를 쓴다 — 같은 작품의
회차가 학습·검증에 나뉘어 들어가면 작품별 조회수 수준을 외워버린다.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================== 피처 화이트리스트
#
# 회차 단위. 라벨(churn_step_clean)이 view_count에서 나오므로, view_count와
# 거기서 파생된 어떤 값도 여기에 들어오면 안 된다.
#
#   금지: view_count / prev_view_count / first_view_count / retention_step /
#         retention_from_first / churn_step* / view_ma5 / view_vs_ma5 /
#         like_per_view / comment_per_view / reaction_per_view
#
# 허용되는 것은 두 종류뿐이다.
#   (1) 이번 회차의 정적 속성 — 게시 시점에 이미 정해져 있는 값
#   (2) 직전 회차까지의 관측 — lag 피처. 과거는 미래를 오염시키지 않는다.
# 완전 공선인 짝은 한쪽만 남긴다. char_estimate = pages × 900, episode_index =
# episode_num - 1 이라 둘 다 넣으면 로지스틱 계수가 반씩 갈려 해석이 망가진다.
EPISODE_STATIC_FEATURES: List[str] = [
    "episode_num",            # 연재 위치. 초반 이탈 구간을 모델이 잡아야 한다
    "days_since_prev",        # 연재 공백 ↔ 이탈 가설의 핵심 변수
    "days_since_first",
    "pages",                  # 분량. "이번 화가 유난히 짧았나"
    "is_free",
    "is_adult",
    "is_notice",
    "is_paywall_boundary",
    "episodes_from_paywall",
    "has_paywall",
]

# 직전 회차에서 끌어오는 lag 피처. 원본 컬럼명 → 생성될 컬럼명.
EPISODE_LAG_SOURCES: List[str] = [
    "churn_step",                 # 직전 회차의 이탈률 — 자기상관이 가장 강한 신호
    "like_per_view",
    "comment_per_view",
    "reaction_per_view",
    "returning_commenter_ratio",  # 고정 팬층 두께
    "new_commenter_ratio",
    "commenter_count",
    "retention_from_first",       # 그 시점까지의 누적 잔존율
]

# 작품 단위 속성. GroupKFold가 작품을 통째로 갈라놓으므로 라벨 누수는 없다.
# 단, 작품 전체 조회/추천 합계는 라벨과 같은 뿌리라서 제외한다.
NOVEL_NUMERIC_FEATURES: List[str] = ["preference_count", "status_code"]
NOVEL_CATEGORICAL_FEATURES: List[str] = ["genre_main"]

# --------------------------------------------------------------- 내용(content) 신호
#
# 본문을 수집하지 않으므로 "이 회차가 어땠나"의 직접 측정치가 없다. 아래 둘이
# 대리 지표다. 둘 다 **이번 회차와 동시에 관측된 값**이라 lag하지 않는다.
#
# 누수가 아닌 이유: 라벨은 view_count의 비율이고, 반응 구성비도 감정 점수도
# view_count에서 산술적으로 유도되지 않는다. 모델이 라벨을 역산할 경로가 없다.
# 다만 동시 관측이므로 **인과가 아니라 연관**이다 — "이탈이 큰 회차에서는 불만
# 댓글이 많다"이지 "불만 댓글이 이탈을 일으킨다"가 아니다. 요인 순위는 후자가
# 아니라 전자를 묻는 것이므로 이 설계가 맞다.
REACTION_FEATURES: List[str] = [
    "reaction_best_pct", "reaction_funny_pct", "reaction_amazing_pct",
    "reaction_cheer_pct", "reaction_impressed_pct", "reaction_entropy",
    # 작품 성향(코미디물은 늘 웃김이 높다)을 걷어낸 회차별 편차
    "d_reaction_best_pct", "d_reaction_funny_pct", "d_reaction_amazing_pct",
    "d_reaction_cheer_pct", "d_reaction_impressed_pct", "d_reaction_entropy",
]

SENTIMENT_FEATURES: List[str] = [
    "sent_boredom", "sent_complaint", "sent_hostility", "sent_enjoyment",
    "sent_anticipation", "sent_attachment", "sent_tension", "sent_confusion",
    "sent_churn_index", "sent_n_comments",
    "d_sent_boredom", "d_sent_complaint", "d_sent_hostility", "d_sent_enjoyment",
    "d_sent_anticipation", "d_sent_attachment", "d_sent_tension",
    "d_sent_confusion", "d_sent_churn_index",
]

# 독자 단위. 전부 "그 시점까지"의 누적값이라 미래를 보지 않는다.
USER_FEATURES: List[str] = [
    "episode_num",
    "novel_progress",              # 작품 내 진행률 0~1
    # episodes_commented_so_far = appearance_index + 1 이므로 한쪽만 쓴다
    "episodes_commented_so_far",   # 지금까지 참여한 회차 수
    "comments_so_far",
    "max_consecutive_so_far",
    "gap_from_prev_appearance",    # 직전 등장 이후 건너뛴 회차 수
    "engagement_density_so_far",
    "mean_body_len_so_far",
    "mean_like_received_so_far",
    "reply_ratio_so_far",
    "cur_body_char_len",           # 이번 댓글 자체의 속성 — 시점상 관측 가능
    "cur_like_count",
    "cur_dislike_count",
    "cur_is_reply",
    "cur_comment_count",           # 이번 회차에 이 독자가 쓴 댓글 수
]

# 라벨 계산에 쓰이는 컬럼 — 실수로 피처에 섞이면 즉시 잡아낸다
_EPISODE_LABEL_SOURCES = {
    "view_count", "prev_view_count", "first_view_count", "retention_step",
    "retention_from_first", "churn_step", "churn_step_ex_paywall",
    "churn_step_clean", "view_ma5", "view_vs_ma5", "like_per_view",
    "comment_per_view", "reaction_per_view", "like_count", "comment_count",
    "reaction_total",
}


@dataclass
class TrainingFrame:
    """모델에 넣기 직전의 상태. X/y/groups와 그 출처를 함께 들고 다닌다."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    numeric: List[str]
    categorical: List[str] = field(default_factory=list)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_desc: str = ""

    def __len__(self) -> int:
        return len(self.X)


# ============================================================ 회차 단위 학습 데이터
def build_episode_training_frame(
    ep_feat: pd.DataFrame,
    label_mode: str = "quantile",
    threshold: float = 0.75,
    min_episode: int = 2,
    max_episode: Optional[int] = None,
    fixed_effect: bool = False,
) -> TrainingFrame:
    """`episode_features.csv`에서 회차 이탈 분류 학습셋을 만든다.

    Args:
        ep_feat: `features` 명령이 만든 회차 피처 테이블.
        label_mode:
            "quantile"     — 전체 회차 이탈률 분포에서 상위 `threshold` 분위수 초과면 1.
            "within_novel" — **작품 내부** 분위수 기준. 작품마다 이탈률 수준이
                             다른 것을 라벨 단계에서 흡수한다. 요인 순위를 볼
                             때는 이쪽이 맞다 — "어느 작품이 이탈이 큰가"가 아니라
                             "한 작품 안에서 어떤 회차가 유독 빠지는가"를 묻게 된다.
            "absolute"     — `churn_step_clean >= threshold` 면 1.
        threshold: 분위수(0~1) 또는 절대 이탈률.
        min_episode: 이 회차 번호 미만 제외. 1화는 직전 회차가 없어 라벨이 없다.
        max_episode: 이 회차 번호 초과 제외. 25로 두면 무료 구간만 본다 —
            유료 구간은 view_count가 세는 대상이 바뀌어 이탈률이 무의미하고,
            댓글도 없어 내용 신호가 통째로 빈다.
        fixed_effect: True면 novel_id를 범주형 피처로 넣는다 (작품 고정효과).
            작품 간 차이를 흡수해 회차 간 변동만 남긴다. **이 경우 GroupKFold를
            쓰면 안 된다** — 검증 폴드의 작품 더미는 학습에서 본 적이 없어 무용지물이
            된다. `cross_validate(group_split=False)`와 짝지어 쓴다.
    """
    if ep_feat.empty:
        raise ValueError("episode_features가 비어 있습니다")
    if "churn_step_clean" not in ep_feat.columns:
        raise ValueError(
            "churn_step_clean 컬럼이 없습니다. 먼저 `features` 명령을 실행하세요")

    df = ep_feat.sort_values(["novel_id", "episode_num"]).reset_index(drop=True)
    df = _add_episode_lags(df)

    ep_num = pd.to_numeric(df["episode_num"], errors="coerce")
    label_raw = pd.to_numeric(df["churn_step_clean"], errors="coerce")
    usable = label_raw.notna() & (ep_num >= min_episode)
    if max_episode is not None:
        usable &= ep_num <= max_episode
    if not usable.any():
        raise ValueError(
            "학습 가능한 회차가 없습니다. churn_step_clean이 전부 비어 있거나 "
            "회차 범위(%s~%s) 안에 남는 행이 없습니다"
            % (min_episode, max_episode if max_episode is not None else "끝"))

    df = df[usable].reset_index(drop=True)
    label_raw = label_raw[usable].reset_index(drop=True)

    if label_mode == "quantile":
        cut = float(label_raw.quantile(threshold))
        y = (label_raw >= cut).astype(int).to_numpy()
        desc = "churn_step_clean >= %.4f (전체 상위 %.0f%%)" % (cut, (1 - threshold) * 100)
    elif label_mode == "within_novel":
        # 작품별로 따로 자른다. 작품 수준 차이가 라벨에서 사라지므로 남는 것은
        # "그 작품 기준으로 이 회차가 유독 빠졌는가" 뿐이다.
        cut = label_raw.groupby(df["novel_id"]).transform(
            lambda s: s.quantile(threshold))
        y = (label_raw >= cut).astype(int).to_numpy()
        desc = "작품 내 상위 %.0f%% 회차 (작품별 분위수)" % ((1 - threshold) * 100)
    elif label_mode == "absolute":
        y = (label_raw >= float(threshold)).astype(int).to_numpy()
        desc = "churn_step_clean >= %.4f (절대 임계)" % threshold
    else:
        raise ValueError(
            "label_mode는 quantile / within_novel / absolute 여야 합니다: %r" % label_mode)

    lag_cols = ["prev_%s" % c for c in EPISODE_LAG_SOURCES if "prev_%s" % c in df.columns]
    numeric = [c for c in (EPISODE_STATIC_FEATURES + lag_cols
                           + REACTION_FEATURES + SENTIMENT_FEATURES
                           + NOVEL_NUMERIC_FEATURES)
               if c in df.columns]
    categorical = [c for c in NOVEL_CATEGORICAL_FEATURES if c in df.columns]

    if fixed_effect:
        df["novel_fe"] = df["novel_id"].astype(str)
        categorical = categorical + ["novel_fe"]

    _assert_no_leakage(numeric + categorical, _EPISODE_LABEL_SOURCES)

    meta_cols = [c for c in ("novel_id", "episode_num", "title", "churn_step_clean",
                             "view_count", "is_free") if c in df.columns]
    return TrainingFrame(
        X=df[numeric + categorical].copy(),
        y=y,
        groups=df["novel_id"].to_numpy(),
        numeric=numeric,
        categorical=categorical,
        meta=df[meta_cols].copy(),
        label_desc=desc,
    )


def _add_episode_lags(df: pd.DataFrame) -> pd.DataFrame:
    """직전 회차 값을 `prev_*` 컬럼으로 당겨온다.

    현재 회차의 참여 지표는 라벨과 같은 시점의 관측이라 쓸 수 없다. 한 칸 밀어
    "직전 회차에 이랬으니 이번 회차가 위험하다"는 형태로만 쓴다.
    """
    df = df.copy()
    g = df.groupby("novel_id", sort=False)
    for col in EPISODE_LAG_SOURCES:
        if col in df.columns:
            df["prev_%s" % col] = g[col].shift(1)
    return df


def _assert_no_leakage(features: Sequence[str], banned: set) -> None:
    """라벨 계산에 쓰인 컬럼이 피처에 섞였는지 확인한다.

    조용히 통과시키면 ROC-AUC 0.99를 보고 모델이 잘 됐다고 착각하게 된다.
    설정 실수를 조기에 크게 터뜨리는 편이 낫다.
    """
    hit = sorted(set(features) & banned)
    if hit:
        raise ValueError(
            "라벨 누수: 라벨 계산에 쓰인 컬럼이 피처에 있습니다 → %s. "
            "이대로 학습하면 모델이 라벨을 역산합니다." % ", ".join(hit))


# ============================================================ 독자 단위 학습 데이터
def build_user_training_frame(
    comments: pd.DataFrame,
    episodes: pd.DataFrame,
    censor_last_n: int = 3,
) -> TrainingFrame:
    """댓글 로그에서 "이 독자가 이 회차를 끝으로 떠나는가" 학습셋을 만든다.

    독자 한 명의 등장 하나가 한 행이다. 라벨은 **그 이후 회차에 다시 나타나는가**
    이고, 나타나지 않으면 이탈(1)이다.

    Args:
        censor_last_n: 작품의 마지막 N개 회차에서의 등장은 버린다.
            아직 다음 회차가 안 나왔을 뿐인데 이탈로 라벨링하면 안 된다
            (우측 절단, right-censoring). 조회수 쪽 `is_mature`와 같은 취지다.
    """
    if comments.empty:
        raise ValueError("comments가 비어 있습니다")

    df = comments.copy()
    df["user_key"] = df["user_key"].astype(str)
    df = df[df["user_key"] != ""]
    df["episode_num"] = pd.to_numeric(df["episode_num"], errors="coerce")
    df = df.dropna(subset=["episode_num"])
    if df.empty:
        raise ValueError("user_key/episode_num이 유효한 댓글이 없습니다")
    df["episode_num"] = df["episode_num"].astype(int)

    for col in ("body_char_len", "like_count", "dislike_count", "is_reply"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)

    # 작품별 관측 한계 — 이 번호를 넘으면 "다음 회차가 없어서" 안 온 것일 수 있다
    if episodes is not None and not episodes.empty:
        last_ep = (pd.to_numeric(episodes["episode_num"], errors="coerce")
                   .groupby(episodes["novel_id"]).max())
    else:
        last_ep = df.groupby("novel_id")["episode_num"].max()

    # (작품, 독자, 회차) 단위로 접어서 "등장" 하나를 한 행으로 만든다
    appear = (df.groupby(["novel_id", "user_key", "episode_num"])
                .agg(cur_comment_count=("comment_id", "count"),
                     cur_body_char_len=("body_char_len", "mean"),
                     cur_like_count=("like_count", "sum"),
                     cur_dislike_count=("dislike_count", "sum"),
                     cur_is_reply=("is_reply", "mean"))
                .reset_index()
                .sort_values(["novel_id", "user_key", "episode_num"]))

    g = appear.groupby(["novel_id", "user_key"], sort=False)

    appear["appearance_index"] = g.cumcount()
    appear["episodes_commented_so_far"] = appear["appearance_index"] + 1
    appear["comments_so_far"] = g["cur_comment_count"].cumsum()
    appear["first_episode"] = g["episode_num"].transform("first")

    prev_ep = g["episode_num"].shift(1)
    appear["gap_from_prev_appearance"] = (appear["episode_num"] - prev_ep).fillna(0.0)
    appear["max_consecutive_so_far"] = _running_max_consecutive(appear, g)

    span = appear["episode_num"] - appear["first_episode"] + 1
    appear["engagement_density_so_far"] = (
        appear["episodes_commented_so_far"] / span.replace(0, np.nan)).fillna(1.0)

    # 누적 평균 — 그 시점까지 쓴 댓글만 반영한다
    for src, dst in (("cur_body_char_len", "mean_body_len_so_far"),
                     ("cur_like_count", "mean_like_received_so_far"),
                     ("cur_is_reply", "reply_ratio_so_far")):
        appear[dst] = (g[src].cumsum() / appear["episodes_commented_so_far"])

    appear["novel_last_episode"] = appear["novel_id"].map(last_ep)
    appear["novel_progress"] = (appear["episode_num"]
                                / appear["novel_last_episode"].replace(0, np.nan))

    # 라벨: 이 등장이 그 작품에서의 마지막 등장인가
    appear["is_last_appearance"] = (g.cumcount(ascending=False) == 0).astype(int)

    # 우측 절단 제거 — 관측 끝자락의 등장은 이탈인지 아직 알 수 없다
    horizon = appear["novel_last_episode"] - censor_last_n
    observable = appear["episode_num"] <= horizon
    appear = appear[observable].reset_index(drop=True)
    if appear.empty:
        raise ValueError(
            "우측 절단 후 남는 행이 없습니다. 회차 수가 너무 적거나 "
            "censor_last_n(%d)이 과합니다" % censor_last_n)

    y = appear["is_last_appearance"].to_numpy()
    numeric = [c for c in USER_FEATURES if c in appear.columns]

    meta = appear[["novel_id", "user_key", "episode_num"]].copy()
    # 같은 독자의 여러 등장이 학습·검증에 갈라져 들어가면 개인 습관을 외운다.
    # user_key 단위로 묶어 분할한다.
    return TrainingFrame(
        X=appear[numeric].copy(),
        y=y,
        groups=appear["user_key"].to_numpy(),
        numeric=numeric,
        categorical=[],
        meta=meta,
        label_desc="이 회차를 끝으로 재등장 없음 (마지막 %d회차는 절단)" % censor_last_n,
    )


def _running_max_consecutive(appear: pd.DataFrame, g) -> pd.Series:
    """등장 시점까지의 '연속 회차 작성' 최대 길이를 누적으로 계산한다."""
    prev_ep = g["episode_num"].shift(1)
    is_consecutive = (appear["episode_num"] - prev_ep) == 1
    # 연속이 끊길 때마다 새 구간 번호를 매기고, 구간 내 순번 + 1 이 현재 연속 길이
    block = (~is_consecutive.fillna(False)).groupby(
        [appear["novel_id"], appear["user_key"]]).cumsum()
    run_len = appear.groupby(
        [appear["novel_id"], appear["user_key"], block]).cumcount() + 1
    return run_len.groupby([appear["novel_id"], appear["user_key"]]).cummax()


# ============================================================ 작품 단위 학습 데이터
def build_novel_training_frame(
    ep_feat: pd.DataFrame,
    novels: pd.DataFrame,
    threshold: float = 0.5,
    early_upto: int = 25,
    min_episodes: int = 10,
) -> TrainingFrame:
    """"왜 이 작품이 저 작품보다 이탈이 큰가"를 묻는 작품 단위 학습셋.

    회차 단위 모델과는 다른 질문이다. 여기서는 **작품 하나가 표본 하나**이므로
    수집한 작품 수가 곧 표본 수다. 회차를 아무리 많이 모아도 이 표본은 늘지 않는다.

    라벨은 초반 구간(1~`early_upto`화)의 평균 이탈률이 중앙값보다 높은가다.
    초반으로 한정하는 이유는 이탈의 거의 전부가 거기서 일어나고, 유료 구간은
    view_count가 세는 대상이 바뀌어 작품 간 비교가 성립하지 않기 때문이다.

    Args:
        threshold: 라벨 분위수. 0.5면 중앙값 기준 상·하위 이분.
        min_episodes: 초반 구간에 이 수 미만의 유효 회차만 있는 작품은 제외.
    """
    if ep_feat.empty:
        raise ValueError("episode_features가 비어 있습니다")

    df = ep_feat.copy()
    ep_num = pd.to_numeric(df["episode_num"], errors="coerce")
    early = df[(ep_num >= 2) & (ep_num <= early_upto)
               & pd.to_numeric(df["churn_step_clean"], errors="coerce").notna()]
    if early.empty:
        raise ValueError("초반 구간(2~%d화)에 유효한 이탈률이 없습니다" % early_upto)

    # 작품마다 회차 단위 값을 평균 내어 작품 한 행으로 접는다
    agg_map = {"churn_early": ("churn_step_clean", "mean"),
               "n_episodes": ("churn_step_clean", "size"),
               "pages_mean": ("pages", "mean"),
               "pages_std": ("pages", "std"),
               "gap_mean": ("days_since_prev", "mean"),
               "gap_std": ("days_since_prev", "std")}
    for c in REACTION_FEATURES + SENTIMENT_FEATURES:
        if c in early.columns and not c.startswith("d_"):
            agg_map[c + "_mean"] = (c, "mean")

    agg = early.groupby("novel_id").agg(**agg_map).reset_index()
    agg = agg[agg["n_episodes"] >= min_episodes]
    if len(agg) < 4:
        raise ValueError(
            "작품 %d개로는 작품 간 비교를 할 수 없습니다. 최소 수십 개, 신뢰할 만한 "
            "요인 순위를 원하면 200개 이상 수집하세요" % len(agg))

    if not novels.empty:
        keep = [c for c in ("novel_id", "genre_main", "status_code", "is_adult",
                            "preference_count", "chapter_count",
                            "free_chapter_count", "total_characters",
                            "reader_male_count", "reader_female_count",
                            "reader_age10s_pct", "reader_age20s_pct",
                            "reader_age30s_pct", "reader_age40s_pct",
                            "reader_age50s_pct") if c in novels.columns]
        agg = agg.merge(novels[keep].drop_duplicates("novel_id"),
                        on="novel_id", how="left")

    if {"reader_male_count", "reader_female_count"} <= set(agg.columns):
        tot = (pd.to_numeric(agg["reader_male_count"], errors="coerce").fillna(0)
               + pd.to_numeric(agg["reader_female_count"], errors="coerce").fillna(0))
        agg["reader_male_ratio"] = pd.to_numeric(
            agg["reader_male_count"], errors="coerce") / tot.replace(0, np.nan)
        agg = agg.drop(columns=["reader_male_count", "reader_female_count"])

    cut = float(agg["churn_early"].quantile(threshold))
    y = (agg["churn_early"] >= cut).astype(int).to_numpy()

    drop = {"novel_id", "churn_early", "n_episodes", "genre_main"}
    numeric = [c for c in agg.columns
               if c not in drop and pd.api.types.is_numeric_dtype(agg[c])]
    categorical = [c for c in ("genre_main",) if c in agg.columns]

    _assert_no_leakage(numeric + categorical, _EPISODE_LABEL_SOURCES)

    return TrainingFrame(
        X=agg[numeric + categorical].copy(),
        y=y,
        # 작품이 곧 표본이라 묶을 그룹이 없다. 층화 분할을 쓴다.
        groups=agg["novel_id"].to_numpy(),
        numeric=numeric,
        categorical=categorical,
        meta=agg[["novel_id", "churn_early", "n_episodes"]].copy(),
        label_desc="초반 %d화 평균 이탈률 >= %.4f (작품 상위 %.0f%%)"
                   % (early_upto, cut, (1 - threshold) * 100),
    )


# ==================================================================== 모델 정의
def _l1_kwargs() -> Dict[str, object]:
    """L1 정규화를 sklearn 버전에 맞는 방식으로 지정한다.

    sklearn 1.8에서 `penalty`가 폐기되고 `l1_ratio`로 옮겨갔다. 두 방식을 섞으면
    경고가 쏟아지고 1.10부터는 아예 동작하지 않는다. 버전을 보고 고른다.
    """
    from sklearn import __version__ as skl_version

    try:
        major, minor = (int(p) for p in skl_version.split(".")[:2])
    except ValueError:          # 개발 버전 문자열 등 — 신형으로 간주한다
        return {"l1_ratio": 1.0}
    return {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}


def build_models(class_weight: Optional[str] = None,
                 random_state: int = 42) -> Dict[str, object]:
    """비교할 sklearn 추정기들.

    로지스틱 회귀가 주력이다. 계수가 곧 해석이라 "연재 공백이 하루 늘면 이탈
    오즈가 몇 배" 같은 문장을 그대로 뽑을 수 있고, 이 데이터 규모(작품 수십 개)에서
    트리 앙상블보다 과적합이 덜하다. 나머지는 비선형 여지가 있는지 보는 대조군이고,
    Dummy는 "모델이 기저율보다 나은가"를 확인하는 하한선이다.

    `class_weight`는 기본이 None이다. 두 태스크 모두 양성률이 25~50%라 불균형
    보정이 필요 없고, "balanced"를 걸면 예측 확률만 위로 밀려 보정(calibration)이
    깨진다 — 실측에서 로지스틱 Brier가 0.181 → 0.240 으로 나빠져 기저율 모델보다
    못해졌다. 순위만 쓸 거라면 영향이 없지만, 확률값을 그대로 읽을 거면 켜지 마라.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=random_state),
        # L1은 계수를 0으로 밀어 피처를 스스로 고른다. 표본 수백 행에 피처 수십 개인
        # 지금 상황에서 "무엇이 실제로 남는가"를 보는 데 L2보다 곧다.
        "logistic_l1": LogisticRegression(
            solver="saga", C=0.3, max_iter=5000,
            class_weight=class_weight, random_state=random_state,
            **_l1_kwargs()),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight=class_weight,
            random_state=random_state, n_jobs=-1),
        "hist_gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=random_state),
    }


def build_pipeline(estimator, numeric: Sequence[str],
                   categorical: Sequence[str]) -> "object":
    """결측 대치 → 스케일링/원핫 → 추정기 파이프라인.

    전처리를 파이프라인 안에 넣어야 교차검증 각 폴드에서 학습 데이터만 보고
    평균·분산을 잡는다. 밖에서 미리 스케일링하면 검증 폴드 정보가 새어든다.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline([
        # lag 피처는 각 작품의 첫 회차에서 반드시 비므로 대치가 필요하다.
        # 중앙값을 쓰는 이유는 조회수 파생 지표의 분포가 심하게 치우쳐 있어서다.
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    transformers = [("num", num_pipe, list(numeric))]
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # min_frequency를 걸면 작품 고정효과 더미가 "희귀 범주"로 뭉개진다.
            # 작품 하나하나가 흡수해야 할 고유 수준이므로 묶으면 안 된다.
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False)),
            ]),
            list(categorical),
        ))

    return Pipeline([
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("clf", estimator),
    ])


# ==================================================================== 교차검증
def cross_validate(frame: TrainingFrame, n_splits: int = 5,
                   class_weight: Optional[str] = None,
                   random_state: int = 42,
                   group_split: bool = True) -> pd.DataFrame:
    """모델들을 교차검증한다. 분할 방식이 곧 "무엇을 묻는가"를 정한다.

    group_split=True — 작품(독자) 단위 GroupKFold. **"처음 보는 작품의 이탈을
        맞힐 수 있는가"** 를 묻는다. 일반화 성능. 작품 고정효과와 함께 쓰면 안 된다.

    group_split=False — 회차를 층화 무작위 분할. **"한 작품 안에서 어떤 회차가
        유독 빠지는가"** 를 묻는다. 요인 순위를 볼 때 쓰는 모드이고, 작품 고정효과가
        의미를 가지려면 같은 작품이 학습·검증 양쪽에 있어야 하므로 이쪽이어야 한다.

    두 모드의 숫자를 나란히 비교하면 안 된다. 묻는 질문이 다르다.
    """
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if group_split:
        n_groups = len(np.unique(frame.groups))
        splits = int(min(n_splits, n_groups))
        if splits < 2:
            raise ValueError(
                "그룹이 %d개뿐이라 그룹 교차검증을 할 수 없습니다. 작품 수를 늘리거나 "
                "group_split=False로 회차 단위 분할을 쓰세요" % n_groups)
        if splits < n_splits:
            log.warning("그룹이 %d개라 폴드를 %d개로 줄입니다", n_groups, splits)
        cv = GroupKFold(n_splits=splits)
        split_args = (frame.X, frame.y, frame.groups)
    else:
        minority = int(min(np.bincount(frame.y.astype(int))))
        splits = int(min(n_splits, max(minority, 2)))
        cv = StratifiedKFold(n_splits=splits, shuffle=True,
                             random_state=random_state)
        split_args = (frame.X, frame.y)

    base_rate = float(frame.y.mean())
    rows = []

    for name, est in build_models(class_weight, random_state).items():
        aucs, aps, briers = [], [], []
        for train_idx, test_idx in cv.split(*split_args):
            y_tr, y_te = frame.y[train_idx], frame.y[test_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue  # 한 클래스만 있는 폴드는 AUC가 정의되지 않는다
            pipe = build_pipeline(est, frame.numeric, frame.categorical)
            pipe.fit(frame.X.iloc[train_idx], y_tr)
            prob = pipe.predict_proba(frame.X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_te, prob))
            aps.append(average_precision_score(y_te, prob))
            briers.append(brier_score_loss(y_te, prob))

        if not aucs:
            log.warning("%s: 유효한 폴드가 없습니다 (라벨이 한쪽으로 쏠림)", name)
            continue
        rows.append({
            "model": name,
            "roc_auc": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            # PR-AUC는 기저율과 비교해야 의미가 있다. 불균형 라벨에서 ROC보다 정직하다.
            "pr_auc": float(np.mean(aps)),
            "pr_auc_lift": float(np.mean(aps) / base_rate) if base_rate else np.nan,
            "brier": float(np.mean(briers)),
            "folds": len(aucs),
        })

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


# ============================================================== 학습 및 해석
def fit_final(frame: TrainingFrame, model: str = "logistic",
              class_weight: Optional[str] = None,
              random_state: int = 42):
    """전체 데이터로 최종 모델을 학습한다 (배포·예측용)."""
    models = build_models(class_weight, random_state)
    if model not in models:
        raise ValueError("알 수 없는 모델: %s (가능: %s)"
                         % (model, ", ".join(models)))
    pipe = build_pipeline(models[model], frame.numeric, frame.categorical)
    pipe.fit(frame.X, frame.y)
    return pipe


def explain(pipe, frame: TrainingFrame) -> pd.DataFrame:
    """모델이 무엇을 보고 있는지 한 표로 만든다.

    로지스틱 회귀는 계수를 오즈비로 바꿔 돌려준다 — 표준화된 피처 기준이라
    "이 변수가 1 표준편차 커지면 이탈 오즈가 N배"로 읽는다.
    트리 계열은 계수가 없으므로 순열 중요도로 대체한다.
    """
    from sklearn.inspection import permutation_importance

    names = _feature_names(pipe, frame)
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "coef_"):
        coef = np.ravel(clf.coef_)
        out = pd.DataFrame({
            "feature": names[: len(coef)],
            "coef": coef,
            "odds_ratio": np.exp(coef),
        })
        out["abs_coef"] = out["coef"].abs()
        return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

    imp = permutation_importance(pipe, frame.X, frame.y, n_repeats=10,
                                 random_state=42, scoring="roc_auc", n_jobs=-1)
    return (pd.DataFrame({"feature": list(frame.X.columns),
                          "importance": imp.importances_mean,
                          "importance_std": imp.importances_std})
            .sort_values("importance", ascending=False))


# 요인 그룹 — 개별 컬럼이 아니라 "요인" 단위로 순위를 매기기 위한 정의.
# 접두사/정확 이름으로 매칭한다.
#
# 세 태스크(episode / user / novel)의 피처를 **모두** 덮어야 한다. 안 잡힌 피처는
# 순위에서 조용히 빠지고, 그러면 "그 요인은 중요하지 않다"가 아니라 "그 요인을
# 아예 재지 않았다"가 된다. 한 컬럼이 두 그룹에 걸려도 안 된다 — 두 번 순열되어
# 중요도가 부풀려진다. 두 조건 모두 테스트로 잠가 뒀다.
FACTOR_GROUPS: Dict[str, List[str]] = {
    # ---------------------------------------------------------- 회차 단위
    "연재 리듬(공백)": ["days_since_prev", "days_since_first", "gap_mean", "gap_std"],
    "회차 분량": ["pages"],              # pages / pages_mean / pages_std
    "연재 위치": ["episode_num", "novel_progress"],
    "페이월 구조": ["is_free", "is_paywall_boundary", "episodes_from_paywall",
                "has_paywall", "is_notice", "free_chapter_count"],
    "직전 회차 성과": ["prev_"],
    "독자 반응 구성": ["reaction_", "d_reaction_"],
    "댓글 감정(KOTE)": ["sent_", "d_sent_"],
    # ---------------------------------------------------------- 독자 단위
    "독자 참여 이력": ["episodes_commented_so_far", "comments_so_far",
                 "max_consecutive_so_far", "engagement_density_so_far",
                 "appearance_index"],
    "독자 이탈 징후": ["gap_from_prev_appearance"],
    "댓글 작성 양상": ["mean_body_len_so_far", "mean_like_received_so_far",
                 "reply_ratio_so_far", "cur_"],
    # ---------------------------------------------------------- 작품 단위
    # 작품 속성을 한 덩어리로 두면 12개 컬럼이 뭉쳐 해상도가 사라진다.
    # "작품이 중요하다"는 결론은 아무것도 알려주지 않으므로 쪼갠다.
    "작품 인기도(선작)": ["preference_count"],
    "장르": ["genre_main"],
    "독자 인구통계": ["reader_"],
    "작품 규모·상태": ["chapter_count", "total_characters", "status_code",
                  "is_adult"],
    "작품 고유(고정효과)": ["novel_fe"],
}


def _match_group(column: str, keys: Sequence[str]) -> bool:
    return any(column == k or column.startswith(k) for k in keys)


def rank_factors(frame: TrainingFrame, model: str = "random_forest",
                 n_splits: int = 5, n_repeats: int = 10,
                 group_split: bool = True, class_weight: Optional[str] = None,
                 random_state: int = 42, scoring: str = "roc_auc") -> pd.DataFrame:
    """요인 **그룹** 단위 순열 중요도 — "무엇이 이탈을 가장 크게 좌우하는가".

    두 가지를 의도적으로 다르게 했다.

    **그룹 단위로 섞는다.** 개별 피처 하나씩 섞으면, `sent_boredom`과
    `sent_complaint`처럼 상관이 높은 피처는 서로가 서로를 대신해 둘 다
    "중요하지 않다"고 나온다. 같은 요인은 함께 섞어야 그 요인 전체의 기여가 보인다.
    표본이 수백 행인데 피처가 수십 개인 상황에서 추정 안정성도 훨씬 낫다.

    **검증 폴드에서 섞는다.** 학습 데이터에서 섞으면 모델이 외운 것을 재는 꼴이다.
    RandomForest는 in-sample AUC가 0.97까지 올라가는데 그 위에서 잰 중요도는
    일반화되는 요인의 순위가 아니다. 폴드마다 학습은 train으로, 순열과 측정은
    test로 한다.

    Returns:
        순위 · 요인 · 컬럼 수 · 중요도(AUC 하락폭) · 표준편차 · 기준성능(폴드 평균).
        중요도가 클수록 그 요인 없이는 못 맞힌다는 뜻이다.
    """
    from sklearn.metrics import get_scorer
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    scorer = get_scorer(scoring)
    est = build_models(class_weight, random_state)[model]

    if group_split:
        n_groups = len(np.unique(frame.groups))
        splits = int(min(n_splits, n_groups))
        if splits < 2:
            raise ValueError("그룹이 %d개뿐이라 요인 순위를 낼 수 없습니다" % n_groups)
        cv = GroupKFold(n_splits=splits)
        split_args = (frame.X, frame.y, frame.groups)
    else:
        minority = int(min(np.bincount(frame.y.astype(int))))
        splits = int(min(n_splits, max(minority, 2)))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        split_args = (frame.X, frame.y)

    groups_cols = {name: [c for c in frame.X.columns if _match_group(c, keys)]
                   for name, keys in FACTOR_GROUPS.items()}
    groups_cols = {k: v for k, v in groups_cols.items() if v}

    drops: Dict[str, List[float]] = {k: [] for k in groups_cols}
    bases: List[float] = []
    rng = np.random.default_rng(random_state)

    for train_idx, test_idx in cv.split(*split_args):
        y_tr, y_te = frame.y[train_idx], frame.y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        pipe = build_pipeline(est, frame.numeric, frame.categorical)
        pipe.fit(frame.X.iloc[train_idx], y_tr)
        X_te = frame.X.iloc[test_idx]
        base = scorer(pipe, X_te, y_te)
        bases.append(base)

        for name, cols in groups_cols.items():
            for _ in range(n_repeats):
                shuffled = X_te.copy()
                # 그룹 내 컬럼을 같은 순열로 옮긴다. 행 단위로 통째로 섞어야
                # 그룹 내부의 상관 구조는 유지되고 라벨과의 관계만 끊긴다.
                order = rng.permutation(len(shuffled))
                shuffled[cols] = shuffled[cols].to_numpy()[order]
                drops[name].append(base - scorer(pipe, shuffled, y_te))

    if not bases:
        raise ValueError("유효한 폴드가 없어 요인 순위를 낼 수 없습니다")

    rows = [{"요인": name, "컬럼수": len(groups_cols[name]),
             "중요도": float(np.mean(v)), "표준편차": float(np.std(v))}
            for name, v in drops.items() if v]
    out = pd.DataFrame(rows).sort_values("중요도", ascending=False).reset_index(drop=True)
    out.insert(0, "순위", range(1, len(out) + 1))
    out["기준성능"] = float(np.mean(bases))
    return out


def _feature_names(pipe, frame: TrainingFrame) -> List[str]:
    """ColumnTransformer가 만든 최종 피처 이름 (원핫·결측 지시자 포함)."""
    try:
        return [n.split("__", 1)[-1]
                for n in pipe.named_steps["prep"].get_feature_names_out()]
    except Exception:  # sklearn 버전차 대비 — 이름 없다고 학습을 실패시킬 이유는 없다
        return list(frame.numeric) + list(frame.categorical)


def predict_frame(pipe, frame: TrainingFrame) -> pd.DataFrame:
    """메타 정보에 예측 확률을 붙여 돌려준다 (위험 회차 랭킹용)."""
    out = frame.meta.copy()
    out["y_true"] = frame.y
    out["churn_prob"] = pipe.predict_proba(frame.X)[:, 1]
    return out.sort_values("churn_prob", ascending=False)


# ==================================================================== 실행 진입점
def run(ep_feat: pd.DataFrame, comments: pd.DataFrame, episodes: pd.DataFrame,
        out_dir: str, task: str = "episode", model: str = "logistic",
        label_mode: str = "quantile", threshold: float = 0.75,
        n_splits: int = 5, class_weight: Optional[str] = None,
        min_episode: int = 2, max_episode: Optional[int] = None,
        fixed_effect: bool = False, novels: Optional[pd.DataFrame] = None,
        save_model: bool = True) -> Dict[str, object]:
    """학습 → 교차검증 → 해석 → 저장을 한 번에 수행한다."""
    if task == "episode":
        frame = build_episode_training_frame(
            ep_feat, label_mode, threshold, min_episode=min_episode,
            max_episode=max_episode, fixed_effect=fixed_effect)
    elif task == "user":
        frame = build_user_training_frame(comments, episodes)
    elif task == "novel":
        frame = build_novel_training_frame(
            ep_feat, novels if novels is not None else pd.DataFrame(),
            early_upto=max_episode or 25)
    else:
        raise ValueError("task는 episode / user / novel 이어야 합니다: %r" % task)

    # 작품 고정효과는 같은 작품이 학습·검증 양쪽에 있어야 의미가 있고,
    # 작품 단위 표본은 묶을 그룹 자체가 없다. 둘 다 그룹 분할을 쓰면 안 된다.
    group_split = not fixed_effect and task != "novel"

    n_groups = len(np.unique(frame.groups))
    log.info("[%s] 표본 %d행 · 그룹 %d개 · 양성 %d행(%.1f%%)",
             task, len(frame), n_groups, int(frame.y.sum()),
             100.0 * frame.y.mean())
    log.info("[%s] 라벨 정의: %s", task, frame.label_desc)
    log.info("[%s] 피처 %d개 (수치 %d · 범주 %d)%s", task,
             len(frame.numeric) + len(frame.categorical),
             len(frame.numeric), len(frame.categorical),
             " · 작품 고정효과 ON" if fixed_effect else "")

    scores = cross_validate(frame, n_splits=n_splits, class_weight=class_weight,
                            group_split=group_split)
    if not scores.empty:
        if group_split:
            how = "GroupKFold, 그룹=%s — 처음 보는 %s에 대한 일반화" % (
                ("novel_id", "작품") if task == "episode" else ("user_key", "독자"))
        else:
            how = "StratifiedKFold — 작품 내 회차 간 변동 설명"
        log.info("[%s] 교차검증 (%s)\n%s", task, how, scores.to_string(index=False))

    # 요인 순위는 교차검증에서 가장 잘 맞힌 모델로 뽑는다. 설명력이 없는 모델의
    # 중요도는 읽을 가치가 없기 때문이다.
    best = model
    if not scores.empty:
        top = scores.iloc[0]
        if top["model"] != "dummy":
            best = str(top["model"])
    pipe = fit_final(frame, model=model, class_weight=class_weight)
    expl = explain(pipe, frame)
    preds = predict_frame(pipe, frame)

    factors = rank_factors(frame, model=best, n_splits=n_splits,
                           group_split=group_split, class_weight=class_weight)
    log.info("[%s] 요인 순위 (%s · 검증 폴드 순열 = ROC-AUC 하락폭)\n%s",
             task, best, factors.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "%s_churn" % task)
    scores.to_csv(prefix + "_cv_scores.csv", index=False, encoding="utf-8-sig")
    expl.to_csv(prefix + "_explain.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(prefix + "_predictions.csv", index=False, encoding="utf-8-sig")
    factors.to_csv(prefix + "_factor_rank.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": task,
        "model": model,
        "rows": len(frame),
        "groups": int(n_groups),
        "positive_rate": float(frame.y.mean()),
        "label": frame.label_desc,
        "cv_split": "group" if group_split else "stratified",
        "fixed_effect": bool(fixed_effect),
        "features": list(frame.X.columns),
        "cv": scores.to_dict("records"),
        "factor_rank_model": best,
        "factor_rank": factors.to_dict("records"),
    }
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_model:
        import joblib
        joblib.dump(pipe, prefix + "_model.joblib")

    log.info("[%s] 저장 완료: %s_*.csv / _summary.json%s", task, prefix,
             " / _model.joblib" if save_model else "")
    return {"frame": frame, "scores": scores, "pipeline": pipe,
            "explain": expl, "predictions": preds, "factors": factors,
            "summary": summary}
