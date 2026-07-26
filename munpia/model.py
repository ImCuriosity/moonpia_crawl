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
) -> TrainingFrame:
    """`episode_features.csv`에서 회차 이탈 분류 학습셋을 만든다.

    Args:
        ep_feat: `features` 명령이 만든 회차 피처 테이블.
        label_mode:
            "quantile" — 전체 회차의 이탈률 분포에서 상위 `threshold` 분위수를 넘으면 1.
                          작품마다 이탈률 수준이 달라 절대 임계는 편향되기 쉽다.
            "absolute" — `churn_step_clean >= threshold` 면 1.
        threshold: 분위수(0~1) 또는 절대 이탈률.
        min_episode: 이 회차 번호 미만은 제외. 1화는 직전 회차가 없어 라벨이 없다.

    Returns:
        TrainingFrame. 라벨이 NaN인 행(페이월 경계·미성숙 회차)은 전부 빠진다.
    """
    if ep_feat.empty:
        raise ValueError("episode_features가 비어 있습니다")
    if "churn_step_clean" not in ep_feat.columns:
        raise ValueError(
            "churn_step_clean 컬럼이 없습니다. 먼저 `features` 명령을 실행하세요")

    df = ep_feat.sort_values(["novel_id", "episode_num"]).reset_index(drop=True)
    df = _add_episode_lags(df)

    label_raw = pd.to_numeric(df["churn_step_clean"], errors="coerce")
    usable = label_raw.notna() & (pd.to_numeric(df["episode_num"],
                                                errors="coerce") >= min_episode)
    if not usable.any():
        raise ValueError(
            "학습 가능한 회차가 없습니다. churn_step_clean이 전부 비어 있습니다 — "
            "수집 회차가 너무 적거나 전부 미성숙(게시 7일 미만)일 수 있습니다")

    df = df[usable].reset_index(drop=True)
    label_raw = label_raw[usable].reset_index(drop=True)

    if label_mode == "quantile":
        cut = float(label_raw.quantile(threshold))
        desc = "churn_step_clean >= %.4f (상위 %.0f%% 분위)" % (cut, (1 - threshold) * 100)
    elif label_mode == "absolute":
        cut = float(threshold)
        desc = "churn_step_clean >= %.4f (절대 임계)" % cut
    else:
        raise ValueError("label_mode는 quantile 또는 absolute 여야 합니다: %r" % label_mode)

    y = (label_raw >= cut).astype(int).to_numpy()

    lag_cols = ["prev_%s" % c for c in EPISODE_LAG_SOURCES if "prev_%s" % c in df.columns]
    numeric = [c for c in EPISODE_STATIC_FEATURES + lag_cols + NOVEL_NUMERIC_FEATURES
               if c in df.columns]
    categorical = [c for c in NOVEL_CATEGORICAL_FEATURES if c in df.columns]

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


# ==================================================================== 모델 정의
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
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
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
                   random_state: int = 42) -> pd.DataFrame:
    """작품(또는 독자) 단위 GroupKFold로 모델들을 평가한다.

    무작위 K-Fold를 쓰면 안 된다. 같은 작품의 회차가 학습·검증에 섞이면 모델이
    그 작품의 조회수 수준을 외우고, 새 작품에 대한 일반화 성능이 과대평가된다.
    """
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(frame.groups))
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        raise ValueError(
            "그룹이 %d개뿐이라 교차검증을 할 수 없습니다. 수집 작품 수를 늘리세요"
            % n_groups)
    if splits < n_splits:
        log.warning("그룹이 %d개라 폴드를 %d개로 줄입니다", n_groups, splits)

    cv = GroupKFold(n_splits=splits)
    base_rate = float(frame.y.mean())
    rows = []

    for name, est in build_models(class_weight, random_state).items():
        aucs, aps, briers = [], [], []
        for train_idx, test_idx in cv.split(frame.X, frame.y, frame.groups):
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
        save_model: bool = True) -> Dict[str, object]:
    """학습 → 교차검증 → 해석 → 저장을 한 번에 수행한다."""
    if task == "episode":
        frame = build_episode_training_frame(ep_feat, label_mode, threshold)
    elif task == "user":
        frame = build_user_training_frame(comments, episodes)
    else:
        raise ValueError("task는 episode 또는 user 여야 합니다: %r" % task)

    n_groups = len(np.unique(frame.groups))
    log.info("[%s] 표본 %d행 · 그룹 %d개 · 양성 %d행(%.1f%%)",
             task, len(frame), n_groups, int(frame.y.sum()),
             100.0 * frame.y.mean())
    log.info("[%s] 라벨 정의: %s", task, frame.label_desc)
    log.info("[%s] 피처 %d개 (수치 %d · 범주 %d)", task,
             len(frame.numeric) + len(frame.categorical),
             len(frame.numeric), len(frame.categorical))

    scores = cross_validate(frame, n_splits=n_splits, class_weight=class_weight)
    if not scores.empty:
        log.info("[%s] 교차검증 (GroupKFold, 그룹=%s)\n%s", task,
                 "novel_id" if task == "episode" else "user_key",
                 scores.to_string(index=False))

    pipe = fit_final(frame, model=model, class_weight=class_weight)
    expl = explain(pipe, frame)
    preds = predict_frame(pipe, frame)

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "%s_churn" % task)
    scores.to_csv(prefix + "_cv_scores.csv", index=False, encoding="utf-8-sig")
    expl.to_csv(prefix + "_explain.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(prefix + "_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": task,
        "model": model,
        "rows": len(frame),
        "groups": int(n_groups),
        "positive_rate": float(frame.y.mean()),
        "label": frame.label_desc,
        "features": list(frame.X.columns),
        "cv": scores.to_dict("records"),
    }
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_model:
        import joblib
        joblib.dump(pipe, prefix + "_model.joblib")

    log.info("[%s] 저장 완료: %s_*.csv / _summary.json%s", task, prefix,
             " / _model.joblib" if save_model else "")
    return {"frame": frame, "scores": scores, "pipeline": pipe,
            "explain": expl, "predictions": preds, "summary": summary}
