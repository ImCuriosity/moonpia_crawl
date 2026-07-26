# -*- coding: utf-8 -*-
"""이탈 추론 모듈 테스트 — 학습셋 구성과 누수 차단을 검증한다.

    python -m tests.test_model

성능 자체는 여기서 보지 않는다. 검증하는 것은 "모델이 봐서는 안 될 값을
보고 있지 않은가"와 "표본이 조용히 사라지지 않는가" 두 가지다.
scikit-learn이 없으면 전부 건너뛴다 (선택 의존성).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from munpia.features import build_dataset  # noqa: E402
from munpia import model as M  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

BASE = 1_700_000_000
COLLECTED = "2026-07-26 00:00:00"


def _fixture(n_novels: int = 6, n_eps: int = 30):
    """작품 여러 개 · 회차 · 댓글을 갖춘 최소 데이터셋.

    이탈률이 회차 번호의 결정함수가 되지 않게 잡음을 섞는다. 결정적으로 만들면
    모델이 AUC 1.0을 찍어서 누수 탐지 검사가 무의미해진다. 대신 "연재 공백이
    길수록 이탈이 크다"는 약한 신호만 심어둔다.
    """
    rng = np.random.default_rng(0)
    novels, episodes, comments = [], [], []
    cid = 1
    for n in range(n_novels):
        nid = 900 + n
        novels.append(dict(novel_id=nid, title="작품%d" % n, genre_main="FANTASY",
                           status_code=0, preference_count=1000 + n * 10,
                           is_free=0, is_adult=0, total_view_count=0,
                           total_like_count=0))
        view = 10000.0
        day = 0
        for e in range(1, n_eps + 1):
            gap = int(rng.integers(1, 5))
            day += gap
            churn = 0.02 + 0.01 * (gap - 1) + float(rng.normal(0, 0.02))
            view = max(50.0, view * (1.0 - min(max(churn, -0.2), 0.6)))
            episodes.append(dict(
                episode_uid="%d_%d" % (nid, e), novel_id=nid, entry_id=e,
                episode_num=e, title="%d화" % e, published_at="",
                published_ts=BASE + day * 86400, view_count=int(view),
                like_count=int(view // 50), comment_count=5, comment_collected=5,
                pages=8 + (e % 5), char_estimate=(8 + (e % 5)) * 900,
                is_free=1 if e <= 10 else 0, is_adult=0, is_notice=0,
                reaction_total=int(view // 100), collected_at=COLLECTED))
            for k in range(6):
                # 독자마다 이탈 시점이 다르도록 참여 구간을 흩어놓는다
                if e > 4 + k * 4 + int(rng.integers(0, 3)):
                    continue
                comments.append(dict(
                    comment_uid="%d_%d_%d" % (nid, e, cid),
                    episode_uid="%d_%d" % (nid, e), novel_id=nid, entry_id=e,
                    episode_num=e, comment_id=cid, parent_id=0,
                    user_key="u_%d_%d" % (n, k), nickname="r%d" % k,
                    blog_url="r%d" % k, content_type="TEXT", body="재밌다",
                    sticker_url="", body_char_len=10 + k,
                    created_at="", created_ts=BASE + day * 86400,
                    like_count=k, dislike_count=0, reply_level=0, is_reply=0,
                    reply_count=0, is_secret=0, is_blocked=0,
                    collected_at=COLLECTED))
                cid += 1
    novels = pd.DataFrame(novels)
    episodes = pd.DataFrame(episodes)
    comments = pd.DataFrame(comments)
    ep_feat, _ = build_dataset(novels, episodes, comments)
    return ep_feat, episodes, comments


def test_leakage_guard_rejects_label_sources():
    """라벨 계산에 쓰인 컬럼이 피처에 들어오면 즉시 실패해야 한다.

    조용히 통과시키면 ROC-AUC 0.99를 보고 모델이 잘 됐다고 착각하게 된다.
    """
    for banned in ("view_count", "retention_step", "churn_step", "like_per_view"):
        try:
            M._assert_no_leakage(["episode_num", banned], M._EPISODE_LABEL_SOURCES)
        except ValueError:
            continue
        raise AssertionError("%s 를 누수로 잡아내지 못했습니다" % banned)


def test_episode_features_contain_no_view_derived_columns():
    """실제로 만들어진 학습셋에 조회수 파생 컬럼이 없어야 한다."""
    ep_feat, _, _ = _fixture()
    frame = M.build_episode_training_frame(ep_feat)
    leaked = set(frame.X.columns) & M._EPISODE_LABEL_SOURCES
    assert not leaked, "피처에 %s 가 남아 있습니다" % sorted(leaked)
    assert "prev_churn_step" in frame.X.columns, "직전 회차 lag 피처가 없습니다"
    assert frame.y.sum() > 0 and frame.y.sum() < len(frame.y), \
        "라벨이 한쪽으로 완전히 쏠렸습니다"


def test_episode_lags_do_not_cross_novels():
    """lag 피처가 작품 경계를 넘어 앞 작품 값을 끌어오면 안 된다."""
    ep_feat, _, _ = _fixture()
    lagged = M._add_episode_lags(
        ep_feat.sort_values(["novel_id", "episode_num"]).reset_index(drop=True))
    first_rows = lagged.groupby("novel_id").head(1)
    assert first_rows["prev_churn_step"].isna().all(), \
        "각 작품의 첫 회차는 직전 값이 없어야 합니다"


def test_absolute_label_mode():
    """절대 임계 라벨링도 동작해야 한다."""
    ep_feat, _, _ = _fixture()
    frame = M.build_episode_training_frame(ep_feat, label_mode="absolute",
                                           threshold=0.03)
    assert "절대 임계" in frame.label_desc
    assert set(np.unique(frame.y)) <= {0, 1}


def test_user_frame_censors_tail_episodes():
    """관측 끝자락의 등장은 이탈로 라벨링하면 안 된다 (우측 절단)."""
    _, episodes, comments = _fixture()
    frame = M.build_user_training_frame(comments, episodes, censor_last_n=3)
    last_ep = episodes.groupby("novel_id")["episode_num"].max()
    for nid, ep in zip(frame.meta["novel_id"], frame.meta["episode_num"]):
        assert ep <= last_ep[nid] - 3, \
            "작품 %d의 %d화가 절단되지 않았습니다 (마지막 %d화)" % (nid, ep, last_ep[nid])


def test_user_frame_features_are_backward_looking():
    """독자 피처는 그 시점까지의 누적값이라 단조 증가해야 한다."""
    _, episodes, comments = _fixture()
    frame = M.build_user_training_frame(comments, episodes)
    df = frame.X.copy()
    df["user_key"] = frame.meta["user_key"].to_numpy()
    df["episode_num"] = frame.meta["episode_num"].to_numpy()
    for _, grp in df.sort_values(["user_key", "episode_num"]).groupby("user_key"):
        counts = grp["episodes_commented_so_far"].to_numpy()
        assert (np.diff(counts) >= 0).all(), "누적 참여 회차가 감소했습니다"


def test_user_labels_mark_only_last_appearance():
    """한 독자의 등장 중 이탈(1)로 표시되는 건 최대 하나여야 한다."""
    _, episodes, comments = _fixture()
    frame = M.build_user_training_frame(comments, episodes)
    df = pd.DataFrame({"key": frame.meta["novel_id"].astype(str) + "|"
                              + frame.meta["user_key"], "y": frame.y})
    assert df.groupby("key")["y"].sum().max() <= 1, \
        "한 독자에게 이탈 라벨이 두 번 이상 붙었습니다"


def test_cross_validation_beats_baseline():
    """GroupKFold 교차검증이 돌고, dummy가 정확히 0.5여야 한다.

    dummy가 0.5가 아니면 평가 코드 자체가 잘못된 것이다.
    """
    if not HAS_SKLEARN:
        print("       (scikit-learn 없음 — 건너뜀)")
        return
    ep_feat, _, _ = _fixture(n_novels=8, n_eps=40)
    frame = M.build_episode_training_frame(ep_feat)
    scores = M.cross_validate(frame, n_splits=4)
    assert not scores.empty, "교차검증 결과가 비었습니다"
    dummy = scores[scores["model"] == "dummy"]["roc_auc"].iloc[0]
    assert abs(dummy - 0.5) < 1e-9, "dummy AUC가 %.4f 입니다" % dummy
    best = scores["roc_auc"].max()
    assert best < 0.999, "AUC %.4f — 누수가 의심됩니다" % best


def test_fit_and_explain():
    """최종 학습 → 계수 해석 → 예측까지 이어져야 한다."""
    if not HAS_SKLEARN:
        print("       (scikit-learn 없음 — 건너뜀)")
        return
    ep_feat, _, _ = _fixture()
    frame = M.build_episode_training_frame(ep_feat)
    pipe = M.fit_final(frame, model="logistic")
    expl = M.explain(pipe, frame)
    assert "odds_ratio" in expl.columns and len(expl) > 0
    preds = M.predict_frame(pipe, frame)
    assert len(preds) == len(frame)
    assert preds["churn_prob"].between(0, 1).all()
    # 확률 내림차순 정렬 — 위험 회차 랭킹으로 바로 쓸 수 있어야 한다
    assert preds["churn_prob"].is_monotonic_decreasing


def test_episode_range_is_respected():
    """min/max_episode 로 자른 범위 밖의 회차가 남으면 안 된다."""
    ep_feat, _, _ = _fixture(n_novels=6, n_eps=40)
    frame = M.build_episode_training_frame(ep_feat, min_episode=2, max_episode=25)
    eps = frame.meta["episode_num"]
    assert eps.min() >= 2 and eps.max() <= 25, "회차 범위 %d~%d" % (eps.min(), eps.max())


def test_within_novel_label_balances_per_novel():
    """작품 내 분위수 라벨은 작품마다 양성 비율이 비슷해야 한다.

    전체 분위수를 쓰면 이탈률이 높은 작품 하나가 양성을 독식한다. 그러면
    모델이 배우는 것은 '어떤 회차가 위험한가'가 아니라 '어떤 작품인가'다.
    """
    ep_feat, _, _ = _fixture(n_novels=6, n_eps=40)
    frame = M.build_episode_training_frame(ep_feat, label_mode="within_novel",
                                           threshold=0.75)
    rates = (pd.DataFrame({"novel_id": frame.meta["novel_id"].to_numpy(),
                           "y": frame.y})
             .groupby("novel_id")["y"].mean())
    assert rates.max() - rates.min() < 0.35, \
        "작품별 양성 비율 편차가 큽니다: %s" % rates.round(2).to_dict()


def test_fixed_effect_adds_novel_dummy():
    """고정효과를 켜면 작품 식별자가 범주형 피처로 들어가야 한다."""
    ep_feat, _, _ = _fixture()
    plain = M.build_episode_training_frame(ep_feat)
    fe = M.build_episode_training_frame(ep_feat, fixed_effect=True)
    assert "novel_fe" not in plain.X.columns
    assert "novel_fe" in fe.X.columns
    assert fe.X["novel_fe"].nunique() == len(set(fe.groups))


def test_factor_rank_is_out_of_fold():
    """요인 순위의 기준성능은 in-sample이 아니라 교차검증 값이어야 한다.

    학습 데이터에서 섞으면 RandomForest의 in-sample AUC(0.97+) 위에서 재게 되어
    '외운 것'의 순위가 나온다.
    """
    if not HAS_SKLEARN:
        print("       (scikit-learn 없음 — 건너뜀)")
        return
    ep_feat, _, _ = _fixture(n_novels=8, n_eps=40)
    frame = M.build_episode_training_frame(ep_feat)
    ranked = M.rank_factors(frame, model="random_forest", n_splits=4, n_repeats=3)
    assert not ranked.empty
    assert list(ranked.columns[:2]) == ["순위", "요인"]
    base = ranked["기준성능"].iloc[0]
    assert 0.3 < base < 0.98, "기준성능 %.3f — in-sample 값으로 보입니다" % base
    assert ranked["순위"].is_monotonic_increasing
    assert ranked["중요도"].is_monotonic_decreasing, "중요도 내림차순이 아닙니다"


def test_factor_groups_cover_declared_features():
    """선언한 피처가 어느 요인 그룹에도 안 잡히면 순위에서 조용히 빠진다."""
    ep_feat, _, _ = _fixture()
    frame = M.build_episode_training_frame(ep_feat, fixed_effect=True)
    unmatched = [c for c in frame.X.columns
                 if not any(M._match_group(c, keys)
                            for keys in M.FACTOR_GROUPS.values())]
    assert not unmatched, "요인 그룹에 배정되지 않은 피처: %s" % unmatched


def test_novel_frame_needs_enough_novels():
    """작품이 너무 적으면 작품 간 비교를 거부해야 한다."""
    ep_feat, _, _ = _fixture(n_novels=3, n_eps=40)
    try:
        M.build_novel_training_frame(ep_feat, pd.DataFrame())
    except ValueError as exc:
        assert "작품" in str(exc)
        return
    raise AssertionError("작품 3개인데 작품 간 학습셋이 만들어졌습니다")


def test_novel_frame_is_one_row_per_novel():
    """작품 단위 학습셋은 작품 하나가 행 하나여야 한다."""
    ep_feat, _, _ = _fixture(n_novels=8, n_eps=40)
    frame = M.build_novel_training_frame(ep_feat, pd.DataFrame(), min_episodes=5)
    assert len(frame) == frame.meta["novel_id"].nunique()
    assert len(frame) <= 8
    leaked = set(frame.X.columns) & M._EPISODE_LABEL_SOURCES
    assert not leaked, "작품 피처에 라벨 출처가 남았습니다: %s" % sorted(leaked)


def test_too_few_groups_is_rejected():
    """작품이 한 개뿐이면 교차검증이 불가능하다는 걸 명시적으로 알려야 한다."""
    if not HAS_SKLEARN:
        print("       (scikit-learn 없음 — 건너뜀)")
        return
    ep_feat, _, _ = _fixture(n_novels=1, n_eps=40)
    frame = M.build_episode_training_frame(ep_feat)
    try:
        M.cross_validate(frame, n_splits=5)
    except ValueError as exc:
        assert "그룹" in str(exc)
        return
    raise AssertionError("그룹이 1개인데 교차검증이 통과했습니다")


def run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failed += 1
            import traceback
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            print("  FAIL  %s (line %d): %s" % (name, tb.lineno, tb.line))
            if str(exc):
                print("        %s" % exc)
        except Exception as exc:
            failed += 1
            print("  ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d개 중 %d개 통과" % (len(tests), len(tests) - failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
