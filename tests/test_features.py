# -*- coding: utf-8 -*-
"""파생 피처 테스트 — 이탈률 계산이 조용히 비는 경우를 잡는다.

    python -m tests.test_features

여기 있는 테스트는 대부분 실제로 터졌던 결함의 회귀 방지용이다. 공통점은
전부 예외 없이 조용히 NaN을 만들었다는 것이라, 학습 단계에 가서야 "데이터가
없다"로 나타났다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from munpia.features import (build_episode_features, build_user_features,  # noqa: E402
                             build_episode_audience_features)

BASE = 1_700_000_000  # 2023-11-14 근처


def _episodes(novel_id: int, n: int, free_upto: int,
              collected_at: str = "2026-07-26 00:00:00") -> pd.DataFrame:
    """조회수가 회차마다 3%씩 빠지는 단순 시계열."""
    rows, view = [], 10000
    for i in range(1, n + 1):
        view = int(view * 0.97)
        rows.append(dict(
            episode_uid="%d_%d" % (novel_id, i), novel_id=novel_id, entry_id=i,
            episode_num=i, title="%d화" % i, published_at="",
            published_ts=BASE + i * 86400, view_count=view,
            like_count=view // 50, comment_count=view // 200,
            pages=10, char_estimate=9000,
            is_free=1 if i <= free_upto else 0, is_adult=0, is_notice=0,
            reaction_total=view // 100, collected_at=collected_at,
        ))
    return pd.DataFrame(rows)


def test_mature_flag_survives_second_resolution_timestamps():
    """collected_at이 초 단위로 파싱돼도 age_days가 정상이어야 한다.

    pandas 2.0부터 to_datetime이 datetime64[s]를 돌려주는데, 예전 코드가
    ns를 가정하고 10**9로 나눠 age_days를 -19000일로 만들었다. 그 결과
    is_mature가 전부 0이 되고 churn_step_clean이 통째로 비었다.
    """
    out = build_episode_features(_episodes(1, 40, free_upto=25))
    assert (out["age_days"] > 0).all(), "경과일이 음수입니다: %s" % out["age_days"].head().tolist()
    assert out["is_mature"].sum() > 0, "성숙 회차가 하나도 없습니다"
    assert out["churn_step_clean"].notna().sum() > 30, \
        "churn_step_clean이 %d행뿐입니다 (학습 권장 컬럼이 비었음)" % \
        out["churn_step_clean"].notna().sum()


def test_all_paid_novel_keeps_churn():
    """무료 회차가 하나도 없는 작품도 이탈률이 남아야 한다.

    last_free_episode가 NaN이면 `NaN.abs() > 1`이 False라, 전 회차가
    churn_step_ex_paywall에서 탈락하던 버그.
    """
    out = build_episode_features(_episodes(2, 20, free_upto=0))
    assert out["has_paywall"].sum() == 0, "무료 회차가 없으면 페이월도 없다"
    assert out["churn_step_ex_paywall"].notna().sum() == 19, \
        "전유료 작품의 이탈률이 %d행만 남았습니다" % out["churn_step_ex_paywall"].notna().sum()


def test_all_free_novel_keeps_last_episodes():
    """무료연재(전 회차 무료)는 마지막 회차가 페이월로 오인되면 안 된다."""
    out = build_episode_features(_episodes(3, 20, free_upto=20))
    assert out["has_paywall"].sum() == 0
    assert out["last_free_episode"].isna().all(), \
        "경계가 없는데 last_free_episode가 채워졌습니다"
    dropped = out.loc[out["churn_step_ex_paywall"].isna(), "episode_num"].tolist()
    assert dropped == [1], "1화 외에 %s 가 부당하게 제외됐습니다" % dropped


def test_real_paywall_boundary_is_excluded():
    """진짜 페이월(무료→유료 전환)은 경계 ±1화가 제외되어야 한다."""
    out = build_episode_features(_episodes(4, 40, free_upto=25))
    assert out["has_paywall"].sum() == 40
    assert out["is_paywall_boundary"].sum() == 1
    dropped = out.loc[out["churn_step_ex_paywall"].isna(), "episode_num"].tolist()
    assert dropped == [1, 24, 25, 26], "제외된 회차가 %s 입니다" % dropped


def test_paywall_flags_are_per_novel():
    """여러 작품을 한 번에 넣어도 작품별로 따로 판정해야 한다."""
    df = pd.concat([_episodes(10, 10, free_upto=5),    # 페이월 있음
                    _episodes(11, 10, free_upto=10),   # 전무료
                    _episodes(12, 10, free_upto=0)],   # 전유료
                   ignore_index=True)
    out = build_episode_features(df)
    flags = out.groupby("novel_id")["has_paywall"].max().to_dict()
    assert flags == {10: 1, 11: 0, 12: 0}, flags


def test_user_features_survive_missing_timestamps():
    """created_ts가 전부 결측이어도 죽지 않아야 한다.

    `int(pd.Series.min() or 0)` 은 min()이 NaN일 때 NaN이 truthy라
    int(NaN) → ValueError 로 전체 실행을 중단시켰다.
    """
    cm = pd.DataFrame([
        dict(novel_id=1, user_key="u_a", nickname="A", episode_num=ep, is_reply=0,
             like_count=1, body_char_len=10, created_ts=np.nan)
        for ep in (1, 2, 3)
    ])
    users = build_user_features(cm)
    assert len(users) == 1
    assert users.iloc[0]["first_ts"] == 0 and users.iloc[0]["last_ts"] == 0
    assert users.iloc[0]["max_consecutive_episodes"] == 3


def test_returning_commenter_ratio():
    """직전 회차에도 댓글을 단 비율이 고정 팬층 지표다."""
    rows = []
    for ep, users in ((1, ["u_a", "u_b"]), (2, ["u_a", "u_c"]), (3, ["u_a", "u_b"])):
        for u in users:
            rows.append(dict(novel_id=1, user_key=u, nickname=u, episode_num=ep,
                             is_reply=0, like_count=0, body_char_len=5,
                             created_ts=BASE))
    out = build_episode_audience_features(pd.DataFrame(rows)).set_index("episode_num")
    assert out.loc[1, "returning_commenter_ratio"] == 0.0      # 첫 회차
    assert out.loc[2, "returning_commenter_ratio"] == 0.5      # u_a만 재등장
    assert out.loc[2, "new_commenter_ratio"] == 0.5            # u_c 신규
    assert out.loc[3, "returning_commenter_ratio"] == 0.5      # u_a 재등장, u_b는 건너뜀
    assert out.loc[3, "new_commenter_ratio"] == 0.0            # 전원 기존 독자


def test_churn_basis_switches_the_label_source():
    """churn_basis=like면 이탈률이 추천수에서 나와야 한다."""
    df = _episodes(20, 10, free_upto=5)
    # 조회수는 일정하게, 추천수만 떨어뜨린다 — 기준이 바뀌면 결과도 바뀌어야 한다
    df["view_count"] = 10000
    df["like_count"] = [1000 - 50 * i for i in range(len(df))]

    by_view = build_episode_features(df, churn_basis="view")
    by_like = build_episode_features(df, churn_basis="like")

    assert (by_view["churn_step"].dropna().abs() < 1e-9).all(), \
        "조회수가 일정한데 조회수 기준 이탈률이 0이 아닙니다"
    assert (by_like["churn_step"].dropna() > 0).all(), \
        "추천수가 감소하는데 추천수 기준 이탈률이 양수가 아닙니다"
    assert by_like["churn_basis"].eq("like").all(), "기준이 기록되지 않았습니다"


def test_churn_basis_rejects_unknown_value():
    """오타를 조용히 조회수 기준으로 처리하면 안 된다."""
    try:
        build_episode_features(_episodes(21, 5, 3), churn_basis="likes")
    except ValueError as exc:
        assert "churn_basis" in str(exc)
        return
    raise AssertionError("알 수 없는 churn_basis가 통과했습니다")


def test_view_columns_survive_like_basis():
    """기준을 바꿔도 조회수 계열 진단 컬럼은 남아야 한다.

    페이월에서 조회수가 무너지는지 확인하려면 두 값이 다 필요하다.
    """
    out = build_episode_features(_episodes(22, 20, free_upto=10), churn_basis="like")
    for col in ("view_count", "prev_view_count", "first_view_count", "view_ma5"):
        assert col in out.columns, "%s 가 사라졌습니다" % col
    assert out["prev_view_count"].notna().sum() == 19


def test_empty_input_is_safe():
    """빈 테이블이 들어와도 예외 없이 빈 결과를 돌려준다."""
    assert build_episode_features(pd.DataFrame()).empty
    assert build_user_features(pd.DataFrame()).empty
    assert build_episode_audience_features(pd.DataFrame()).empty


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
