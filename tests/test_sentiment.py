# -*- coding: utf-8 -*-
"""감정 분석 모듈 테스트 — 모델 없이 도는 부분만 검증한다.

    python -m tests.test_sentiment

KOTE 추론 자체(수백 MB 모델 다운로드)는 테스트하지 않는다. 검증하는 것은
라벨 분류 체계의 정합성과, 점수가 주어졌을 때의 집계 로직이다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from munpia import sentiment as S  # noqa: E402

# KOTE가 실제로 내놓는 44개 라벨 (모델 출력에서 확인한 값)
KOTE_LABELS = [
    '즐거움/신남', '감동/감탄', '기쁨', '기대감', '행복', '환영/호의', '신기함/관심',
    '고마움', '놀람', '뿌듯함', '안심/신뢰', '아껴주는', '편안/쾌적',
    '흐뭇함(귀여움/예쁨)', '깨달음', '존경', '안타까움/실망', '없음', '불평/불만',
    '당황/난처', '어이없음', '재미없음', '슬픔', '짜증', '우쭐댐/무시함', '불안/걱정',
    '힘듦/지침', '비장함', '의심/불신', '한심함', '부담/안_내킴', '부끄러움', '경악',
    '불쌍함/연민', '귀찮음', '화남/분노', '지긋지긋', '공포/무서움', '절망', '서러움',
    '패배/자기혐오', '증오/혐오', '역겨움/징그러움', '죄책감',
]


def test_groups_cover_every_kote_label_exactly_once():
    """44개 라벨이 빠짐없이, 중복 없이 9개 축에 배정돼야 한다.

    하나라도 빠지면 그 감정은 조용히 무시되고, 중복되면 이중 계상된다.
    둘 다 예외 없이 지나가므로 여기서 잡지 않으면 알 수 없다.
    """
    assigned = S._flatten_labels()
    assert len(assigned) == len(set(assigned)), \
        "중복 배정: %s" % [x for x in set(assigned) if assigned.count(x) > 1]
    missing = set(KOTE_LABELS) - set(assigned)
    extra = set(assigned) - set(KOTE_LABELS)
    assert not missing, "배정되지 않은 라벨: %s" % sorted(missing)
    assert not extra, "KOTE에 없는 라벨: %s" % sorted(extra)
    assert len(assigned) == 44


def test_tension_is_separated_from_churn_signals():
    """슬픔·공포는 이탈 축에 들어가면 안 된다.

    긴장감 있는 전개에 몰입한 반응이지 떠나겠다는 신호가 아니다. 이걸
    '부정 감정'으로 뭉뚱그리면 boredom 신호와 상쇄된다.
    """
    tension = set(S.SENTIMENT_GROUPS["tension"])
    for label in ("슬픔", "공포/무서움", "절망", "불안/걱정"):
        assert label in tension, "%s 가 tension 축에 없습니다" % label
    churnish = (set(S.SENTIMENT_GROUPS["boredom"])
                | set(S.SENTIMENT_GROUPS["complaint"])
                | set(S.SENTIMENT_GROUPS["hostility"]))
    assert not (tension & churnish), "tension이 이탈 축과 겹칩니다"


def _scored(rows):
    """(novel_id, episode_num, {라벨: 점수}) 목록을 점수 테이블로."""
    out = []
    for i, (nid, ep, scores) in enumerate(rows):
        rec = {"comment_uid": "c%d" % i, "novel_id": nid, "episode_num": ep}
        rec.update({lab: 0.0 for lab in KOTE_LABELS})
        rec.update(scores)
        out.append(rec)
    return pd.DataFrame(out)


def test_churn_index_direction():
    """이탈 지수는 '떠나겠다'에서 '붙잡는다'를 뺀 값이어야 한다."""
    bored = _scored([(1, 1, {"재미없음": 1.0, "지긋지긋": 1.0, "귀찮음": 1.0})] * 3)
    happy = _scored([(2, 1, {"즐거움/신남": 1.0, "기쁨": 1.0, "행복": 1.0,
                             "편안/쾌적": 1.0})] * 3)
    out = S.build_episode_sentiment(pd.concat([bored, happy], ignore_index=True))
    idx = dict(zip(out.novel_id, out.sent_churn_index))
    assert idx[1] > 0, "지루함만 있는 회차의 이탈 지수가 %.3f" % idx[1]
    assert idx[2] < 0, "즐거움만 있는 회차의 이탈 지수가 %.3f" % idx[2]


def test_tension_does_not_raise_churn_index():
    """슬픔·공포가 아무리 높아도 이탈 지수를 올리면 안 된다."""
    sad = _scored([(1, 1, {"슬픔": 1.0, "공포/무서움": 1.0, "절망": 1.0})] * 3)
    out = S.build_episode_sentiment(sad)
    assert out["sent_tension"].iloc[0] > 0, "tension 축이 잡히지 않았습니다"
    assert out["sent_churn_index"].iloc[0] <= 0, \
        "몰입형 부정 정서가 이탈 지수를 올렸습니다 (%.3f)" % out.sent_churn_index.iloc[0]


def test_thin_episodes_become_nan():
    """댓글 2건으로 만든 감정 평균을 '회차 분위기'라 부를 수 없다."""
    rows = [(1, 1, {"기쁨": 1.0}), (1, 1, {"기쁨": 1.0}),      # 2건 → NaN
            (1, 2, {"기쁨": 1.0}), (1, 2, {"기쁨": 1.0}), (1, 2, {"기쁨": 1.0})]
    out = S.build_episode_sentiment(_scored(rows), min_comments=3).set_index("episode_num")
    assert pd.isna(out.loc[1, "sent_enjoyment"]), "댓글 2건 회차가 비워지지 않았습니다"
    assert not pd.isna(out.loc[2, "sent_enjoyment"]), "댓글 3건 회차가 비워졌습니다"
    # 건수 자체는 남아야 결측 지시자로 쓸 수 있다
    assert out.loc[1, "sent_n_comments"] == 2


def test_author_comments_are_dropped():
    """작가 본인 댓글은 독자 반응이 아니다."""
    comments = pd.DataFrame([
        dict(novel_id=1, nickname="다물랑", body="댓글 감사합니다", content_type="TEXT",
             body_char_len=7, comment_uid="a", episode_num=1, user_key="u_a"),
        dict(novel_id=1, nickname="독자1", body="재밌어요", content_type="TEXT",
             body_char_len=4, comment_uid="b", episode_num=1, user_key="u_b"),
    ])
    novels = pd.DataFrame([dict(novel_id=1, author_name="다물랑")])
    out = S.select_comments(comments, novels)
    assert list(out["nickname"]) == ["독자1"], list(out["nickname"])


def test_stickers_and_empty_bodies_are_dropped():
    """스티커는 본문이 없고, 한 글자짜리는 노이즈만 는다."""
    comments = pd.DataFrame([
        dict(novel_id=1, nickname="a", body="", content_type="STICKER",
             body_char_len=0, comment_uid="1", episode_num=1, user_key="u"),
        dict(novel_id=1, nickname="b", body="ㅇ", content_type="TEXT",
             body_char_len=1, comment_uid="2", episode_num=1, user_key="u"),
        dict(novel_id=1, nickname="c", body="재밌다", content_type="TEXT",
             body_char_len=3, comment_uid="3", episode_num=1, user_key="u"),
    ])
    out = S.select_comments(comments, min_chars=2)
    assert list(out["comment_uid"]) == ["3"], list(out["comment_uid"])


def test_deltas_do_not_cross_novels():
    """감정 변화량이 작품 경계를 넘으면 안 된다."""
    ep_sent = pd.DataFrame([
        dict(novel_id=1, episode_num=1, sent_boredom=0.1),
        dict(novel_id=1, episode_num=2, sent_boredom=0.3),
        dict(novel_id=2, episode_num=1, sent_boredom=0.9),
    ])
    for col in S.GROUP_COLUMNS + ["sent_churn_index"]:
        if col not in ep_sent.columns:
            ep_sent[col] = 0.0
    out = S.add_sentiment_deltas(ep_sent).set_index(["novel_id", "episode_num"])
    assert pd.isna(out.loc[(1, 1), "d_sent_boredom"])
    assert abs(out.loc[(1, 2), "d_sent_boredom"] - 0.2) < 1e-9
    assert pd.isna(out.loc[(2, 1), "d_sent_boredom"]), \
        "작품 2의 첫 회차가 작품 1의 값을 끌어왔습니다"


def test_empty_input_is_safe():
    assert S.build_episode_sentiment(pd.DataFrame()).empty
    assert S.add_sentiment_deltas(pd.DataFrame()).empty


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
