# -*- coding: utf-8 -*-
"""KOTE 기반 댓글 감정 분석 — 회차 단위 '내용' 신호를 만든다.

본문은 robots.txt가 막고 있어 수집하지 않는다. 그래서 "이 회차가 어땠나"를
직접 측정할 수 없고, 독자 반응으로 대리한다. 반응 버튼은 5종뿐이고 그나마
'웃김'이 96%를 차지해 해상도가 낮다. 남는 건 댓글 본문이고, 그걸 44개 감정으로
펼치는 것이 KOTE(searle-j/kote_for_easygoing_people)다.

## 44개를 그대로 쓰지 않는 이유

작품 수십 개 규모에서 회차당 44개 피처는 과적합의 지름길이다. 그리고 이탈
분석에서는 44개가 서로 다른 무게를 갖지 않는다 — 중요한 건 **어떤 부정 정서인가**다.

    "재미없음", "지긋지긋"  → 이탈 예고. 독자가 떠나기 직전의 말이다.
    "슬픔", "공포", "절망"  → 이탈 신호가 아니다. 오히려 작품이 잘 작동하고
                              있다는 증거다. 긴장감 있는 전개에 몰입한 반응이다.

이 둘을 "부정 감정"으로 뭉뜽그리면 신호가 상쇄된다. 그래서 이탈과의 관계를
기준으로 9개 축으로 묶는다 (`SENTIMENT_GROUPS`). 원본 44개 점수도 함께
저장하므로 축 정의를 바꿔 재집계할 때 모델을 다시 돌릴 필요는 없다.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KOTE_MODEL = "searle-j/kote_for_easygoing_people"

# 44개 KOTE 라벨 → 이탈 분석용 9개 축.
# 묶는 기준은 '감정의 종류'가 아니라 '이탈과 어떤 관계인가'다.
SENTIMENT_GROUPS: Dict[str, List[str]] = {
    # 이탈 직전의 말. 이탈 예측에서 가장 직접적인 신호여야 한다.
    "boredom": ["재미없음", "귀찮음", "지긋지긋"],
    # 불만은 있지만 아직 읽고 있다. 이탈 예고이되 boredom보다 약하다.
    "complaint": ["안타까움/실망", "불평/불만", "어이없음", "짜증", "한심함"],
    # 작품·작가를 향한 적대. 드물지만 나오면 강한 신호다.
    "hostility": ["우쭐댐/무시함", "화남/분노", "증오/혐오", "역겨움/징그러움"],
    # 재미있게 읽고 있다.
    "enjoyment": ["즐거움/신남", "기쁨", "행복", "편안/쾌적"],
    # 다음 화를 부르는 힘. 잔존과 가장 관련이 클 것으로 본다.
    "anticipation": ["기대감", "신기함/관심", "놀람", "깨달음"],
    # 작품·작가에 대한 애착. 고정 팬층의 언어.
    "attachment": ["감동/감탄", "환영/호의", "고마움", "뿌듯함", "안심/신뢰",
                   "아껴주는", "흐뭇함(귀여움/예쁨)", "존경"],
    # 몰입형 부정 정서 — 이탈이 아니라 몰입의 증거일 수 있다. 반드시 분리한다.
    "tension": ["슬픔", "불안/걱정", "비장함", "경악", "불쌍함/연민",
                "공포/무서움", "절망", "서러움"],
    # 전개를 못 따라가는 상태. 이탈로 이어질 수 있는 중립적 혼란.
    "confusion": ["당황/난처", "힘듦/지침", "의심/불신", "부담/안_내킴",
                  "부끄러움", "패배/자기혐오", "죄책감"],
    # KOTE가 감정 없음으로 본 것. 잡담·스포일러 방지용 한 줄 등.
    "neutral": ["없음"],
}

# 회차 피처로 나가는 컬럼 이름
GROUP_COLUMNS = ["sent_%s" % k for k in SENTIMENT_GROUPS]


def _flatten_labels() -> List[str]:
    out: List[str] = []
    for labels in SENTIMENT_GROUPS.values():
        out.extend(labels)
    return out


def load_pipeline(model_name: str = KOTE_MODEL, device: int = -1):
    """KOTE 파이프라인을 만든다.

    다중 라벨 분류라서 softmax가 아니라 sigmoid를 써야 한다. softmax를 쓰면
    44개 점수의 합이 1로 강제되어, "재미있으면서 기대된다" 같은 동시 발생을
    표현하지 못한다.
    """
    from transformers import pipeline

    log.info("KOTE 모델 로드: %s (device=%s)", model_name,
             "cpu" if device < 0 else "cuda:%d" % device)
    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        top_k=None,                     # 44개 라벨 전부 받는다
        function_to_apply="sigmoid",    # 다중 라벨
    )


def select_comments(comments: pd.DataFrame,
                    novels: Optional[pd.DataFrame] = None,
                    min_chars: int = 2) -> pd.DataFrame:
    """감정 분석에 넣을 댓글만 고른다.

    - 스티커는 본문이 없다 (content_type == STICKER).
    - 너무 짧은 것은 KOTE가 '없음'으로 밀어버려 노이즈만 는다.
    - **작가 본인 댓글은 뺀다.** 실측에서 한 작품은 댓글의 49.9%가 작가였다.
      작가는 독자 반응이 아니고, 매 회차 답글을 달기 때문에 그대로 두면
      회차별 감정이 작가의 인사말로 덮인다.
    """
    df = comments.copy()
    df = df[df.get("content_type", "TEXT").astype(str).str.upper() == "TEXT"]
    df = df[df["body"].astype(str).str.strip() != ""]
    df = df[pd.to_numeric(df.get("body_char_len"), errors="coerce").fillna(0) >= min_chars]

    if novels is not None and not novels.empty and "author_name" in novels.columns:
        author = dict(zip(novels["novel_id"], novels["author_name"].astype(str)))
        is_author = [str(nick) == author.get(nid, "\0")
                     for nid, nick in zip(df["novel_id"], df["nickname"])]
        dropped = int(np.sum(is_author))
        if dropped:
            log.info("작가 본인 댓글 %d건 제외 (%.1f%%)", dropped, 100.0 * dropped / len(df))
        df = df[~np.array(is_author, dtype=bool)]

    return df.reset_index(drop=True)


def score_comments(comments: pd.DataFrame,
                   novels: Optional[pd.DataFrame] = None,
                   model_name: str = KOTE_MODEL,
                   device: int = -1,
                   batch_size: int = 32,
                   max_length: int = 128,
                   cache_path: Optional[str] = None,
                   min_chars: int = 2) -> pd.DataFrame:
    """댓글마다 44개 감정 점수를 매긴다.

    Args:
        cache_path: 지정하면 이미 채점한 comment_uid는 건너뛰고 이어서 한다.
            CPU에서 수천 건이 수 분 걸리므로 중단·재개가 필요하다.

    Returns:
        comment_uid · novel_id · episode_num + 44개 라벨 점수 컬럼.
    """
    target = select_comments(comments, novels, min_chars=min_chars)
    if target.empty:
        log.warning("감정 분석 대상 댓글이 없습니다")
        return pd.DataFrame()

    done = pd.DataFrame()
    if cache_path and os.path.exists(cache_path):
        done = pd.read_csv(cache_path)
        known = set(done["comment_uid"].astype(str))
        before = len(target)
        target = target[~target["comment_uid"].astype(str).isin(known)]
        log.info("캐시 %d건 재사용, 신규 %d건 (전체 %d)",
                 len(done), len(target), before)
        if target.empty:
            return done

    pipe = load_pipeline(model_name, device)
    labels = _flatten_labels()
    texts = target["body"].astype(str).tolist()

    rows: List[Dict[str, float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start:start + batch_size]
        outputs = pipe(chunk, batch_size=len(chunk), truncation=True,
                       max_length=max_length)
        for out in outputs:
            rows.append({d["label"]: float(d["score"]) for d in out})
        if (start // batch_size) % 20 == 0:
            log.info("감정 분석 %d / %d (%.0f%%)", min(start + batch_size, total),
                     total, 100.0 * min(start + batch_size, total) / total)

    scores = pd.DataFrame(rows, columns=labels).fillna(0.0)
    meta = target[["comment_uid", "novel_id", "episode_num", "user_key",
                   "body_char_len"]].reset_index(drop=True)
    fresh = pd.concat([meta, scores], axis=1)

    out = pd.concat([done, fresh], ignore_index=True) if not done.empty else fresh
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
        log.info("감정 점수 저장: %s (%d건)", cache_path, len(out))
    return out


def to_groups(scored: pd.DataFrame) -> pd.DataFrame:
    """44개 라벨 점수를 9개 축으로 접는다 (축별 평균)."""
    out = scored[["comment_uid", "novel_id", "episode_num"]].copy()
    for name, labels in SENTIMENT_GROUPS.items():
        cols = [c for c in labels if c in scored.columns]
        out["sent_%s" % name] = scored[cols].mean(axis=1) if cols else 0.0
    return out


def build_episode_sentiment(scored: pd.DataFrame,
                            min_comments: int = 3) -> pd.DataFrame:
    """회차별 감정 집계 — 모델에 바로 넣을 수 있는 형태.

    Args:
        min_comments: 이 수 미만의 댓글로 계산된 회차는 값을 NaN으로 둔다.
            댓글 1건으로 만든 감정 평균을 '이 회차의 분위기'라고 부를 수 없다.
            버리지 않고 NaN으로 두는 이유는, 대치(impute) 여부를 모델 쪽에서
            결정하게 하고 결측 지시자로도 쓸 수 있게 하기 위해서다.
    """
    if scored.empty:
        return pd.DataFrame()

    grouped = to_groups(scored)
    agg = (grouped.groupby(["novel_id", "episode_num"])[GROUP_COLUMNS]
           .mean().reset_index())
    counts = (grouped.groupby(["novel_id", "episode_num"])
              .size().rename("sent_n_comments").reset_index())
    out = agg.merge(counts, on=["novel_id", "episode_num"])

    # 이탈 압력 지수 — 떠나겠다는 신호에서 붙잡는 신호를 뺀 값.
    # tension(슬픔·공포 등)은 일부러 넣지 않는다. 몰입의 증거이지 이탈이 아니다.
    out["sent_churn_index"] = (
        out["sent_boredom"] + out["sent_complaint"] + out["sent_hostility"]
        - out["sent_enjoyment"] - out["sent_anticipation"] - out["sent_attachment"]
    )

    thin = out["sent_n_comments"] < min_comments
    if thin.any():
        log.info("댓글 %d건 미만이라 감정값을 비운 회차: %d개 (전체 %d)",
                 min_comments, int(thin.sum()), len(out))
    value_cols = GROUP_COLUMNS + ["sent_churn_index"]
    out.loc[thin, value_cols] = np.nan
    return out


def add_sentiment_deltas(ep_sent: pd.DataFrame) -> pd.DataFrame:
    """직전 회차 대비 감정 변화량.

    이탈은 "작품이 원래 어떤가"보다 "지금까지 읽어온 흐름 대비 이번 화가
    어땠나"의 함수다. 절대 수준보다 변화량이 회차 단위 신호에 가깝다.
    """
    if ep_sent.empty:
        return ep_sent
    df = ep_sent.sort_values(["novel_id", "episode_num"]).reset_index(drop=True)
    g = df.groupby("novel_id", sort=False)
    for col in GROUP_COLUMNS + ["sent_churn_index"]:
        df["d_%s" % col] = df[col] - g[col].shift(1)
    return df


def run(comments: pd.DataFrame, novels: pd.DataFrame, out_dir: str,
        model_name: str = KOTE_MODEL, device: int = -1,
        batch_size: int = 32, min_comments: int = 3) -> pd.DataFrame:
    """댓글 감정 분석 전 과정 — 채점 → 축 집계 → 회차별 저장."""
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "comment_sentiment.csv")

    scored = score_comments(comments, novels, model_name=model_name,
                            device=device, batch_size=batch_size,
                            cache_path=cache)
    if scored.empty:
        return pd.DataFrame()

    ep_sent = add_sentiment_deltas(build_episode_sentiment(scored, min_comments))
    path = os.path.join(out_dir, "episode_sentiment.csv")
    ep_sent.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("회차 감정 %d행 저장: %s", len(ep_sent), path)
    return ep_sent
