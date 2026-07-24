# -*- coding: utf-8 -*-
"""수집 결과를 구조화 파일로 내보낸다 (요구사항 3.4).

작품/회차/댓글을 3개 테이블로 나누고 식별자로 조인 가능하게 저장한다.
결과는 수집 즉시 append 하므로 중간에 크롤이 죽어도 그때까지의 데이터가 남고,
완료 목록(_completed.txt)을 근거로 재실행 시 이미 끝난 작품을 건너뛴다.

CSV는 Excel 호환을 위해 utf-8-sig로 쓴다 (BOM 없으면 한글이 깨져 보인다).
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set

from .schema import CommentRecord, EpisodeRecord, NovelRecord

log = logging.getLogger(__name__)

NOVELS_FILE = "novels"
EPISODES_FILE = "episodes"
COMMENTS_FILE = "comments"
COMPLETED_FILE = "_completed.txt"
ERROR_FILE = "_errors.log"


class _Table(object):
    """헤더를 한 번만 쓰고 이후 append 하는 CSV/JSONL 라이터."""

    def __init__(self, path: str, fieldnames: List[str], fmt: str):
        self.path = path
        self.fieldnames = fieldnames
        self.fmt = fmt
        self.count = 0
        exists = os.path.exists(path) and os.path.getsize(path) > 0

        if fmt == "csv":
            self._fh = open(path, "a", encoding="utf-8-sig", newline="")
            self._writer = csv.DictWriter(
                self._fh, fieldnames=fieldnames, extrasaction="ignore",
                quoting=csv.QUOTE_MINIMAL,
            )
            if not exists:
                self._writer.writeheader()
        else:  # jsonl
            self._fh = open(path, "a", encoding="utf-8")
            self._writer = None

    def write(self, rows: Iterable[Dict[str, Any]]) -> int:
        written = 0
        for row in rows:
            if self.fmt == "csv":
                self._writer.writerow(row)
            else:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        if written:
            self._fh.flush()
        self.count += written
        return written

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class DatasetWriter(object):
    """novels / episodes / comments 3개 테이블을 동시에 관리한다.

    사용:
        with DatasetWriter("data/raw", fmt="csv") as w:
            for result in crawler.crawl_many(ids):
                w.write_result(result)
    """

    def __init__(self, out_dir: str, fmt: str = "csv", resume: bool = True):
        if fmt not in ("csv", "jsonl"):
            raise ValueError("fmt는 csv 또는 jsonl 이어야 합니다")
        self.out_dir = out_dir
        self.fmt = fmt
        os.makedirs(out_dir, exist_ok=True)

        ext = "csv" if fmt == "csv" else "jsonl"
        self.novels = _Table(os.path.join(out_dir, "%s.%s" % (NOVELS_FILE, ext)),
                             NovelRecord.FIELDS, fmt)
        self.episodes = _Table(os.path.join(out_dir, "%s.%s" % (EPISODES_FILE, ext)),
                               EpisodeRecord.FIELDS, fmt)
        self.comments = _Table(os.path.join(out_dir, "%s.%s" % (COMMENTS_FILE, ext)),
                               CommentRecord.FIELDS, fmt)

        self._completed_path = os.path.join(out_dir, COMPLETED_FILE)
        self._error_path = os.path.join(out_dir, ERROR_FILE)
        self.completed: Set[int] = self._load_completed() if resume else set()
        if self.completed:
            log.info("이미 수집된 작품 %d개 — 재실행 시 건너뜁니다", len(self.completed))

    def _load_completed(self) -> Set[int]:
        if not os.path.exists(self._completed_path):
            return set()
        done: Set[int] = set()
        try:
            with open(self._completed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        done.add(int(line))
        except Exception as exc:
            log.warning("완료 목록 로드 실패: %s", exc)
        return done

    def is_done(self, novel_id: int) -> bool:
        return int(novel_id) in self.completed

    def mark_done(self, novel_id: int) -> None:
        novel_id = int(novel_id)
        if novel_id in self.completed:
            return
        self.completed.add(novel_id)
        with open(self._completed_path, "a", encoding="utf-8") as f:
            f.write("%d\n" % novel_id)

    def log_errors(self, novel_id: Optional[int], errors: List[str]) -> None:
        if not errors:
            return
        with open(self._error_path, "a", encoding="utf-8") as f:
            for err in errors:
                f.write("[novel=%s] %s\n" % (novel_id, err))

    def write_result(self, result: Any) -> None:
        """NovelResult 하나를 세 테이블에 반영하고 완료 처리한다."""
        self.log_errors(result.novel.novel_id if result.novel else None, result.errors)
        if result.novel is None:
            return
        self.novels.write([result.novel.to_dict()])
        self.episodes.write(e.to_dict() for e in result.episodes)
        self.comments.write(c.to_dict() for c in result.comments)
        self.mark_done(result.novel.novel_id)

    def summary(self) -> Dict[str, int]:
        return {
            "novels": self.novels.count,
            "episodes": self.episodes.count,
            "comments": self.comments.count,
            "completed_total": len(self.completed),
        }

    def close(self) -> None:
        for table in (self.novels, self.episodes, self.comments):
            table.close()

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
