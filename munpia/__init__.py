# -*- coding: utf-8 -*-
"""문피아 웹소설 데이터 수집/전처리 파이프라인.

독자 이탈률 추정 및 고정 팬층 반응 분석용 ML 학습 데이터셋을 만든다.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .client import MunpiaClient, MunpiaAPIError, PermissionRequired, NovelUnavailable
from .crawler import MunpiaCrawler
from .storage import DatasetWriter

__all__ = [
    "MunpiaClient",
    "MunpiaAPIError",
    "PermissionRequired",
    "NovelUnavailable",
    "MunpiaCrawler",
    "DatasetWriter",
]
