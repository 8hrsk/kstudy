"""kstudy — стенд Э1: метрики обучения без эталона."""

from .scoring import Score, Scorer, HFScorer, CacheNgramScorer
from .metrics import (
    ChunkMetrics,
    NoteMetrics,
    TriageThresholds,
    calibrate_thresholds,
    score_chunk,
    score_note,
    triage,
)

__all__ = [
    "Score",
    "Scorer",
    "HFScorer",
    "CacheNgramScorer",
    "ChunkMetrics",
    "NoteMetrics",
    "TriageThresholds",
    "calibrate_thresholds",
    "score_chunk",
    "score_note",
    "triage",
]
