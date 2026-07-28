"""
Анализ Э1: коррелирует ли метрика с человеческой оценкой.

Это и есть решающий вопрос этапа. Если MDL-выигрыш не коррелирует с вашей
оценкой «полезная заметка / вода» — вся конструкция стоит на песке, и хорошо
бы узнать об этом сейчас, а не через месяц генерации.

Порог для решения (задан заранее, чтобы не подгонять после того, как увидели
числа):

    Spearman ρ ≥ 0.5  и  AUC ≥ 0.75   — идти в Э2
    ρ 0.3-0.5                          — метрику чинить, в Э2 не идти
    ρ < 0.3                            — гипотеза не подтвердилась

Пишется без scipy — нужен только numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._compat import enable_utf8_console


# --------------------------------------------------------------------------
# Статистика без scipy
# --------------------------------------------------------------------------


def _rank(x: Sequence[float]) -> np.ndarray:
    """Ранги со средним для связок."""
    a = np.asarray(x, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # усреднение по группам одинаковых значений
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def spearman_ci(
    x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Бутстрэп-интервал. На сотне примеров он широкий — это нормально и полезно видеть."""
    rng = np.random.default_rng(seed)
    n = len(x)
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(xs[idx], ys[idx])
        if not math.isnan(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    ROC AUC через статистику Манна-Уитни. labels: 1 — полезная заметка, 0 — нет.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y == 1].sum() - pos * (pos - 1) / 2) / (pos * neg))


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------

METRIC_FIELDS = [
    ("task_gain_bits", "MDL-выигрыш на экзамене"),
    ("leverage", "польза на бит заметки"),
    ("recall", "доля закрытого экзамена"),
    ("savings_bits", "сырая экономия"),
    ("mdl_gain_bits", "поверхностный MDL"),
]


@dataclass
class MetricReport:
    field: str
    label: str
    rho: float
    lo: float
    hi: float
    auc: float
    n: int

    @property
    def verdict(self) -> str:
        if math.isnan(self.rho):
            return "нет данных"
        if self.rho >= 0.5 and self.auc >= 0.75:
            return "ГОДЕН"
        if self.rho >= 0.3:
            return "слабо"
        return "не работает"


def analyse(rows: Sequence[dict], rating_key: str = "rating") -> list[MetricReport]:
    """
    rows — объединённые записи: метрики заметки + поле rating (1..5 от человека).
    Бинарная метка для AUC: rating >= 4 считается полезной заметкой.
    """
    rated = [r for r in rows if r.get(rating_key) is not None]
    reports = []
    for field, label in METRIC_FIELDS:
        vals = [r[field] for r in rated if field in r]
        rate = [r[rating_key] for r in rated if field in r]
        if len(vals) < 5:
            continue
        lo, hi = spearman_ci(vals, rate)
        reports.append(
            MetricReport(
                field=field,
                label=label,
                rho=spearman(vals, rate),
                lo=lo,
                hi=hi,
                auc=auc(vals, [1 if x >= 4 else 0 for x in rate]),
                n=len(vals),
            )
        )
    return reports


def control_separation(rows: Sequence[dict], field: str = "task_gain_bits") -> dict:
    """
    Разделяет ли метрика заведомые контроли? Это проверка без всякой разметки,
    и её стоит смотреть первой: если модельные заметки не отделяются от воды
    и копий, размечать сотню примеров рано.
    """
    out: dict[str, dict] = {}
    for kind in ("model", "copy", "water", "unrelated"):
        vals = [r[field] for r in rows if r.get("kind") == kind and field in r]
        if vals:
            a = np.asarray(vals, float)
            out[kind] = {
                "n": len(vals),
                "median": float(np.median(a)),
                "mean": float(a.mean()),
                "frac_kept": float(np.mean([v > 0 for v in vals])),
            }
    return out


def print_report(rows: Sequence[dict], rating_key: str = "rating") -> bool:
    """Печатает отчёт. Возвращает True, если можно идти в Э2."""
    print("=" * 72)
    print("Э1 — отчёт")
    print("=" * 72)

    sep = control_separation(rows)
    if sep:
        print("\nКонтроли (без разметки, поле task_gain_bits):")
        print(f"  {'тип':11} {'n':>4} {'медиана':>10} {'среднее':>10} {'доля>0':>8}")
        for kind, st in sep.items():
            print(f"  {kind:11} {st['n']:4d} {st['median']:10.1f} "
                  f"{st['mean']:10.1f} {st['frac_kept']:7.0%}")
        m, w = sep.get("model"), sep.get("water")
        if m and w:
            ok = m["median"] > w["median"]
            print(f"  модельные заметки {'выше' if ok else 'НЕ выше'} воды "
                  f"{'✓' if ok else '✗'}")

    reports = analyse(rows, rating_key)
    if not reports:
        print("\nРазмеченных примеров нет — запустите kstudy/rate.py")
        return False

    print(f"\nКорреляция с оценкой человека (n={reports[0].n}):")
    print(f"  {'метрика':26} {'ρ':>6} {'95% ДИ':>16} {'AUC':>6}  вердикт")
    for r in reports:
        ci = f"[{r.lo:.2f}, {r.hi:.2f}]"
        print(f"  {r.label:26} {r.rho:6.2f} {ci:>16} {r.auc:6.2f}  {r.verdict}")

    best = max(reports, key=lambda r: (-1 if math.isnan(r.rho) else r.rho))
    print(f"\nЛучшая метрика: {best.label} (ρ={best.rho:.2f})")

    go = best.rho >= 0.5 and best.auc >= 0.75
    print("\n" + ("→ Порог пройден, можно в Э2." if go else
                  "→ Порог НЕ пройден. В Э2 не идти: метрика не отражает пользу."))
    if not go and best.rho >= 0.3:
        print("  Что попробовать: больше вопросов в экзамене; строже перефразирование;")
        print("  модель побольше; проверить exam_leakage — не течёт ли источник в ответы.")
    return go


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_jsonl", help="выход run_e1.py")
    ap.add_argument("--ratings", help="выход kstudy/rate.py")
    a = ap.parse_args()
    enable_utf8_console()

    rows = [json.loads(l) for l in Path(a.metrics_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.ratings:
        rated = {
            r["note_id"]: r["rating"]
            for r in (json.loads(l) for l in Path(a.ratings).read_text(encoding="utf-8").splitlines() if l.strip())
        }
        for r in rows:
            if r.get("note_id") in rated:
                r["rating"] = rated[r["note_id"]]

    return 0 if print_report(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
