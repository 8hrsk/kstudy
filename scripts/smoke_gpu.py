#!/usr/bin/env python3
"""
Дымовой тест на настоящей модели. Запускать ПЕРВЫМ на машине с 3060.

    python scripts/smoke_gpu.py --model Qwen/Qwen3-1.7B

Занимает минуты. Проверяет то, чего не может проверить стенд на CPU, и прежде
всего — предусловие всей Э1.

ПРОВЕРКА A — главная. Разрыв в битах на токен.
Вся MDL-машинерия держится на том, что модель считает гладкую осмысленную
заметку СУЩЕСТВЕННО дешевле плотного технического источника. Если разрыва нет,
цена заметки не отличает сжатие от копирования, и метрика не различает ничего.
Это надо узнать за десять минут, а не за месяц.

Если A провалилась — не чинить остальное. Варианты: взять модель побольше,
взять модель, менее знакомую с доменом, либо признать, что для этого корпуса
подход не работает, и это честный отрицательный результат.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kstudy._compat import enable_utf8_console  # noqa: E402
from kstudy.metrics import (  # noqa: E402
    QA,
    exam_baseline,
    score_note,
    score_note_on_exam,
)
from kstudy.scoring import HFScorer  # noqa: E402

# --------------------------------------------------------------------------
# Материал. Взято из Documentation/RCU по смыслу, переписано, чтобы модель
# не могла ответить дословным воспроизведением заученного файла.
# --------------------------------------------------------------------------

CHUNK = """A grace period is an interval of time during which every CPU has
passed through at least one quiescent state. A quiescent state is any point at
which the CPU is guaranteed not to be inside an RCU read-side critical section;
a context switch, an idle loop iteration, and a return to user space all
qualify. Once a grace period has elapsed, every reader that could possibly have
held a reference to the pre-update version of the data has finished, so the
updater may reclaim the old version safely. synchronize_rcu blocks the calling
updater until a grace period completes; call_rcu instead registers a callback
that runs after the grace period, letting the updater continue immediately."""

NOTES = {
    "good": (
        "Grace period = interval in which every CPU hits a quiescent state at "
        "least once. Quiescent = provably outside an RCU read-side section: "
        "context switch, idle, return to userspace. After it, no reader can "
        "hold the pre-update version, so reclaim is safe. synchronize_rcu "
        "blocks the updater; call_rcu registers a callback and returns at once."
    ),
    "copy": CHUNK,
    "water": (
        "This passage covers an important kernel synchronisation concept. "
        "Understanding it properly is valuable, since these mechanisms matter "
        "a great deal for both correctness and performance in real systems, "
        "and they come up frequently in practice."
    ),
    "unrelated": (
        "The completely fair scheduler keeps runnable tasks in per-CPU run "
        "queues ordered by virtual runtime, always picking the task with the "
        "smallest vruntime. Load balancing migrates tasks between queues once "
        "the imbalance passes a threshold."
    ),
}

# Ответы НАМЕРЕННО перефразированы прочь от формулировок источника. Иначе
# экзамен вознаграждает дословное запоминание, и копия выигрывает механически
# (это ровно то, что поймал стенд на CPU — см. tests/test_metrics.py).
EXAM = [
    QA(
        "What has to happen on every CPU before a grace period is considered over?",
        "Each processor must reach a point where it definitely is not running "
        "inside a reader section at least once.",
    ),
    QA(
        "Why is it safe to release the superseded copy afterwards?",
        "Nobody can still be looking at it, because all readers that might "
        "have grabbed it have already finished.",
    ),
    QA(
        "How do the blocking and non-blocking reclamation calls differ?",
        "One suspends the writer until it is over, the other hands the cleanup "
        "to a deferred routine so the writer keeps going.",
    ),
    QA(
        "Give an example of a moment that counts as a quiescent state.",
        "Switching to another task, sitting in the idle loop, or crossing back "
        "into user mode.",
    ),
]


def hr(title: str = "") -> None:
    print("\n" + "─" * 72)
    if title:
        print(title)
        print("─" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--min-gap",
        type=float,
        default=1.25,
        help="во сколько раз заметка должна быть дешевле источника на токен",
    )
    args = ap.parse_args()
    enable_utf8_console()

    print(f"Загружаю {args.model} на {args.device} ({args.dtype})…")
    scorer = HFScorer(args.model, device=args.device, dtype=args.dtype)

    failures: list[str] = []

    # ---------------------------------------------------------------- A
    hr("A. ПРЕДУСЛОВИЕ: разрыв в битах на токен")
    s_chunk = scorer.score(CHUNK)
    s_note = scorer.score(NOTES["good"])
    s_water = scorer.score(NOTES["water"])

    gap = s_chunk.bits_per_token / s_note.bits_per_token if s_note.bits_per_token else 0
    print(f"  источник      {s_chunk.bits_per_token:6.2f} б/т  ({s_chunk.n_tokens} ток)")
    print(f"  заметка       {s_note.bits_per_token:6.2f} б/т  ({s_note.n_tokens} ток)")
    print(f"  вода          {s_water.bits_per_token:6.2f} б/т  ({s_water.n_tokens} ток)")
    print(f"  РАЗРЫВ        {gap:6.2f}×   (нужно ≥ {args.min_gap})")
    if gap < args.min_gap:
        failures.append(
            f"A: разрыв {gap:.2f}× < {args.min_gap}. Цена заметки не отличает "
            f"сжатие от копирования — дальше идти бессмысленно."
        )
        print("  ✗ ПРОВАЛ — остальные проверки уже не имеют смысла")
    else:
        print("  ✓")

    # ---------------------------------------------------------------- B
    hr("B. Индукция: копия в контексте должна почти обнулять стоимость")
    self_cond = scorer.score(CHUNK, context=f"Notes:\n{CHUNK}\n\nSource:\n")
    frac = 1 - self_cond.bits / s_chunk.bits
    print(f"  L(C)={s_chunk.bits:8.1f}   L(C|C)={self_cond.bits:8.1f}   экономия {frac:.1%}")
    if frac < 0.80:
        failures.append(f"B: экономия на копии {frac:.1%} < 80 %. Похоже на ошибку "
                        f"смещения индексов в HFScorer — проверьте выравнивание logits.")
        print("  ✗ ПРОВАЛ")
    else:
        print("  ✓")

    # ---------------------------------------------------------------- C
    hr("C. Экзамен: сжатая заметка должна побеждать дословную копию")
    baseline = exam_baseline(scorer, EXAM)
    print(f"  L(ответы|вопросы) без заметки = {baseline[0]:.1f} бит / {baseline[1]} ток")
    print()
    print(f"  {'заметка':11} {'ток':>4} {'L(N)':>8} {'экономия':>9} {'recall':>7} "
          f"{'выигрыш':>9} {'lev':>6}  вердикт")
    results = {}
    for kind, text in NOTES.items():
        m = score_note_on_exam(scorer, text, EXAM, kind=kind, cached_baseline=baseline)
        results[kind] = m
        print(f"  {kind:11} {m.n_tokens_note:4d} {m.L_note_bits:8.1f} "
              f"{m.savings_bits:9.1f} {m.recall:7.1%} {m.task_gain_bits:9.1f} "
              f"{m.leverage:6.2f}  {m.verdict}")

    print()
    if results["good"].task_gain_bits <= results["copy"].task_gain_bits:
        failures.append(
            "C: копия побеждает сжатую заметку. Либо ответы экзамена слишком "
            "близки к формулировкам источника, либо разрыв из A недостаточен."
        )
        print("  ✗ ПРОВАЛ: копия побеждает")
    elif results["good"].task_gain_bits <= 0:
        failures.append("C: даже хорошая заметка не окупается (выигрыш ≤ 0).")
        print("  ✗ ПРОВАЛ: хорошая заметка не окупается")
    else:
        print("  ✓")

    # ---------------------------------------------------------------- D
    hr("D. Мусор должен отвергаться")
    ok = True
    for kind in ("water", "unrelated"):
        m = results[kind]
        good = m.verdict != "keep"
        print(f"  {kind:11} выигрыш={m.task_gain_bits:9.1f}  recall={m.recall:6.1%}  "
              f"{m.verdict}  {'✓' if good else '✗'}")
        ok &= good
    if not ok:
        failures.append("D: мусорная заметка прошла фильтр.")

    # ---------------------------------------------------------------- E
    hr("E. Поверхностная метрика — только для сведения")
    base = scorer.score(CHUNK)
    for kind in ("good", "copy"):
        m = score_note(scorer, CHUNK, NOTES[kind], kind=kind, cached_chunk_score=base)
        print(f"  {kind:11} экономия={m.savings_frac:6.1%}  выигрыш={m.mdl_gain_bits:9.1f}  "
              f"lev={m.leverage:5.2f}")
    print("  (ожидаемо копия здесь впереди — поэтому основной метрикой служит экзамен)")

    # ---------------------------------------------------------------- итог
    hr("ИТОГ")
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"\n  {len(failures)} проверок провалено — Э2 не начинать.")
        return 1
    print("  ✓ все проверки пройдены. Можно размечать сотню примеров (kstudy/rate.py)")
    print("    и считать корреляцию (kstudy/analyze.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
