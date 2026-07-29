#!/usr/bin/env python3
"""
Шаг 1 из docs/e1-debug-A-C.md: измерить, ничего не меняя.

    python scripts/diagnose_ac.py --kernel ./linux

Прогоняет четыре различающих теста из разбора — по одному на гипотезу H1-H4.
Ничего не чинит и ничего не подкручивает: только печатает числа и говорит,
какая гипотеза подтвердилась. Правки (шаги 2-5) осознанно оставлены человеку.

  H1  потолок экзамена ниже стоимости заметки   ->  база vs L(N)
  H2  модель уже знает RCU                       ->  биты/токен на 4 текстах
  H3  заметка написана в дорогом стиле           ->  3 стилевых варианта
  H4  экзамен смоука течёт                       ->  exam_leakage(EXAM, CHUNK)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kstudy._compat import enable_utf8_console  # noqa: E402
from kstudy.metrics import exam_baseline, score_note_on_exam  # noqa: E402
from kstudy.notes import exam_leakage  # noqa: E402
from kstudy.scoring import HFScorer  # noqa: E402

# Материал смоука берём как есть — диагностируем именно его.
from smoke_gpu import CHUNK, EXAM, NOTES  # noqa: E402

# --------------------------------------------------------------------------
# H3: одно и то же содержание в трёх стилях.
# Содержание строго совпадает; меняется только форма изложения.
# --------------------------------------------------------------------------

STYLES = {
    "телеграфный": NOTES["good"],
    "проза": (
        "A grace period is the interval during which every CPU passes through "
        "at least one quiescent state. A CPU is in a quiescent state whenever "
        "it is provably outside an RCU read-side critical section, which "
        "happens when it switches context, sits in the idle loop, or returns "
        "to user space. Once the grace period ends, no reader can still be "
        "holding the pre-update version, so it is safe to reclaim it. The "
        "function synchronize_rcu blocks the updater until that point, whereas "
        "call_rcu registers a callback and lets the updater carry on at once."
    ),
    "список": (
        "- Grace period: the interval in which every CPU passes through at "
        "least one quiescent state.\n"
        "- Quiescent state: any moment a CPU is provably outside an RCU "
        "read-side critical section.\n"
        "- Examples of quiescent states: a context switch, the idle loop, a "
        "return to user space.\n"
        "- After the grace period, no reader holds the pre-update version, so "
        "reclaiming it is safe.\n"
        "- synchronize_rcu blocks the updater until the grace period ends.\n"
        "- call_rcu registers a callback instead, so the updater continues "
        "immediately."
    ),
}

# H2, текст 2: обычная английская проза, ничего технического.
PLAIN_PROSE = (
    "The market had been busy since early morning, and by the time she "
    "arrived most of the good fruit was already gone. She walked slowly "
    "between the stalls, stopping now and then to look at something she had "
    "no intention of buying. An old man sold flowers near the entrance, and "
    "he nodded at her the way he did every week, without ever asking her "
    "name. She bought bread, a bag of apples, and a small jar of honey that "
    "cost more than she wanted to pay. On the way home the sky turned grey "
    "and it began to rain, so she walked faster, holding the bread against "
    "her coat to keep it dry."
)


def head_tokens(scorer: HFScorer, text: str, n: int) -> str:
    """Обрезает текст примерно до n токенов — чтобы сравнивать сравнимое."""
    ids = scorer._ids(text)
    if len(ids) <= n:
        return text
    return scorer.tokenizer.decode(ids[:n])


def hr(title: str) -> None:
    print("\n" + "─" * 74)
    print(title)
    print("─" * 74)


def main() -> int:
    enable_utf8_console()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kernel", default="./linux", help="дерево из fetch_corpus.py")
    args = ap.parse_args()

    print(f"Загружаю {args.model} на {args.device} ({args.dtype})…")
    scorer = HFScorer(args.model, device=args.device, dtype=args.dtype)

    verdicts: list[str] = []

    # ------------------------------------------------------------------ H4
    # Первым: не требует модели вообще, три строки, может закрыть вопрос.
    hr("H4. Течёт ли экзамен смоука (дословные куски источника в ответах)")
    leak = exam_leakage(EXAM, CHUNK)
    print(f"  exam_leakage(EXAM, CHUNK) = {leak:.1%}")
    print("  порог run_e1.py: чанк выбрасывается при > 25 %")
    if leak > 0.25:
        print("  ✗ H4 ПОДТВЕРЖДЕНА: экзамен течёт, копия выигрывает механически")
        verdicts.append("H4 подтверждена: утечка экзамена")
    else:
        print("  ✓ H4 отклонена: ответы действительно перефразированы")

    # ------------------------------------------------------------------ H1
    hr("H1. Потолок экзамена против стоимости заметки")
    base_bits, base_toks = exam_baseline(scorer, EXAM)
    l_note = scorer.score(NOTES["good"])
    l_copy = scorer.score(NOTES["copy"])
    ratio = base_bits / l_note.bits if l_note.bits else 0.0

    print(f"  база экзамена  Σ L(A|Q) = {base_bits:8.1f} бит "
          f"({len(EXAM)} вопр. / {base_toks} ток)")
    print(f"  L(N) good               = {l_note.bits:8.1f} бит ({l_note.n_tokens} ток)")
    print(f"  L(N) copy               = {l_copy.bits:8.1f} бит ({l_copy.n_tokens} ток)")
    print(f"  база / L(N_good)        = {ratio:8.2f}×   (нужно ≥ 3, разбор просит)")
    print()
    print(f"  ПОТОЛОК выигрыша = база − L(N) = {base_bits - l_note.bits:+.1f} бит")
    print("  (это максимум при recall = 100 %, недостижимый идеал)")

    if base_bits - l_note.bits <= 0:
        print("  ✗ H1 ПОДТВЕРЖДЕНА: положительный выигрыш арифметически невозможен")
        verdicts.append("H1 подтверждена: потолок ниже стоимости заметки")
    elif ratio < 2.0:
        print("  ✗ H1 ПОДТВЕРЖДЕНА (база < 2·L(N)): запас есть, но исчезающе малый")
        verdicts.append("H1 подтверждена: база < 2·L(N)")
    elif ratio < 3.0:
        print("  ~ H1 частично: 2 ≤ база/L(N) < 3, запас мал")
        verdicts.append("H1 частично: база/L(N) < 3")
    else:
        print("  ✓ H1 отклонена: потолка хватает")

    # ------------------------------------------------------------------ H2
    hr("H2. Знает ли модель материал (абсолютные биты/токен)")
    doc = (Path(args.kernel) / "Documentation/RCU/whatisRCU.rst").read_text(
        encoding="utf-8", errors="replace"
    )
    # Берём середину файла: начало — оглавление и заголовки, оно нерепрезентативно.
    doc_excerpt = doc[len(doc) // 3 : len(doc) // 3 + 6000]
    own_code = (
        Path(__file__).resolve().parent.parent / "kstudy" / "scoring.py"
    ).read_text(encoding="utf-8", errors="replace")

    texts = {
        "Documentation/RCU": doc_excerpt,
        "обычная проза": PLAIN_PROSE,
        "код ядра kernel/rcu": (
            Path(args.kernel) / "kernel/rcu/tree.c"
        ).read_text(encoding="utf-8", errors="replace")[20000:26000],
        "свой код (не в претрейне)": own_code,
        "CHUNK из смоука": CHUNK,
    }

    # Длина влияет на биты/токен: у длинного текста хвост дешевле. Режем всё
    # под один размер, иначе сравнение ничего не значит.
    N = 200
    print(f"  все тексты обрезаны до ~{N} токенов — иначе длина искажает б/т\n")
    print(f"  {'текст':28} {'ток':>4} {'бит':>9} {'б/т':>7}")
    bpt = {}
    for name, text in texts.items():
        s = scorer.score(head_tokens(scorer, text, N))
        bpt[name] = s.bits_per_token
        print(f"  {name:28} {s.n_tokens:4d} {s.bits:9.1f} {s.bits_per_token:7.2f}")

    print()
    print("  ориентир разбора: обычная проза ~3–4 б/т; по-настоящему новый")
    print("  технический текст должен давать 5–7")
    rcu_bpt = bpt["Documentation/RCU"]
    prose_bpt = bpt["обычная проза"]
    unseen_bpt = bpt["свой код (не в претрейне)"]
    print(f"  RCU / проза            = {rcu_bpt / prose_bpt:5.2f}×")
    print(f"  RCU / заведомо новый   = {rcu_bpt / unseen_bpt:5.2f}×")

    if rcu_bpt <= prose_bpt * 1.15:
        print("  ✗ H2 ПОДТВЕРЖДЕНА: RCU не дороже обычной прозы — материал знаком")
        verdicts.append("H2 подтверждена: корпус в претрейне")
    elif rcu_bpt < unseen_bpt * 0.7:
        print("  ~ H2 частично: RCU заметно дешевле заведомо нового текста")
        verdicts.append("H2 частично: RCU дешевле нового текста")
    else:
        print("  ✓ H2 отклонена: RCU стоит как незнакомый технический текст")

    # ------------------------------------------------------------------ H3
    hr("H3. Стиль заметки: что дороже на токен")
    print(f"  {'стиль':14} {'ток':>4} {'бит':>9} {'б/т':>7}")
    st = {}
    for name, text in STYLES.items():
        s = scorer.score(text)
        st[name] = s
        print(f"  {name:14} {s.n_tokens:4d} {s.bits:9.1f} {s.bits_per_token:7.2f}")

    tele = st["телеграфный"].bits_per_token
    prose = st["проза"].bits_per_token
    print()
    print(f"  телеграфный / проза = {tele / prose:.2f}×   "
          f"(разбор ожидает 1.3–1.6×)")
    if tele > prose * 1.10:
        print("  ✗ H3 ПОДТВЕРЖДЕНА: эталонная заметка написана в дорогом стиле")
        verdicts.append("H3 подтверждена: телеграфный стиль дороже прозы")
    else:
        print("  ✓ H3 отклонена: стиль заметки цену почти не двигает")

    # ----------------------------------------------------------------- H3b
    # H3 подтверждена в битах НА ТОКЕН — но правка из шага 3 (переписать
    # NOTE_PROMPT на прозу) меняет не б/т, а task_gain, а тот вычитает L(N)
    # в АБСОЛЮТНЫХ битах. Проза длиннее, поэтому суммарно может выйти дороже
    # телеграфа, даже будучи дешевле на токен. Прежде чем править промпт,
    # надо посмотреть, что победит на самом экзамене.
    hr("H3b. Те же три стиля, но прогнанные через экзамен (решает шаг 3)")
    base = exam_baseline(scorer, EXAM)
    print(f"  база Σ L(A|Q) = {base[0]:.1f} бит ({len(EXAM)} вопросов)\n")
    print(f"  {'стиль':14} {'ток':>4} {'L(N)':>8} {'экономия':>9} {'recall':>7} "
          f"{'выигрыш':>9} {'lev':>6} {'head':>6}")
    gains = {}
    for name, text in STYLES.items():
        m = score_note_on_exam(scorer, text, EXAM, kind=name, cached_baseline=base)
        gains[name] = m
        print(f"  {name:14} {m.n_tokens_note:4d} {m.L_note_bits:8.1f} "
              f"{m.savings_bits:9.1f} {m.recall:7.1%} {m.task_gain_bits:9.1f} "
              f"{m.leverage:6.2f} {m.headroom:6.2f}")

    best = max(gains, key=lambda k: gains[k].task_gain_bits)
    print()
    print(f"  лучший по выигрышу: {best}")
    if best == "телеграфный":
        print("  ⇒ шаг 3 (проза в NOTE_PROMPT) НЕ оправдан: на токен проза дешевле,")
        print("    но она длиннее, и метрика платит за суммарные биты.")
        verdicts.append("шаг 3 не оправдан: телеграф выигрывает по task_gain")
    else:
        print(f"  ⇒ шаг 3 оправдан: {best} бьёт телеграфный стиль по выигрышу")
        verdicts.append(f"шаг 3 оправдан: лучший стиль — {best}")

    # ---------------------------------------------------------------- итог
    hr("ИТОГ")
    if verdicts:
        for v in verdicts:
            print(f"  • {v}")
    else:
        print("  ни одна из H1–H4 не подтвердилась — смотреть H5 (λ) и §5 разбора")
    print("\n  Правки (шаги 2–5 разбора) намеренно НЕ применены: сначала числа.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
