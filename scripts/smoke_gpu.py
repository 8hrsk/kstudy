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
    MIN_EXAM_HEADROOM,
    QA,
    chunk_is_measurable,
    exam_baseline,
    exam_headroom,
    score_note,
    score_note_on_exam,
)
from kstudy.notes import exam_leakage  # noqa: E402
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

# ПОЧЕМУ ИХ ШЕСТНАДЦАТЬ, А НЕ ЧЕТЫРЕ. Первый прогон дал базу 590.4 бита при
# стоимости заметки 571.7 — потолок выигрыша +18.7 бита. Проверка C не могла
# пройти ни при каком качестве заметки: мерялось соотношение размеров, а не
# понимание. Экономия ограничена сверху базой, поэтому база обязана быть
# заметно больше цены заметки — см. MIN_EXAM_HEADROOM в kstudy/metrics.py.
# Ответы по два-три предложения по той же причине: короткий ответ даёт мало
# бит, а потолок складывается именно из них.
#
# ПОЧЕМУ ОНИ НА ВЫВОД, А НЕ НА ИЗВЛЕЧЕНИЕ. Прогон на 33 чанках: копия
# закрывала экзамен на 47.8 % против 26.5 % у осмысленного конспекта, то есть
# была вдвое полезнее и проигрывала лишь штрафом за длину (попарно конспект
# побеждал в 55 % случаев — чуть лучше монетки). Если ответ дословно лежит в
# источнике, переписанный источник оптимален ПО ПОСТРОЕНИЮ, и экзамен меряет
# поиск, а не понимание. Отсюда вопросы на следствие, на необходимость
# условия, на контрфактический вариант и на сравнение двух примитивов.
#
# Ответы по-прежнему перефразированы прочь от формулировок источника — иначе
# экзамен вознаграждает дословное запоминание (находка стенда на CPU, см.
# tests/test_metrics.py). Утечка меряется секцией 0.
#
# Чанк небольшой, поэтому шестнадцать вопросов неизбежно ходят вокруг одних и
# тех же фактов с разных сторон. Для метрики это безвредно: и база, и экономия
# растут вместе, recall остаётся долей. Разнообразие УГЛА здесь важнее
# разнообразия фактов.
#
# Критерий, по которому судим правку, зафиксирован ДО прогона:
# docs/e1-exam-v2-criteria.md.
EXAM = [
    QA(
        "What use-after-free hazard would appear if an updater reclaimed the "
        "old version of a data structure immediately, without waiting for a "
        "grace period to elapse first?",
        "A CPU could still be inside an RCU read-side critical section "
        "holding a reference to the old version, so freeing that memory out "
        "from under it would leave the reader dereferencing freed data. "
        "Waiting for the grace period exists precisely to rule this out, so "
        "skipping it reintroduces the exact hazard the mechanism is designed "
        "to prevent.",
    ),
    QA(
        "Why isn't it enough for a single CPU to pass through a quiescent "
        "state before the updater reclaims old data, when the system has "
        "several CPUs?",
        "Some other CPU could still be a reader inside its RCU read-side "
        "critical section working with the old version, so clearing just one "
        "processor says nothing about activity happening elsewhere. Because "
        "the hazard is system-wide, a grace period has to cover every CPU "
        "passing through a quiescent state, not merely the first one that "
        "happens to look safe.",
    ),
    QA(
        "Suppose a CPU stayed inside a single RCU read-side critical section "
        "indefinitely, never taking a context switch, never entering the "
        "idle loop, and never returning to user space. What would this do to "
        "the grace period?",
        "That CPU would never reach a quiescent state, so any grace period "
        "that needs to account for it could never be declared complete. The "
        "updater would be stuck waiting indefinitely, since a grace period's "
        "completion depends on every CPU eventually producing one of the "
        "recognized quiescent-state transitions.",
    ),
    QA(
        "A developer wants the updater to keep running other work "
        "immediately after triggering reclamation, without blocking on a "
        "grace period elsewhere in the system. Should they use "
        "synchronize_rcu or call_rcu, and why?",
        "call_rcu is the right choice here because it registers a callback "
        "that performs the reclaim once the grace period completes, letting "
        "the updater continue immediately instead of waiting. synchronize_rcu "
        "would instead block the calling updater until the grace period "
        "finishes, which is exactly the delay the developer is trying to "
        "avoid.",
    ),
    QA(
        "Contrast what the updater is doing while the grace period is still "
        "in progress, under synchronize_rcu versus call_rcu.",
        "Under synchronize_rcu, the updater is blocked, suspended until the "
        "grace period has actually completed across every CPU. Under "
        "call_rcu, the updater never blocks at all; instead the registered "
        "callback executes later, independently, once the same grace period "
        "eventually elapses.",
    ),
    QA(
        "If the definition of a quiescent state were loosened to include any "
        "CPU finishing a single instruction, rather than a context switch, "
        "idle-loop iteration, or return to user space, what would go wrong?",
        "A CPU can finish an instruction while still deep inside an RCU "
        "read-side critical section, so that looser definition would not "
        "actually guarantee the reader has left. A grace period built on "
        "such weak evidence could complete while a reader is still holding a "
        "reference to the old version, letting the updater reclaim memory "
        "that is still in use.",
    ),
    QA(
        "Why do a context switch, an idle-loop iteration, and a return to "
        "user space all count as the same thing, a quiescent state, even "
        "though they are three different events?",
        "None of the three can occur while a CPU is still inside an RCU "
        "read-side critical section, so whichever one happens first is proof "
        "that the CPU has left such a section. They arise for unrelated "
        "reasons, rescheduling, having no work, or exiting to a process, but "
        "they agree on the one guarantee a quiescent state needs to provide.",
    ),
    QA(
        "An updater needs certainty, at the moment a function call returns, "
        "that reclaiming old data is now safe. Does synchronize_rcu or "
        "call_rcu give that guarantee directly, and what must be true "
        "internally before it can return?",
        "synchronize_rcu gives this guarantee directly, since it does not "
        "return control to the updater until the grace period has genuinely "
        "completed. Internally, this requires that every CPU has by then "
        "passed through a quiescent state, confirming no reader still holds "
        "a reference to the pre-update version.",
    ),
    QA(
        "Two updates happen close together on the same object, one issued "
        "through synchronize_rcu and one through call_rcu. Which updater is "
        "more likely to have already resumed other work by the time the "
        "call_rcu callback actually runs?",
        "The synchronize_rcu updater would already be back to normal "
        "execution, because its block only lasted as long as its own grace "
        "period took to complete. The call_rcu updater was never blocked to "
        "begin with, so its callback fires independently later, with no "
        "particular relationship to what the other updater is doing at that "
        "point.",
    ),
    QA(
        "What is the practical cost of always using synchronize_rcu instead "
        "of call_rcu, even in a code path where the updater has other useful "
        "work available?",
        "The updater would sit blocked for however long the grace period "
        "takes, which can be a real delay if some CPU is slow to reach a "
        "quiescent state. That cost is avoidable, since call_rcu gives the "
        "same eventual safety without ever stalling the updater, by "
        "deferring the reclaim to a callback instead.",
    ),
    QA(
        "Explain why concluding that the old version can be reclaimed safely "
        "depends on combining the definition of a quiescent state with the "
        "definition of a grace period, rather than either alone.",
        "A quiescent state on its own only shows that one CPU has stepped "
        "outside its RCU read-side critical section, saying nothing about "
        "the rest of the machine. It is the grace period's requirement that "
        "every CPU pass through such a state that turns that local "
        "observation into a system-wide guarantee strong enough to let the "
        "updater reclaim the old version.",
    ),
    QA(
        "Imagine a broken implementation of call_rcu where the callback ran "
        "before the grace period had actually elapsed. What would go wrong?",
        "The callback's job is to reclaim the old version, so running it "
        "early would mean doing so while some CPU might still be a reader "
        "inside its RCU read-side critical section referencing that data. "
        "That reintroduces the exact use-after-free hazard the grace period "
        "exists to prevent, since the callback's safety depends entirely on "
        "the grace period having genuinely completed first.",
    ),
    QA(
        "An idle CPU doesn't sound like it has anything to do with data "
        "access. Why does an idle-loop iteration still count as a quiescent "
        "state?",
        "A CPU running the idle loop cannot, at the same moment, be a reader "
        "inside an RCU read-side critical section, so reaching the idle loop "
        "proves it has left any such section behind. What matters is not "
        "idleness itself but that this state is simply incompatible with "
        "still holding a reference to the old version.",
    ),
    QA(
        "A developer proposes replacing grace-period tracking with a fixed "
        "timer, say reclaiming old data automatically every 10 milliseconds "
        "regardless of what any CPU is doing. What's wrong with that compared "
        "to how synchronize_rcu and call_rcu actually determine when "
        "reclamation is safe?",
        "A fixed timer has no connection to what any given CPU is actually "
        "executing, so it could easily fire while a reader is still "
        "legitimately inside a long RCU read-side critical section, causing "
        "the updater to reclaim memory still in use. The grace-period "
        "approach avoids this by tying safety to each CPU actually reaching "
        "a quiescent state, rather than to an arbitrary clock interval.",
    ),
    QA(
        "What must be true about a CPU's execution model for the whole "
        "grace-period mechanism to work at all?",
        "Every CPU must be capable of eventually leaving any RCU read-side "
        "critical section it enters, whether through a context switch, an "
        "idle-loop iteration, or a return to user space, since these are the "
        "only transitions that count as a quiescent state. A CPU that could "
        "remain in one critical section forever, with none of those "
        "transitions ever occurring, would give the grace period no way to "
        "certify that CPU as done, and it would never complete.",
    ),
    QA(
        "If a kernel only supported synchronize_rcu and not call_rcu, what "
        "capability would updaters lose, and what would code that needed to "
        "avoid blocking have to do instead?",
        "Losing call_rcu would force every updater to block until its grace "
        "period completes, even in places where continuing immediately would "
        "have been preferable. Code that needed to avoid that stall would "
        "have to build its own deferred-callback mechanism by hand, "
        "essentially reconstructing what call_rcu already provides for "
        "registering work to run once the grace period elapses.",
    ),
]


def hr(title: str = "") -> None:
    print("\n" + "─" * 72)
    if title:
        print(title)
        print("─" * 72)


def main() -> int:
    # До ArgumentParser: справка и ошибки argparse содержат кириллицу и
    # печатаются ещё внутри parse_args().
    enable_utf8_console()

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

    failures: list[str] = []

    # ---------------------------------------------------------------- 0
    # Модель для этого не нужна, поэтому первым: если экзамен течёт, все
    # последующие числа меряют запоминание, а не понимание.
    #
    # Раньше этой проверки здесь не было: run_e1.py утечку мерил, а смоук —
    # нет, хотя именно он объявлен точкой невозврата. Ответы были объявлены
    # перефразированными и ни разу не измерены.
    hr("0. ГИГИЕНА: не течёт ли собственный экзамен смоука")
    leak = exam_leakage(EXAM, CHUNK)
    print(f"  вопросов        {len(EXAM)}")
    print(f"  exam_leakage    {leak:6.1%}   (порог run_e1.py: > 25 % — чанк выбросить)")
    if leak > 0.25:
        failures.append(
            f"0: экзамен течёт ({leak:.1%}). Ответы слишком близки к тексту "
            "источника — копия будет выигрывать посимвольно, а не по смыслу."
        )
        print("  ✗ ПРОВАЛ — экзамен надо переписать, остальное меряет запоминание")
    else:
        print("  ✓")

    print(f"\nЗагружаю {args.model} на {args.device} ({args.dtype})…")
    scorer = HFScorer(args.model, device=args.device, dtype=args.dtype)

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

    # ---------------------------------------------------------------- A2
    # A мерит биты НА ТОКЕН, а task_gain вычитает L(N) в АБСОЛЮТНЫХ битах —
    # величина, которой в A нет вовсе. Это не одно и то же: конспект несёт то
    # же содержание меньшим числом токенов, поэтому его цена за токен растёт
    # почти по определению сжатия, тогда как суммарная цена падает.
    #
    # Поэтому A2 печатается РЯДОМ с A, а не вместо неё. Порог A не тронут и
    # её результат не отменён: если A провалилась, а A2 прошла, это значит,
    # что предусловие сформулировано не в той величине — и такое решение
    # принимает человек, а не скрипт. См. docs/e1-debug-A-C.md.
    hr("A2. То же предусловие, но в суммарных битах — их и потребляет метрика")
    ratio_total = s_note.bits / s_chunk.bits if s_chunk.bits else 0.0
    print(f"  L(источник)   {s_chunk.bits:8.1f} бит  ({s_chunk.n_tokens} ток)")
    print(f"  L(заметка)    {s_note.bits:8.1f} бит  ({s_note.n_tokens} ток)")
    print(f"  L(N) / L(C)   {ratio_total:8.2f}    (нужно < 1: заметка дешевле источника)")
    if ratio_total >= 1.0:
        failures.append(
            f"A2: заметка суммарно дороже источника ({ratio_total:.2f}). "
            "Копия будет дешевле осмысленной заметки — вот это действительно "
            "убивает метрику."
        )
        print("  ✗ ПРОВАЛ")
    else:
        print("  ✓")
    if gap < args.min_gap and ratio_total < 1.0:
        print("  ! A и A2 расходятся: на токен заметка дороже, суммарно дешевле.")
        print("    Это ожидаемо для всякого сжатия и означает, что A задаёт")
        print("    предусловие не в той величине. Решение — за человеком.")

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

    # ---------------------------------------------------------------- C0
    # Гейт пригодности. Ставится ДО C, потому что при низком потолке провал C
    # означает лишь короткий экзамен, а не плохую заметку — ровно та ошибка,
    # на которой встал первый прогон (база 590.4 против L(N) 571.7).
    hr("C0. ПРИГОДНОСТЬ: хватает ли потолка, чтобы C вообще что-то мерила")
    baseline = exam_baseline(scorer, EXAM)
    base_bits, base_toks = baseline
    head = exam_headroom(base_bits, s_note.bits)
    print(f"  база Σ L(A|Q)      {base_bits:9.1f} бит  ({len(EXAM)} вопр. / {base_toks} ток)")
    print(f"  L(N) good          {s_note.bits:9.1f} бит")
    print(f"  headroom           {head:9.2f}×   (нужно ≥ {MIN_EXAM_HEADROOM})")
    print(f"  потолок выигрыша   {base_bits - s_note.bits:+9.1f} бит  (при recall = 100 %)")
    if not chunk_is_measurable(base_bits, s_note.bits):
        failures.append(
            f"C0: headroom {head:.2f}× < {MIN_EXAM_HEADROOM}. Потолок выигрыша "
            "сравним с ценой заметки — на этом экзамене C меряет соотношение "
            "размеров, а не понимание. Результат C ниже недействителен."
        )
        print("  ✗ ПРОВАЛ — C ниже считается, но её результат ничего не значит")
    else:
        print("  ✓")

    # ---------------------------------------------------------------- C
    hr("C. Экзамен: сжатая заметка должна побеждать дословную копию")
    print(f"  L(ответы|вопросы) без заметки = {base_bits:.1f} бит / {base_toks} ток")
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

    # У C два независимых условия, и раньше они сливались в одно сообщение.
    # Различать их важно: «копия побеждает» — это сломанное различение и повод
    # остановиться; «заметка не окупается» при верном порядке — это вопрос
    # величины recall, то есть совсем другой диагноз и другая правка.
    print()
    good, copy = results["good"], results["copy"]
    beats_copy = good.task_gain_bits > copy.task_gain_bits
    pays_off = good.task_gain_bits > 0

    print(f"  C.1 заметка бьёт копию     {'✓' if beats_copy else '✗'}   "
          f"({good.task_gain_bits:.1f} против {copy.task_gain_bits:.1f})")
    print(f"  C.2 заметка окупается      {'✓' if pays_off else '✗'}   "
          f"(выигрыш {good.task_gain_bits:+.1f} бит)")
    if not pays_off:
        need = good.L_note_bits / good.L_answers_bits
        print(f"      чтобы окупиться, нужен recall > {need:.1%}; "
              f"сейчас {good.recall:.1%}")

    if not beats_copy:
        failures.append(
            "C.1: копия побеждает сжатую заметку. Различение внутри не-мусора "
            "не работает — это блокер, а не вопрос настройки."
        )
        print("  ✗ ПРОВАЛ: копия побеждает")
    if not pays_off:
        failures.append(
            f"C.2: хорошая заметка не окупается (выигрыш "
            f"{good.task_gain_bits:+.1f}). Порядок верный, но recall "
            f"{good.recall:.1%} ниже точки окупаемости {need:.1%}."
        )
        print("  ✗ ПРОВАЛ: хорошая заметка не окупается")
    if beats_copy and pays_off:
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
