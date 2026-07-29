"""
Инварианты метрик Э1. Гоняются на CacheNgramScorer — без GPU и без сети.

Смысл: до того, как тратить дни на генерацию и обучение, убедиться, что метрика
отличает полезную заметку от бесполезной. Если инварианты падают на игрушечной
модели, на трансформере они сами не выпрямятся.

ЧТО ЭТОТ СТЕНД МОЖЕТ ПРОВЕРИТЬ:
  - порядок величин и знак ЭКОНОМИИ (savings)
  - что вода и нерелевантный текст отвергаются
  - что экзаменационная метрика ставит сжатую заметку выше дословной копии
  - абляции (λ=0 и т. п.)

ЧЕГО НЕ МОЖЕТ:
  - знак строгого MDL-выигрыша на поверхности текста. У слабой LM стоимость
    в битах на токен почти одинакова для гладкой прозы и для плотного
    технического текста (здесь: 12.3 против 11.0 б/т). У настоящей модели
    разница кратная, и знак держится именно на ней. Эта проверка вынесена
    в scripts/smoke_gpu.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kstudy.scoring import CacheNgramScorer  # noqa: E402
from kstudy.metrics import (  # noqa: E402
    MIN_EXAM_HEADROOM,
    QA,
    calibrate_thresholds,
    chunk_is_measurable,
    exam_baseline,
    exam_headroom,
    score_chunk,
    score_note,
    score_note_on_exam,
    triage,
)

# --------------------------------------------------------------------------
# Игрушечный корпус в стиле Documentation/RCU
# --------------------------------------------------------------------------

CORPUS = [
    """RCU is a synchronization mechanism that allows reads to proceed
    concurrently with updates. Readers enter a read-side critical section with
    rcu_read_lock and leave it with rcu_read_unlock. Updaters publish new
    versions with rcu_assign_pointer and wait for pre-existing readers with
    synchronize_rcu before reclaiming memory.""",
    """A grace period is a time interval during which every CPU has passed
    through at least one quiescent state. Once a grace period elapses, no
    reader can still hold a reference to the old version, so the updater may
    safely free it. call_rcu defers the reclamation to a callback instead of
    blocking the updater.""",
    """The scheduler tracks runnable tasks in per-CPU run queues. The
    completely fair scheduler orders tasks by virtual runtime and picks the
    task with the smallest vruntime. Load balancing migrates tasks between
    run queues when the imbalance exceeds a threshold.""",
    """Memory barriers enforce ordering between memory operations. smp_mb
    provides a full barrier, smp_rmb orders reads, and smp_wmb orders writes.
    Without barriers the compiler and the CPU are free to reorder accesses,
    which breaks lockless algorithms in subtle ways.""",
    """The page allocator hands out physically contiguous pages using the
    buddy system. Requests are rounded up to a power of two order. The slab
    allocator sits on top and carves pages into small objects of fixed size,
    keeping per-CPU caches to avoid contention.""",
]

CHUNK = CORPUS[1]  # про grace period

NOTES = {
    "good": (
        "Grace period: interval where every CPU passed a quiescent state. "
        "After it elapses no reader holds the old version, so the updater "
        "may free it. call_rcu defers reclamation to a callback instead of "
        "blocking."
    ),
    "copy": CHUNK,
    "water": (
        "This section describes an important concept in the kernel. It is "
        "useful to understand the mechanism well, because it matters a great "
        "deal for correctness and performance in practice."
    ),
    "unrelated": (
        "The completely fair scheduler orders tasks by virtual runtime and "
        "picks the smallest vruntime. Load balancing migrates tasks between "
        "run queues."
    ),
}

# Экзамен составлен ПО чанку; отвечать надо без него.
EXAM = [
    QA("What is a grace period?",
       "A time interval during which every CPU has passed through at least "
       "one quiescent state."),
    QA("When may an updater safely free the old version?",
       "Once a grace period elapses, because no reader can still hold a "
       "reference to the old version."),
    QA("What does call_rcu do?",
       "It defers the reclamation to a callback instead of blocking the "
       "updater."),
]


def build_scorer() -> CacheNgramScorer:
    return CacheNgramScorer(order=3).fit(CORPUS)


def surface_metrics():
    s = build_scorer()
    base = s.score(CHUNK, context="")
    return {
        k: score_note(s, CHUNK, t, kind=k, cached_chunk_score=base)
        for k, t in NOTES.items()
    }


def exam_metrics(lam: float = 1.0):
    s = build_scorer()
    baseline = exam_baseline(s, EXAM)
    return {
        k: score_note_on_exam(
            s, t, EXAM, kind=k, lam=lam, cached_baseline=baseline
        )
        for k, t in NOTES.items()
    }


# ==========================================================================
# A. Поверхностная метрика: проверяем ЭКОНОМИЮ, не знак выигрыша
# ==========================================================================


def test_copy_saves_almost_everything():
    """
    Дословная копия обязана снимать почти всю стоимость чанка. Это режим
    индукционных голов; если его нет — сломан скорер, и все остальные
    числа бессмысленны.
    """
    m = surface_metrics()["copy"]
    assert m.savings_frac > 0.85, f"копия экономит лишь {m.savings_frac:.1%}"


def test_savings_ordering():
    m = surface_metrics()
    assert (
        m["copy"].savings_frac
        > m["good"].savings_frac
        > m["water"].savings_frac
        > m["unrelated"].savings_frac
    ), {k: round(v.savings_frac, 3) for k, v in m.items()}


def test_good_note_saves_substantially():
    m = surface_metrics()["good"]
    assert m.savings_frac > 0.2, f"осмысленная заметка экономит {m.savings_frac:.1%}"


def test_unrelated_note_saves_nothing():
    m = surface_metrics()["unrelated"]
    assert m.savings_frac < 0.05


def test_surface_mdl_structurally_favours_copying():
    """
    Документируем ограничение как тест, чтобы оно не потерялось: по строгому
    MDL на поверхности текста копия выигрывает у сжатой заметки. Это и есть
    причина, по которой основной метрикой сделан экзамен.
    Если этот тест однажды упадёт — поверхностную метрику можно повышать
    в статусе, но пока он держится, полагаться на неё нельзя.
    """
    m = surface_metrics()
    assert m["copy"].mdl_gain_bits > m["good"].mdl_gain_bits


def test_zero_lambda_lets_copy_win_by_a_mile():
    """Абляция: без платы за заметку копия побеждает с огромным отрывом."""
    s = build_scorer()
    base = s.score(CHUNK, context="")
    good = score_note(s, CHUNK, NOTES["good"], lam=0.0, cached_chunk_score=base)
    copy = score_note(s, CHUNK, NOTES["copy"], lam=0.0, cached_chunk_score=base)
    assert copy.mdl_gain_bits > 2 * good.mdl_gain_bits


# ==========================================================================
# B. Экзаменационная метрика — основная
#
# Что здесь можно и чего нельзя проверить на игрушке.
#
# Механизм, ради которого экзамен введён: польза заметки НАСЫЩАЕТСЯ (больше,
# чем L(A|Q), на ответах не сэкономишь), а её цена растёт с длиной. Поэтому
# копия, будучи в 2-3 раза дороже, не может быть в 2-3 раза полезнее, и
# проигрывает. Это свойство ФОРМУЛЫ, и оно проверяется арифметикой — см.
# test_saturation_*.
#
# Но на игрушечном скорере good vs copy не разделяются, и причина не в формуле:
# у слабой LM стоимость почти одинакова на токен для любого текста (замер:
# 12.3 б/т у гладкой заметки против 11.0 у плотного технического чанка). Вся
# дискриминация держится на том, что у настоящей модели этот разрыв кратный.
#
# Отсюда предусловие всего Э1, которое проверяется первым делом на 3060:
#     bits/token(заметка) должно быть заметно НИЖЕ, чем bits/token(чанк).
# Если разрыва нет — MDL-машинерия не различает, и это надо знать за 10 минут,
# а не за месяц. См. scripts/smoke_gpu.py, проверка A.
# ==========================================================================


def _synthetic(savings: float, note_bits: float, base: float = 240.0):
    """Собирает TaskNoteMetrics из готовых чисел — для проверки арифметики."""
    from kstudy.metrics import TaskNoteMetrics

    return TaskNoteMetrics(
        note_id="", chunk_id="", kind="synthetic",
        n_questions=3, n_tokens_note=1, n_tokens_answers=60,
        L_note_bits=note_bits,
        L_answers_bits=base,
        L_answers_given_note_bits=base - savings,
        lam=1.0,
    )


def test_saturation_penalises_expensive_note():
    """
    Ядро замысла. Копия чуть полезнее (180 против 150 бит экономии), но втрое
    дороже (265 против 100). Формула обязана предпочесть дешёвую заметку.
    Числа взяты из ожидаемого порядка величин для 1.7B на техническом тексте.
    """
    good = _synthetic(savings=150, note_bits=100)
    copy = _synthetic(savings=180, note_bits=265)
    assert good.task_gain_bits > 0
    assert copy.task_gain_bits < 0
    assert good.task_gain_bits > copy.task_gain_bits


def test_saturation_savings_cannot_exceed_baseline():
    """Экономия ограничена сверху стоимостью ответов — отсюда и насыщение."""
    m = _synthetic(savings=240, note_bits=100, base=240.0)
    assert m.recall == 1.0
    assert m.L_answers_given_note_bits == 0.0


def test_gain_is_monotone_in_note_cost():
    """При равной пользе более дешёвая заметка обязана выигрывать. Всегда."""
    cheap = _synthetic(savings=150, note_bits=80)
    dear = _synthetic(savings=150, note_bits=140)
    assert cheap.task_gain_bits > dear.task_gain_bits
    assert cheap.leverage > dear.leverage


def test_gain_is_monotone_in_savings():
    a = _synthetic(savings=200, note_bits=100)
    b = _synthetic(savings=120, note_bits=100)
    assert a.task_gain_bits > b.task_gain_bits


def test_verdict_boundaries():
    assert _synthetic(savings=0, note_bits=100).verdict == "reject"
    assert _synthetic(savings=50, note_bits=100).verdict == "weak"
    assert _synthetic(savings=200, note_bits=100).verdict == "keep"


def test_exam_rejects_water_and_unrelated():
    """Это игрушка проверить МОЖЕТ: мусор не должен давать экономии."""
    m = exam_metrics()
    assert m["water"].task_gain_bits < 0
    assert m["unrelated"].task_gain_bits < 0
    assert m["water"].verdict != "keep"
    assert m["unrelated"].verdict != "keep"


def test_exam_recall_ordering():
    """Осмысленная заметка закрывает экзамен лучше мусора."""
    m = exam_metrics()
    assert m["good"].recall > m["water"].recall
    assert m["good"].recall > m["unrelated"].recall


def test_exam_recall_bounded():
    for m in exam_metrics().values():
        assert m.recall <= 1.0 + 1e-9


def test_verbatim_exam_answers_reward_memorisation():
    """
    Документируем вторую находку стенда, чтобы её не потеряли при переносе
    на GPU: если ответы экзамена — дословные куски источника, копия выигрывает
    за счёт посимвольного совпадения, а не понимания. Значит генератор вопросов
    ОБЯЗАН перефразировать ответы прочь от формулировок источника.
    Здесь ответы намеренно дословные, и копия действительно побеждает.
    """
    m = exam_metrics()
    assert m["copy"].recall > m["good"].recall


# ==========================================================================
# C. Триаж
# ==========================================================================


def test_triage_splits_corpus():
    s = build_scorer()
    prior = CORPUS[0]
    ms = []
    for i, text in enumerate(CORPUS):
        m = score_chunk(s, text, prior_notes=prior)
        m.chunk_id = f"c{i}"
        ms.append(m)

    th = calibrate_thresholds(ms)
    labels = {m.chunk_id: triage(m, th) for m in ms}

    assert labels["c0"] == "known", labels
    assert len(set(labels.values())) > 1, labels


def test_delta_positive_for_self_context():
    s = build_scorer()
    m = score_chunk(s, CORPUS[0], prior_notes=CORPUS[0])
    assert m.delta_bits > 0
    assert 0.0 < m.delta_frac <= 1.0


def test_delta_near_zero_for_unrelated_context():
    """Нерелевантный контекст не должен создавать иллюзию знакомства."""
    s = build_scorer()
    rel = score_chunk(s, CORPUS[1], prior_notes=CORPUS[1])
    unrel = score_chunk(s, CORPUS[1], prior_notes=CORPUS[2])
    assert unrel.delta_frac < 0.15
    assert rel.delta_frac > unrel.delta_frac


def test_token_counts_stable_across_contexts():
    s = build_scorer()
    a = s.score(CHUNK, context="")
    b = s.score(CHUNK, context=NOTES["good"])
    assert a.n_tokens == b.n_tokens


# ==========================================================================
# D. Пригодность экзамена — третья находка стенда
#
# Первый прогон smoke_gpu.py на Qwen3-1.7B провалил проверку C, и причина
# оказалась не в качестве заметки, а в соотношении размеров: база экзамена
# 590.4 бита против стоимости заметки 571.7. Потолок выигрыша +18.7 бита,
# то есть C не могла пройти НИ ПРИ КАКОМ качестве заметки.
#
# Записываем это тестами, чтобы не потерялось: разбор в docs/e1-debug-A-C.md.
# ==========================================================================


def test_low_headroom_makes_positive_gain_impossible():
    """
    Ядро находки H1. Если база экзамена не превышает стоимость заметки,
    выигрыш отрицателен даже при стопроцентном recall. Меряется соотношение
    размеров, а не понимание.
    """
    m = _synthetic(savings=240, note_bits=260, base=240.0)
    assert m.recall == 1.0, "recall должен быть предельным"
    assert m.max_possible_gain_bits < 0
    assert m.task_gain_bits < 0
    assert not chunk_is_measurable(m.L_answers_bits, m.L_note_bits)


def test_first_smoke_run_would_be_rejected_by_the_gate():
    """
    Точные числа первого прогона. Гейт обязан был отбраковать этот чанк
    ДО оценки — тогда провал C не выглядел бы провалом метрики.
    """
    assert not chunk_is_measurable(590.4, 571.7)
    assert exam_headroom(590.4, 571.7) < 1.1


def test_headroom_gate_accepts_adequate_exam():
    """Экзамен втрое дороже заметки — измерять можно."""
    assert chunk_is_measurable(2000.0, 600.0)
    assert exam_headroom(2000.0, 600.0) >= MIN_EXAM_HEADROOM


def test_max_possible_gain_is_the_ceiling():
    """Выигрыш никогда не превышает базу минус цену заметки. Это и есть насыщение."""
    for savings in (0, 50, 120, 200, 240):
        m = _synthetic(savings=savings, note_bits=100, base=240.0)
        assert m.task_gain_bits <= m.max_possible_gain_bits + 1e-9


def test_headroom_is_independent_of_note_quality():
    """
    Гейт обязан зависеть только от размеров, не от того, хороша заметка или
    нет. Иначе он превращается в скрытый отбор по результату.
    """
    good = _synthetic(savings=200, note_bits=100, base=400.0)
    bad = _synthetic(savings=0, note_bits=100, base=400.0)
    assert good.headroom == bad.headroom == 4.0


# ==========================================================================
# E. Разбор ответа модели
#
# Четвёртая находка: парсер экзамена читал строго по строке на JSON. Модель
# выдаёт то JSONL, то тот же JSON с отступами — и во втором случае экзамен
# выходил пустым, чанк отсеивался «за каприз модели». Терялась примерно
# половина чанков, молча. Дефект данных, а не метрики, но на измерения он
# влияет сильнее многих метрических.
# ==========================================================================


def test_exam_parser_reads_jsonl():
    from kstudy.notes import parse_exam

    raw = '{"q": "A?", "a": "one"}\n{"q": "B?", "a": "two"}'
    assert len(parse_exam(raw, 16)) == 2


def test_exam_parser_reads_pretty_printed():
    """Ровно то, на чём терялись чанки."""
    from kstudy.notes import parse_exam

    raw = '{\n  "q": "A?",\n  "a": "one"\n}\n\n{\n  "q": "B?",\n  "a": "two"\n}'
    got = parse_exam(raw, 16)
    assert len(got) == 2, got
    assert got[0].question == "A?"


def test_exam_parser_survives_braces_inside_strings():
    """Вопрос про код со скобками не должен рвать разбор."""
    from kstudy.notes import parse_exam

    raw = '{"q": "What does {} mean?", "a": "An empty initialiser {0}."}'
    got = parse_exam(raw, 16)
    assert len(got) == 1, got
    assert got[0].answer == "An empty initialiser {0}."


def test_exam_parser_respects_limit():
    from kstudy.notes import parse_exam

    raw = "\n".join('{"q": "q%d", "a": "a%d"}' % (i, i) for i in range(30))
    assert len(parse_exam(raw, 16)) == 16


def test_truncated_thinking_yields_nothing():
    """
    Оборванный <think> без закрывающего тега — это не заметка, а размышления
    модели вслух. Пустая строка честнее: вызывающий код пропустит чанк.
    """
    from kstudy.notes import strip_thinking

    assert strip_thinking("<think>I should start by") == ""
    assert strip_thinking("<think>reasoning</think>Note text") == "Note text"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"  ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
