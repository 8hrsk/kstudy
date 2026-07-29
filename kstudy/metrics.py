"""
Метрики Э1. Три числа, всё остальное — производные.

  L_base   = -log2 P(C)              насколько чанк удивителен сам по себе
  L_info   = -log2 P(C | заметки)    насколько помогает уже накопленное
  MDL-выигрыш = L(C) - L(N) - L(C|N) окупается ли заметка N

Про MDL-выигрыш отдельно, потому что это главная идея.

Хочется штрафовать заметку за длину, иначе оптимальная «заметка» — дословная
копия чанка: она обнуляет L(C|N) и выигрывает всё. Но брать штраф вида
λ·len(N) в символах — произвол: непонятно, откуда взять λ, и метрика перестаёт
быть сравнимой между корпусами.

Правильный штраф уже есть в теории. Полная длина сообщения при передаче чанка
через заметку = (закодировать заметку) + (закодировать чанк, зная заметку).
Заметка платит за себя своей же длиной кода:

    выигрыш = L(C) - [ L(N) + L(C|N) ]

Никакого свободного параметра. И копия наказывается автоматически: копия стоит
ровно столько же, сколько сам чанк, поэтому L(N) ≈ L(C) и выигрыш ≈ 0.

λ оставлен только как ручка для абляции. λ = 1 — честный MDL, и это дефолт.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Sequence

from .scoring import Score, Scorer

Triage = Literal["known", "zpd", "too_far"]


# --------------------------------------------------------------------------
# Триаж чанка: что читать дальше
# --------------------------------------------------------------------------


@dataclass
class ChunkMetrics:
    chunk_id: str
    n_tokens: int
    L_base_bits: float
    L_info_bits: float

    @property
    def L_base_per_token(self) -> float:
        return self.L_base_bits / self.n_tokens if self.n_tokens else 0.0

    @property
    def L_info_per_token(self) -> float:
        return self.L_info_bits / self.n_tokens if self.n_tokens else 0.0

    @property
    def delta_bits(self) -> float:
        """Сколько бит экономит уже накопленное знание. Всегда >= 0 у разумной модели."""
        return self.L_base_bits - self.L_info_bits

    @property
    def delta_per_token(self) -> float:
        return self.delta_bits / self.n_tokens if self.n_tokens else 0.0

    @property
    def delta_frac(self) -> float:
        """Доля стоимости, снятая контекстом. Безразмерно, сравнимо между чанками."""
        return self.delta_bits / self.L_base_bits if self.L_base_bits > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            L_base_per_token=self.L_base_per_token,
            delta_bits=self.delta_bits,
            delta_per_token=self.delta_per_token,
            delta_frac=self.delta_frac,
        )
        return d


@dataclass
class TriageThresholds:
    """
    Пороги калибруются по квантилям корпуса (см. calibrate_thresholds), потому
    что абсолютные биты на токен зависят от модели и предметной области.
    """

    known_below: float = 2.0       # L_info/tok ниже — материал уже усвоен
    delta_frac_min: float = 0.10   # доля экономии выше — материал «примыкает»


def triage(m: ChunkMetrics, th: TriageThresholds) -> Triage:
    """
    known   — уже усвоено, генерацию не тратить
    zpd     — зона ближайшего развития: ново, но цепляется за известное
    too_far — ново и не за что зацепиться, сначала предпосылки

    Порог «известного» берётся по L_info, а НЕ по L_base. Это не косметика:
    L_base не знает, что модель уже прочитала, поэтому только что изученный
    материал по ней выглядит таким же новым, как непрочитанный. Вопрос
    «знаю ли я это» — это «предсказуемо ли это ПРИ МОИХ ЗАМЕТКАХ», то есть
    именно L_info.
    """
    if m.L_info_per_token < th.known_below:
        return "known"
    return "zpd" if m.delta_frac >= th.delta_frac_min else "too_far"


def calibrate_thresholds(
    metrics: Sequence[ChunkMetrics],
    known_quantile: float = 0.30,
    delta_quantile: float = 0.50,
) -> TriageThresholds:
    """Пороги по квантилям корпуса — чтобы не подгонять руками под каждый домен."""
    if not metrics:
        return TriageThresholds()
    info = sorted(m.L_info_per_token for m in metrics)
    dfrac = sorted(m.delta_frac for m in metrics)

    def q(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        i = min(int(round(p * (len(xs) - 1))), len(xs) - 1)
        return xs[i]

    # Строго «ниже порога», поэтому берём чуть выше квантиля, иначе сам
    # элемент-квантиль в класс не попадёт.
    return TriageThresholds(
        known_below=q(info, known_quantile) * 1.0001,
        delta_frac_min=q(dfrac, delta_quantile),
    )


def score_chunk(
    scorer: Scorer, chunk_text: str, prior_notes: str = ""
) -> ChunkMetrics:
    base = scorer.score(chunk_text, context="")
    info = scorer.score(chunk_text, context=prior_notes) if prior_notes else base
    return ChunkMetrics(
        chunk_id="",
        n_tokens=base.n_tokens,
        L_base_bits=base.bits,
        L_info_bits=info.bits,
    )


# --------------------------------------------------------------------------
# MDL-выигрыш заметки: окупается ли она
# --------------------------------------------------------------------------


@dataclass
class NoteMetrics:
    note_id: str
    chunk_id: str
    kind: str                    # откуда заметка: model / copy / water / unrelated
    n_tokens_chunk: int
    n_tokens_note: int
    L_chunk_bits: float          # L(C)     — чанк сам по себе
    L_note_bits: float           # L(N)     — стоимость передать заметку
    L_chunk_given_note_bits: float  # L(C|N) — чанк, когда заметка известна
    lam: float = 1.0

    @property
    def savings_bits(self) -> float:
        """Сырая экономия, без платы за заметку. Копия выигрывает по этой метрике —
        поэтому одной её недостаточно, но смотреть на неё полезно."""
        return self.L_chunk_bits - self.L_chunk_given_note_bits

    @property
    def mdl_gain_bits(self) -> float:
        """Главное число Э1."""
        return self.savings_bits - self.lam * self.L_note_bits

    @property
    def mdl_gain_per_chunk_token(self) -> float:
        return (
            self.mdl_gain_bits / self.n_tokens_chunk if self.n_tokens_chunk else 0.0
        )

    @property
    def savings_frac(self) -> float:
        return (
            self.savings_bits / self.L_chunk_bits if self.L_chunk_bits > 0 else 0.0
        )

    @property
    def leverage(self) -> float:
        """
        Во сколько раз заметка экономит больше, чем стоит сама.
        > 1 — окупается. Безразмерно, поэтому сравнимо между чанками разной длины,
        и на практике ранжирует лучше, чем сырые биты.
        """
        return (
            self.savings_bits / self.L_note_bits if self.L_note_bits > 0 else 0.0
        )

    @property
    def verdict(self) -> str:
        if self.mdl_gain_bits <= 0:
            return "reject"
        return "keep" if self.leverage >= 1.5 else "weak"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            savings_bits=self.savings_bits,
            mdl_gain_bits=self.mdl_gain_bits,
            mdl_gain_per_chunk_token=self.mdl_gain_per_chunk_token,
            savings_frac=self.savings_frac,
            leverage=self.leverage,
            verdict=self.verdict,
        )
        return d


def score_note(
    scorer: Scorer,
    chunk_text: str,
    note_text: str,
    *,
    note_id: str = "",
    chunk_id: str = "",
    kind: str = "model",
    lam: float = 1.0,
    note_prefix: str = "Notes:\n",
    chunk_prefix: str = "\n\nSource:\n",
    cached_chunk_score: Score | None = None,
) -> NoteMetrics:
    """
    Считает три величины и складывает их в NoteMetrics.

    cached_chunk_score: L(C) не зависит от заметки, поэтому при сравнении
    нескольких заметок к одному чанку его считают один раз.
    """
    base = cached_chunk_score or scorer.score(chunk_text, context="")
    note = scorer.score(note_text, context="")
    context = f"{note_prefix}{note_text}{chunk_prefix}"
    cond = scorer.score(chunk_text, context=context)

    return NoteMetrics(
        note_id=note_id,
        chunk_id=chunk_id,
        kind=kind,
        n_tokens_chunk=base.n_tokens,
        n_tokens_note=note.n_tokens,
        L_chunk_bits=base.bits,
        L_note_bits=note.bits,
        L_chunk_given_note_bits=cond.bits,
        lam=lam,
    )


# --------------------------------------------------------------------------
# Закрытый экзамен: MDL на ответах, а не на поверхности текста
# --------------------------------------------------------------------------
#
# Почему поверхностной метрики выше НЕДОСТАТОЧНО — это выяснилось на стенде,
# и это не артефакт игрушечной модели.
#
# Строгий MDL спрашивает: дешевле ли передать (заметка + чанк при заметке),
# чем чанк напрямую? Для этой задачи дословная копия — теоретически почти
# оптимальный код: платим L(C) за копию и ~0 за чанк, итого выигрыш ≈ 0,
# что лучше любого сжатия с потерями. Замер на стенде: копия даёт 95.9 %
# экономии и выигрыш -26 бит, тогда как осмысленная сжатая заметка — 39 %
# и -227 бит. То есть поверхностный MDL СТРУКТУРНО поощряет копирование,
# и плата за длину заметки этого не чинит.
#
# Причина в том, что L(C) берёт плату за точную словесную форму, а понимание
# к словесной форме не сводится. Значит метрика меряет не то.
#
# Правильный вопрос — не «восстанови текст», а «ответь на вопросы по нему
# с закрытой книгой». Именно так человек проверяет себя без учителя.
#
#     выигрыш = [ L(A|Q) - L(A|Q,N) ] - λ·L(N)
#
# Здесь копия наказывается сама собой: польза от неё на коротких ответах
# ограничена сверху (больше, чем L(A|Q), не сэкономишь), а платит она полную
# L(C) > L(N). Заметка выигрывает именно тем, что дешевле при той же пользе.


@dataclass
class QA:
    question: str
    answer: str


@dataclass
class TaskNoteMetrics:
    """MDL-выигрыш заметки, измеренный на закрытом экзамене."""

    note_id: str
    chunk_id: str
    kind: str
    n_questions: int
    n_tokens_note: int
    n_tokens_answers: int
    L_note_bits: float
    L_answers_bits: float             # Σ L(A|Q)      — без заметки
    L_answers_given_note_bits: float  # Σ L(A|Q,N)    — с заметкой
    lam: float = 1.0

    @property
    def savings_bits(self) -> float:
        return self.L_answers_bits - self.L_answers_given_note_bits

    @property
    def task_gain_bits(self) -> float:
        """Главное число Э1."""
        return self.savings_bits - self.lam * self.L_note_bits

    @property
    def recall(self) -> float:
        """Доля экзамена, закрытая заметкой. 1.0 — отвечает на всё."""
        return (
            self.savings_bits / self.L_answers_bits
            if self.L_answers_bits > 0
            else 0.0
        )

    @property
    def leverage(self) -> float:
        """Польза на бит заметки. Безразмерно, сравнимо между чанками."""
        return (
            self.savings_bits / self.L_note_bits if self.L_note_bits > 0 else 0.0
        )

    @property
    def headroom(self) -> float:
        """
        Во сколько раз потолок экономии превышает цену заметки. См.
        MIN_EXAM_HEADROOM: ниже 3 отрицательный выигрыш не означает ничего,
        кроме короткого экзамена.
        """
        return exam_headroom(self.L_answers_bits, self.L_note_bits)

    @property
    def max_possible_gain_bits(self) -> float:
        """
        Выигрыш при стопроцентном recall — недостижимый потолок. Полезен
        именно как диагностика: если он отрицателен, заметку можно не считать.
        """
        return self.L_answers_bits - self.lam * self.L_note_bits

    @property
    def verdict(self) -> str:
        if self.savings_bits <= 0:
            return "reject"
        if self.task_gain_bits <= 0:
            return "weak"
        return "keep"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            savings_bits=self.savings_bits,
            task_gain_bits=self.task_gain_bits,
            recall=self.recall,
            leverage=self.leverage,
            headroom=self.headroom,
            max_possible_gain_bits=self.max_possible_gain_bits,
            verdict=self.verdict,
        )
        return d


def score_note_on_exam(
    scorer: Scorer,
    note_text: str,
    exam: Sequence[QA],
    *,
    note_id: str = "",
    chunk_id: str = "",
    kind: str = "model",
    lam: float = 1.0,
    cached_baseline: tuple[float, int] | None = None,
) -> TaskNoteMetrics:
    """
    Вопросы должны быть составлены ПО ЧАНКУ (с чанком в контексте — это
    извлечение, безопасная операция), а отвечать модель обязана без чанка.
    Иначе экзамен проверяет не заметку, а способность читать.

    cached_baseline: (Σ L(A|Q), Σ токенов ответов) — не зависит от заметки,
    поэтому при сравнении нескольких заметок считается один раз.
    """
    note = scorer.score(note_text, context="")

    if cached_baseline is None:
        base_bits, n_ans_tokens = 0.0, 0
        for qa in exam:
            s = scorer.score(qa.answer, context=f"Question: {qa.question}\nAnswer:")
            base_bits += s.bits
            n_ans_tokens += s.n_tokens
    else:
        base_bits, n_ans_tokens = cached_baseline

    cond_bits = 0.0
    for qa in exam:
        ctx = f"Notes:\n{note_text}\n\nQuestion: {qa.question}\nAnswer:"
        cond_bits += scorer.score(qa.answer, context=ctx).bits

    return TaskNoteMetrics(
        note_id=note_id,
        chunk_id=chunk_id,
        kind=kind,
        n_questions=len(exam),
        n_tokens_note=note.n_tokens,
        n_tokens_answers=n_ans_tokens,
        L_note_bits=note.bits,
        L_answers_bits=base_bits,
        L_answers_given_note_bits=cond_bits,
        lam=lam,
    )


def exam_baseline(
    scorer: Scorer, exam: Sequence[QA]
) -> tuple[float, int]:
    """Считает Σ L(A|Q) один раз, чтобы переиспользовать между заметками."""
    bits, toks = 0.0, 0
    for qa in exam:
        s = scorer.score(qa.answer, context=f"Question: {qa.question}\nAnswer:")
        bits += s.bits
        toks += s.n_tokens
    return bits, toks


# --------------------------------------------------------------------------
# Пригодность чанка к измерению (находка H1, docs/e1-debug-A-C.md)
# --------------------------------------------------------------------------

MIN_EXAM_HEADROOM = 3.0
"""
Во сколько раз база экзамена должна превышать стоимость заметки.

Откуда взялось. Экономия ограничена сверху базой: больше, чем Σ L(A|Q), на
ответах не сэкономишь. Значит

    выигрыш = экономия - λ·L(N)  ≤  база - λ·L(N)

и при базе ≈ L(N) даже стопроцентный recall даёт выигрыш около нуля. Первый
прогон смоука поймал ровно это: база 590.4 бит против L(N) 571.7, потолок
+18.7 бита. Проверка C не могла пройти ни при каком качестве заметки — мерялся
не смысл, а соотношение размеров.

Тройка — не подобранный порог, а требование, чтобы потолок был заметно больше
измеряемого эффекта. При headroom = 3 заметка, закрывшая половину экзамена,
даёт выигрыш +0.5·база, то есть сигнал вдвое выше цены заметки.

Это ФИЛЬТР ПРИГОДНОСТИ, а не вердикт: чанк с низким headroom не оценивается
вовсе. Смешивать его с verdict нельзя — verdict отвечает на вопрос «хороша ли
заметка», а здесь мы говорим «этим экзаменом вообще ничего измерить нельзя».
"""


def exam_headroom(base_bits: float, note_bits: float) -> float:
    """Во сколько раз потолок экономии превышает цену заметки."""
    return base_bits / note_bits if note_bits > 0 else float("inf")


def chunk_is_measurable(
    base_bits: float, note_bits: float, min_headroom: float = MIN_EXAM_HEADROOM
) -> bool:
    """
    Можно ли на этом экзамене вообще что-то измерить.

    Проверять ДО оценки заметки и выбрасывать чанк при False — иначе в выборку
    попадают чанки, где отрицательный выигрыш означает лишь короткий экзамен,
    а не плохую заметку, и корреляция с человеческой оценкой размывается шумом.
    """
    return exam_headroom(base_bits, note_bits) >= min_headroom
