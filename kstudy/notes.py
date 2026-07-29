"""
Генерация заметок и экзаменов.

Два правила, оба выведены из провалов стенда (tests/test_metrics.py) —
нарушите любое, и метрика начнёт мерить не то:

1. Экзамен составляется ПО чанку (чанк в контексте — это извлечение,
   безопасная операция), а отвечает модель БЕЗ чанка. Иначе экзамен проверяет
   умение читать, а не заметку.

2. Ответы экзамена обязаны быть ПЕРЕФРАЗИРОВАНЫ прочь от формулировок
   источника. Если ответ — дословный кусок текста, побеждает тот «конспект»,
   который просто скопировал источник, и метрика меряет запоминание вместо
   понимания. Стенд ловит это тестом test_verbatim_exam_answers_reward_memorisation.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

from .metrics import QA

# --------------------------------------------------------------------------
# Промпты
# --------------------------------------------------------------------------

NOTE_PROMPT = """You are studying kernel source material. Write compact study \
notes for the passage below.

Rules:
- Capture the mechanism and the reasons, not the wording.
- Do NOT copy phrases from the passage; restate everything in your own words.
- No preamble, no "here are the notes". Just the notes.
- Aim for roughly a third of the length of the passage.

Passage:
{chunk}

Notes:"""

# Вопросы НА ВЫВОД, а не на извлечение. Прогон на 33 чанках показал, почему
# это принципиально: дословная копия источника закрывала экзамен на 47.8 %
# против 26.5 % у осмысленного конспекта, то есть копия была вдвое полезнее и
# проигрывала только штрафом за длину (попарно конспект побеждал в 55 %
# случаев — чуть лучше монетки).
#
# Причина структурная: если ответ дословно лежит в тексте, переписанный
# источник — оптимальная шпаргалка ПО ПОСТРОЕНИЮ. Соревнование идёт в поиске,
# а не в понимании. Вопрос на вывод отнимает у копии это преимущество:
# готового ответа в источнике нет, а конспект с выделенными связями помогает.
# Критерий проверки — docs/e1-exam-v2-criteria.md.
EXAM_PROMPT = """Read the passage below and write {n} exam questions with \
short answers that test whether someone understood it, not whether they can \
locate a phrase in it.

Critical rule — no extraction questions: for each question you draft, check \
whether it could be answered by finding one sentence in the passage and \
copying it. If so, discard the question and write a different one. A \
well-formed question requires reasoning that goes beyond anything the \
passage states directly: the reader must work out a consequence, a cause, a \
comparison, or an application that is never spelled out in those words.

Draw the {n} questions from a mix of these angles:
- Consequence: what would break, or what hazard would appear, if a described \
step were skipped, delayed, done out of order, or done differently.
- Necessity: why some condition, check, or component is required — what \
goes wrong if it were relaxed, loosened, or removed.
- Counterfactual variant: given a hypothetical change to the mechanism, or a \
made-up alternative design, what problem would that change introduce.
- Comparison / transfer: contrast two mechanisms, functions, or code paths \
from the passage along a specific dimension (cost, blocking behavior, which \
one a developer should pick and when), or apply the passage's logic to a \
new situation it never mentions.

This must work whether the passage is C source with comments or prose \
documentation (.rst). For code, ask about why it is structured that way, \
what invariant a check or ordering enforces, or what would break if a line \
were removed or reordered — never what a comment literally says.

Rules on wording — read carefully, this has two separate parts:
- Paraphrase the SENTENCE STRUCTURE: the order of ideas, the sentence \
shape, the connecting logic ("because", "unlike", "only if"). Never reuse a \
run of 6 or more consecutive words from the passage for how an explanation \
is built. This is critical and is checked automatically.
- Do NOT paraphrase domain terminology: technical identifiers, function or \
API names, and established domain terms MUST be reused as-is, not \
described around. A term is not plagiarism — inventing a vaguer stand-in \
for it (e.g. writing "the deferred style" instead of "call_rcu", or "the \
recognized signal" instead of "quiescent state") makes the answer unusable: \
it can no longer be predicted from a note that correctly uses the real \
vocabulary of the passage, which silently destroys the entire measurement. \
If the passage names a function, variable, flag, or concept, your answer \
should name it too, every time it is the thing being discussed.

  Example — passage mentions a function called flush_cache() and explains \
it discards dirty entries before a context switch.
  BAD (over-paraphrased, avoid): "The synchronization routine clears out \
pending entries before the handoff to different work happens."
  GOOD (structure paraphrased, terms kept): "flush_cache() has to run before \
the context switch, because any dirty entries left behind would be invisible \
to the next task using the CPU. Skipping the call would let stale state leak \
across the switch instead of being discarded first."

- Each answer: two or three full sentences. Short answers are not enough: \
the exam baseline is the ceiling on how much a note can ever save, and a \
thin exam makes the whole measurement meaningless.
- Cover different facts and different angles; do not ask the same thing twice.
- Output strictly one JSON object per line: {{"q": "...", "a": "..."}}
- No other text.

Passage:
{chunk}

Questions:"""

# Контроли для калибровки: с ними видно, что метрика вообще различает.
WATER_TEMPLATES = [
    "This passage covers an important concept in the kernel. Understanding it "
    "properly is valuable, since these mechanisms matter a great deal for both "
    "correctness and performance, and they come up frequently in practice.",
    "The material here describes a core piece of the subsystem. It is worth "
    "studying carefully, because the details have real consequences and the "
    "topic recurs throughout the codebase in various forms.",
]


@dataclass
class NoteRecord:
    note_id: str
    chunk_id: str
    kind: str          # model | copy | water | unrelated
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Генератор
# --------------------------------------------------------------------------


class Generator:
    """Тонкая обёртка над transformers для чат-генерации."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        # Экзамен на 16 вопросов с ответами по 2-3 предложения — это ~1100
        # токенов. При прежних 400 генерация обрывалась, не дойдя до JSON, и
        # ВСЕ чанки отсеивались с «экзамен не распарсился». Запас нужен.
        max_new_tokens: int = 2000,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).to(device)
        self.model.eval()

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        msgs = [{"role": "user", "content": prompt}]
        # Qwen3 по умолчанию рассуждает вслух в <think>…</think>. Нам нужен
        # только JSON экзамена, а рассуждение съедает бюджет генерации: замер
        # на 16 вопросах — 400 токенов уходили в think целиком, до ответа
        # дело не доходило. Просим шаблон выключить режим; у моделей без
        # такого параметра аргумент просто не поддерживается.
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt
        except Exception:
            text = prompt
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature or None,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def strip_thinking(text: str) -> str:
    """
    Убирает <think>…</think> у reasoning-моделей вроде Qwen3.

    Отдельно обрабатывается ОБРЫВ: если открывающий тег есть, а закрывающего
    нет, генерация кончилась прямо посреди рассуждения. Раньше такой текст
    возвращался как есть и мог уехать в метрики в роли заметки — то есть мы
    померили бы стоимость размышлений модели вслух. Возвращаем пустую строку:
    вызывающий код уже умеет отбрасывать пустые заметки и нераспарсенные
    экзамены, и пропуск чанка честнее подсунутого мусора.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        return ""
    return text


def make_note(gen: Generator, chunk_text: str, temperature: float = 0.7) -> str:
    return strip_thinking(
        gen.generate(NOTE_PROMPT.format(chunk=chunk_text), temperature)
    )


def iter_json_objects(raw: str):
    """
    Выдаёт сбалансированные {…} блоки из произвольного текста.

    Разбор ПО СТРОКАМ, который был здесь раньше, требовал ровно один JSON на
    строку. Модель это соблюдает не всегда: на одном чанке выдаёт JSONL, на
    следующем — тот же JSON с отступами, и тогда строка «{» не парсится, а
    экзамен выходит пустым. Отсеивалась примерно половина чанков, причём
    молча — по причине «экзамен не распарсился», которая выглядит как каприз
    модели, а не как дефект парсера.

    Скобки внутри строковых литералов и экранирование учитываются, иначе
    вопрос со знаком «{» в тексте рвёт разбор.
    """
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield raw[start : i + 1]
                    start = None


def parse_exam(raw: str, n: int) -> list[QA]:
    """Достаёт вопросы из ответа модели. Чистая функция — тестируется без GPU."""
    out: list[QA] = []
    for blob in iter_json_objects(raw):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("q") and d.get("a"):
            out.append(QA(str(d["q"]).strip(), str(d["a"]).strip()))
    return out[:n]


def make_exam(
    gen: Generator, chunk_text: str, n: int = 4, temperature: float = 0.6
) -> list[QA]:
    raw = strip_thinking(
        gen.generate(EXAM_PROMPT.format(chunk=chunk_text, n=n), temperature)
    )
    return parse_exam(raw, n)


# --------------------------------------------------------------------------
# Контроли
# --------------------------------------------------------------------------


def verbatim_overlap(note: str, chunk: str, n: int = 6) -> float:
    """
    Доля n-грамм заметки, дословно встречающихся в источнике.

    Служит двум целям: ловит заметки-копии до всякого счёта бит, и проверяет,
    что генератор экзамена действительно перефразирует (правило 2 выше).
    Порог для тревоги — примерно 0.25.
    """
    def grams(s: str) -> set[tuple[str, ...]]:
        w = re.findall(r"\w+", s.lower())
        return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}

    a, b = grams(note), grams(chunk)
    return len(a & b) / len(a) if a else 0.0


def exam_leakage(exam: Sequence[QA], chunk: str, n: int = 6) -> float:
    """Средняя дословная утечка источника в ответы. Должна быть низкой."""
    if not exam:
        return 0.0
    return sum(verbatim_overlap(qa.answer, chunk, n) for qa in exam) / len(exam)


def build_controls(
    chunk_id: str,
    chunk_text: str,
    other_chunks: Sequence[str],
    rng: random.Random | None = None,
) -> list[NoteRecord]:
    """Копия, вода и чужая заметка — три контроля к каждому чанку."""
    r = rng or random.Random(0)
    out = [
        NoteRecord(f"{chunk_id}::copy", chunk_id, "copy", chunk_text),
        NoteRecord(f"{chunk_id}::water", chunk_id, "water", r.choice(WATER_TEMPLATES)),
    ]
    if other_chunks:
        out.append(
            NoteRecord(
                f"{chunk_id}::unrelated",
                chunk_id,
                "unrelated",
                r.choice(list(other_chunks))[:900],
            )
        )
    return out


def save_jsonl(records: Sequence, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            d = rec.to_dict() if hasattr(rec, "to_dict") else asdict(rec)
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
