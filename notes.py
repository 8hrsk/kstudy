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

EXAM_PROMPT = """Read the passage below and write {n} exam questions with \
short answers that test whether someone understood it.

Rules:
- Answers must be PARAPHRASED. Never reuse a phrase from the passage \
verbatim — use different words for the same content. This is critical.
- Each answer: one or two sentences.
- Cover different facts; do not ask the same thing twice.
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
        max_new_tokens: int = 400,
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
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
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
    """Убирает <think>…</think> у reasoning-моделей вроде Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_note(gen: Generator, chunk_text: str, temperature: float = 0.7) -> str:
    return strip_thinking(
        gen.generate(NOTE_PROMPT.format(chunk=chunk_text), temperature)
    )


def make_exam(
    gen: Generator, chunk_text: str, n: int = 4, temperature: float = 0.6
) -> list[QA]:
    raw = strip_thinking(
        gen.generate(EXAM_PROMPT.format(chunk=chunk_text, n=n), temperature)
    )
    out: list[QA] = []
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            if d.get("q") and d.get("a"):
                out.append(QA(str(d["q"]).strip(), str(d["a"]).strip()))
        except json.JSONDecodeError:
            continue
    return out[:n]


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
