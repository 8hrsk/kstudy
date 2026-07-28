"""
Скореры: превращают пару (контекст, целевой текст) в число бит.

Всё остальное в проекте зависит только от этого интерфейса. Поэтому метрики
можно проверять без GPU (CacheNgramScorer), а считать по-настоящему — на
трансформере (HFScorer).

Соглашение: везде БИТЫ, не наты. Бит — это то, что можно осмысленно сравнивать
с длиной заметки, а вся конструкция MDL про длину сообщения.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

LN2 = math.log(2.0)


@dataclass(frozen=True)
class Score:
    """Стоимость передачи target при данном context."""

    bits: float
    n_tokens: int

    @property
    def bits_per_token(self) -> float:
        return self.bits / self.n_tokens if self.n_tokens else 0.0


class Scorer(ABC):
    """Минимальный интерфейс. Реализаций может быть много."""

    @abstractmethod
    def score(self, target: str, context: str = "") -> Score:
        """Сколько бит нужно, чтобы закодировать target, если известен context."""

    def score_many(
        self, pairs: Sequence[tuple[str, str]], batch_size: int = 8
    ) -> list[Score]:
        """Пакетная версия. Наивная реализация — переопределяйте, если умеете лучше."""
        return [self.score(t, c) for t, c in pairs]


# --------------------------------------------------------------------------
# Рабочий бэкенд: HuggingFace transformers
# --------------------------------------------------------------------------


class HFScorer(Scorer):
    """
    Считает -log2 P(target | context) точно, по токенам target.

    Важная деталь реализации: logits[i] предсказывает токен i+1. Значит для
    токенов target на позициях [s, e) нужны logits на [s-1, e-1). Ошибка на
    единицу здесь даёт правдоподобно выглядящие, но неверные числа — это
    самый частый способ незаметно испортить весь эксперимент.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 8192,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype),
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()

    def _ids(self, text: str, add_special: bool = False) -> list[int]:
        return self.tokenizer(text, add_special_tokens=add_special).input_ids

    @property
    def _bos(self) -> list[int]:
        bos = self.tokenizer.bos_token_id
        return [bos] if bos is not None else []

    def score(self, target: str, context: str = "") -> Score:
        return self.score_many([(target, context)], batch_size=1)[0]

    def score_many(
        self, pairs: Sequence[tuple[str, str]], batch_size: int = 8
    ) -> list[Score]:
        torch = self.torch
        out: list[Score] = []

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            rows = []
            for target, context in batch:
                ctx_ids = self._bos + (self._ids(context) if context else [])
                tgt_ids = self._ids(target)
                if not tgt_ids:
                    rows.append(None)
                    continue
                # Если не влезаем — режем КОНТЕКСТ слева, target трогать нельзя,
                # иначе числа станут несравнимыми между чанками.
                budget = self.max_length - len(tgt_ids)
                if budget < 1:
                    tgt_ids = tgt_ids[: self.max_length - 1]
                    budget = 1
                if len(ctx_ids) > budget:
                    ctx_ids = ctx_ids[-budget:]
                rows.append((ctx_ids, tgt_ids))

            real = [r for r in rows if r is not None]
            if not real:
                out.extend(Score(0.0, 0) for _ in batch)
                continue

            maxlen = max(len(c) + len(t) for c, t in real)
            pad_id = self.tokenizer.pad_token_id or 0

            input_ids, attn, spans = [], [], []
            for r in rows:
                if r is None:
                    continue
                ctx_ids, tgt_ids = r
                seq = ctx_ids + tgt_ids
                pad = maxlen - len(seq)
                # Левый паддинг: так конец последовательности всегда на одном
                # месте, и спаны считать проще.
                input_ids.append([pad_id] * pad + seq)
                attn.append([0] * pad + [1] * len(seq))
                s = pad + len(ctx_ids)
                spans.append((s, s + len(tgt_ids)))

            ids_t = torch.tensor(input_ids, device=self.device)
            attn_t = torch.tensor(attn, device=self.device)

            with torch.no_grad():
                logits = self.model(input_ids=ids_t, attention_mask=attn_t).logits
                logprobs = torch.log_softmax(logits.float(), dim=-1)

            results = []
            for row, (s, e) in enumerate(spans):
                # logits[s-1 : e-1] предсказывают токены [s : e]
                idx = torch.arange(s, e, device=self.device)
                tok_lp = logprobs[row, idx - 1, :].gather(
                    1, ids_t[row, idx].unsqueeze(1)
                ).squeeze(1)
                nats = -tok_lp.sum().item()
                results.append(Score(bits=nats / LN2, n_tokens=int(e - s)))

            it = iter(results)
            out.extend(Score(0.0, 0) if r is None else next(it) for r in rows)

        return out


# --------------------------------------------------------------------------
# Тестовый бэкенд: n-грамма с кэшем, чистый Python
# --------------------------------------------------------------------------


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


class CacheNgramScorer(Scorer):
    """
    Игрушечная LM для проверки логики метрик без GPU. Три компоненты:

      1. n-грамма с backoff  — базовое знание «языка»
      2. кэш по контексту    — слова из контекста дешевеют (Kuhn & De Mori, 1990)
      3. суффиксное совпадение — если последние k токенов уже встречались
         в контексте, предсказываем то, что следовало за ними там

    Третья компонента принципиальна. Без неё дословная копия чанка в роли
    заметки экономит всего ~20 % бит, тогда как настоящий трансформер на копии
    экономит ~95 % — за счёт индукционных голов, которые продолжают ровно то,
    что уже видели. Суффиксное совпадение — это индукционная голова на минималках,
    и без неё стенд не воспроизводит тот режим, ради проверки которого он нужен.

    Ограничение, которое надо знать: у слабой LM стоимость в битах на токен почти
    одинакова для гладкой прозы и для плотного технического текста. У настоящей
    LM она отличается в разы, и именно на этой разнице держится знак MDL-выигрыша.
    Поэтому здесь проверяются ПОРЯДОК величин и знак экономии, а знак строгого
    выигрыша — на реальной модели (см. scripts/smoke_gpu.py).
    """

    def __init__(
        self,
        order: int = 3,
        cache_weight: float = 0.35,
        match_weight: float = 0.9,
        max_match: int = 6,
        vocab_floor: int = 50_000,
    ):
        self.order = order
        self.cache_weight = cache_weight
        self.match_weight = match_weight
        self.max_match = max_match
        self.vocab_floor = vocab_floor
        self.counts: list[dict[tuple[str, ...], dict[str, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order)
        ]
        self.totals: list[dict[tuple[str, ...], int]] = [
            defaultdict(int) for _ in range(order)
        ]
        self.vocab: set[str] = set()

    def fit(self, texts: Iterable[str]) -> "CacheNgramScorer":
        for text in texts:
            toks = simple_tokenize(text)
            self.vocab.update(toks)
            for n in range(self.order):
                for i in range(len(toks)):
                    if i < n:
                        continue
                    hist = tuple(toks[i - n : i])
                    self.counts[n][hist][toks[i]] += 1
                    self.totals[n][hist] += 1
        return self

    @property
    def _vocab_size(self) -> int:
        return max(len(self.vocab), self.vocab_floor)

    def _p_ngram(self, hist: Sequence[str], word: str) -> float:
        """Backoff от самого длинного контекста к униграмме, с аддитивным сглаживанием."""
        for n in range(min(self.order - 1, len(hist)), -1, -1):
            h = tuple(hist[len(hist) - n :]) if n else ()
            total = self.totals[n].get(h, 0)
            if total >= 2:
                c = self.counts[n][h].get(word, 0)
                return (c + 0.1) / (total + 0.1 * self._vocab_size)
        return 1.0 / self._vocab_size

    @staticmethod
    def _build_match_index(
        tokens: Sequence[str], max_match: int
    ) -> dict[tuple[str, ...], list[str]]:
        """ngram -> список токенов, которые следовали за ним в контексте."""
        index: dict[tuple[str, ...], list[str]] = defaultdict(list)
        n = len(tokens)
        for i in range(n):
            for L in range(1, min(max_match, i) + 1):
                index[tuple(tokens[i - L : i])].append(tokens[i])
        return index

    def _p_match(
        self, history: Sequence[str], index: dict[tuple[str, ...], list[str]]
    ) -> tuple[dict[str, float] | None, int]:
        """Самое длинное совпадение суффикса истории с контекстом."""
        for L in range(min(self.max_match, len(history)), 1, -1):
            nxt = index.get(tuple(history[-L:]))
            if nxt:
                total = len(nxt)
                dist: dict[str, float] = defaultdict(float)
                for w in nxt:
                    dist[w] += 1.0 / total
                return dist, L
        return None, 0

    def score(self, target: str, context: str = "") -> Score:
        tgt = simple_tokenize(target)
        if not tgt:
            return Score(0.0, 0)

        ctx = simple_tokenize(context) if context else []
        cache: dict[str, int] = defaultdict(int)
        for w in ctx:
            cache[w] += 1
        cache_total = len(ctx)
        has_ctx = cache_total > 0

        # Индекс совпадений строится по контексту и достраивается по мере
        # чтения target — как растущий KV-кэш.
        seen: list[str] = list(ctx)
        index: dict[tuple[str, ...], list[str]] = self._build_match_index(
            ctx, self.max_match
        )

        bits = 0.0
        history: list[str] = list(ctx)
        for w in tgt:
            p_ng = self._p_ngram(history[-(self.order - 1) :] if self.order > 1 else [], w)

            if has_ctx:
                # Кэш: доля слова среди виденного. Нули допустимы — их
                # закрывает n-граммная компонента в смеси.
                p_cache = cache.get(w, 0) / cache_total if cache_total else 0.0
                p = (1 - self.cache_weight) * p_ng + self.cache_weight * p_cache

                dist, L = self._p_match(history, index)
                if dist is not None:
                    # Доверие растёт с длиной совпадения: 2 токена — так себе,
                    # 6 токенов — почти наверняка дословное продолжение.
                    conf = self.match_weight * (1.0 - 1.0 / (1.0 + L))
                    p = (1 - conf) * p + conf * dist.get(w, 0.0)
            else:
                p = p_ng

            p = min(max(p, 1e-12), 1.0)
            bits += -math.log2(p)

            # Дописываем прочитанное в контекст модели.
            pos = len(seen)
            seen.append(w)
            for L in range(1, min(self.max_match, pos) + 1):
                index[tuple(seen[pos - L : pos])].append(w)
            history.append(w)
            cache[w] += 1
            cache_total += 1
            has_ctx = True

        return Score(bits=bits, n_tokens=len(tgt))
