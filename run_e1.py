#!/usr/bin/env python3
"""
Э1 целиком: чанки -> триаж -> заметки и экзамены -> метрики.

    python run_e1.py --kernel ~/linux --subsystem rcu --n-chunks 40

На RTX 3060 с Qwen3-1.7B это порядка часа на 40 чанков (4 заметки на чанк:
модельная плюс три контроля). Результат — data/metrics.jsonl, дальше руками
размечается сотня примеров и считается корреляция.

Перед первым запуском обязательно: python scripts/smoke_gpu.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from kstudy._compat import enable_utf8_console
from kstudy.corpus import load_subsystem, save_chunks
from kstudy.metrics import (
    MIN_EXAM_HEADROOM,
    calibrate_thresholds,
    chunk_is_measurable,
    exam_baseline,
    score_chunk,
    score_note,
    score_note_on_exam,
    triage,
)
from kstudy.notes import (
    Generator,
    NoteRecord,
    build_controls,
    exam_leakage,
    make_exam,
    make_note,
    save_jsonl,
    verbatim_overlap,
)
from kstudy.scoring import HFScorer


def main() -> int:
    # Строго до ArgumentParser: argparse печатает справку и свои ошибки прямо
    # из parse_args(), а в них есть кириллица. На неперенастроенном stdout
    # (cp1251/cp1252, и всегда при перенаправлении вывода) это падает с
    # UnicodeEncodeError ещё до первой строки полезной работы.
    enable_utf8_console()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="путь к дереву ядра")
    ap.add_argument("--subsystem", default="rcu")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-chunks", type=int, default=40)
    # Шестнадцать, а не четыре. При четырёх база экзамена оказывается сравнима
    # со стоимостью заметки (замер: 590.4 против 571.7 бита), потолок выигрыша
    # +18.7 бита, и метрика меряет соотношение размеров вместо понимания.
    # См. MIN_EXAM_HEADROOM и docs/e1-debug-A-C.md.
    ap.add_argument("--n-questions", type=int, default=16)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    # ---- 1. корпус -------------------------------------------------------
    print(f"[1/5] Нарезаю {a.subsystem} из {a.kernel}")
    chunks = [
        c for c in load_subsystem(a.kernel, a.subsystem)
        if a.min_chars <= len(c.text) <= a.max_chars
    ]
    save_chunks(chunks, out / "chunks_all.jsonl")
    print(f"      {len(chunks)} чанков подходящего размера")

    sample = rng.sample(chunks, min(a.n_chunks, len(chunks)))
    save_chunks(sample, out / "chunks.jsonl")
    print(f"      взято в работу: {len(sample)}")

    # ---- 2. модель -------------------------------------------------------
    print(f"[2/5] Загружаю {a.model}")
    scorer = HFScorer(a.model, device=a.device, dtype=a.dtype)
    gen = Generator(a.model, device=a.device, dtype=a.dtype)

    # ---- 3. триаж --------------------------------------------------------
    # «Уже прочитанным» считаем документацию — она задаёт контекст, в котором
    # оценивается новизна кода. Для полноценного цикла сюда пойдут накопленные
    # заметки, но в Э1 обучения ещё нет.
    print("[3/5] Триаж")
    prior = "\n\n".join(c.text for c in sample if c.kind == "doc")[:6000]
    cms = []
    for c in sample:
        m = score_chunk(scorer, c.text, prior_notes=prior)
        m.chunk_id = c.chunk_id
        cms.append(m)
    th = calibrate_thresholds(cms)
    labels = {m.chunk_id: triage(m, th) for m in cms}
    dist: dict[str, int] = {}
    for v in labels.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"      пороги: known_below={th.known_below:.2f} б/т  "
          f"delta_frac_min={th.delta_frac_min:.3f}")
    print(f"      распределение: {dist}")
    with (out / "triage.jsonl").open("w", encoding="utf-8") as fh:
        for m in cms:
            d = m.to_dict()
            d["triage"] = labels[m.chunk_id]
            fh.write(json.dumps(d) + "\n")

    # ---- 4. заметки и экзамены -------------------------------------------
    print(f"[4/5] Генерирую заметки и экзамены ({len(sample)} чанков)")
    all_notes: list[NoteRecord] = []
    exams: dict[str, list] = {}
    texts = [c.text for c in sample]
    t0 = time.time()

    for i, c in enumerate(sample, 1):
        note = make_note(gen, c.text)
        if len(note) < 40:
            print(f"      [{i}] пустая заметка, пропуск")
            continue
        exam = make_exam(gen, c.text, n=a.n_questions)
        if len(exam) < 2:
            print(f"      [{i}] экзамен не распарсился ({len(exam)} вопросов), пропуск")
            continue

        leak = exam_leakage(exam, c.text)
        if leak > 0.25:
            print(f"      [{i}] ⚠ утечка источника в ответы {leak:.0%} — "
                  f"экзамен вознаграждает запоминание, чиню промпт? пропуск")
            continue

        exams[c.chunk_id] = exam
        all_notes.append(NoteRecord(f"{c.chunk_id}::model", c.chunk_id, "model", note))
        others = [t for t in texts if t is not c.text]
        all_notes.extend(build_controls(c.chunk_id, c.text, others, rng))

        if i % 5 == 0:
            el = time.time() - t0
            print(f"      {i}/{len(sample)}  {el:.0f}с  "
                  f"(~{el / i * (len(sample) - i) / 60:.0f} мин осталось)")

    save_jsonl(all_notes, out / "notes.jsonl")
    with (out / "exams.jsonl").open("w", encoding="utf-8") as fh:
        for cid, exam in exams.items():
            fh.write(json.dumps({
                "chunk_id": cid,
                "exam": [{"q": q.question, "a": q.answer} for q in exam],
            }, ensure_ascii=False) + "\n")
    print(f"      {len(all_notes)} заметок по {len(exams)} чанкам")

    # ---- 5. метрики ------------------------------------------------------
    print("[5/5] Считаю метрики")
    by_id = {c.chunk_id: c for c in sample}
    rows = []
    baselines: dict[str, tuple[float, int]] = {}
    chunk_scores: dict[str, object] = {}

    # Гейт пригодности. Считается ОДИН РАЗ на чанк и по МОДЕЛЬНОЙ заметке:
    # контроли (копия) заведомо дороже, и мерить по ним значило бы выбрасывать
    # чанк за то, что копия длинная. Если потолок сравним с ценой заметки,
    # выбрасывается весь чанк со всеми контролями — иначе в выборку попадают
    # строки, где отрицательный выигрыш означает лишь короткий экзамен.
    model_note = {n.chunk_id: n.text for n in all_notes if n.kind == "model"}
    unmeasurable: set[str] = set()
    for cid, exam in exams.items():
        if cid not in baselines:
            baselines[cid] = exam_baseline(scorer, exam)
        note_text = model_note.get(cid)
        if note_text is None:
            continue
        l_note = scorer.score(note_text).bits
        if not chunk_is_measurable(baselines[cid][0], l_note):
            unmeasurable.add(cid)

    if unmeasurable:
        print(f"      ⚠ {len(unmeasurable)} чанков непригодны: headroom < "
              f"{MIN_EXAM_HEADROOM} (экзамен слишком дёшев относительно заметки)")

    for j, nr in enumerate(all_notes, 1):
        exam = exams.get(nr.chunk_id)
        chunk = by_id.get(nr.chunk_id)
        if not exam or not chunk:
            continue
        if nr.chunk_id in unmeasurable:
            continue
        if nr.chunk_id not in baselines:
            baselines[nr.chunk_id] = exam_baseline(scorer, exam)
        if nr.chunk_id not in chunk_scores:
            chunk_scores[nr.chunk_id] = scorer.score(chunk.text)

        tm = score_note_on_exam(
            scorer, nr.text, exam, note_id=nr.note_id, chunk_id=nr.chunk_id,
            kind=nr.kind, lam=a.lam, cached_baseline=baselines[nr.chunk_id],
        )
        sm = score_note(
            scorer, chunk.text, nr.text, note_id=nr.note_id, chunk_id=nr.chunk_id,
            kind=nr.kind, lam=a.lam, cached_chunk_score=chunk_scores[nr.chunk_id],
        )
        row = tm.to_dict()
        row.update({
            "mdl_gain_bits": sm.mdl_gain_bits,
            "savings_bits_surface": sm.savings_bits,
            "surface_savings_frac": sm.savings_frac,
            "verbatim_overlap": verbatim_overlap(nr.text, chunk.text),
            "triage": labels.get(nr.chunk_id, "?"),
        })
        rows.append(row)
        if j % 20 == 0:
            print(f"      {j}/{len(all_notes)}")

    with (out / "metrics.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nГотово: {out/'metrics.jsonl'} ({len(rows)} строк)")
    print("\nДальше:")
    print(f"  python -m kstudy.analyze {out}/metrics.jsonl          # контроли, без разметки")
    print(f"  python -m kstudy.rate {out}/notes.jsonl {out}/chunks.jsonl")
    print(f"  python -m kstudy.analyze {out}/metrics.jsonl --ratings {out}/ratings.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
