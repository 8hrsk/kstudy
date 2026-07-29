#!/usr/bin/env python3
"""
Досчитывает шаг 5 run_e1.py по уже сохранённым артефактам.

    python scripts/finish_metrics.py --outdir data

Зачем отдельно. Генерация заметок и экзаменов занимает ~55 минут на 40 чанках,
а расчёт метрик — минуты. Если прогон оборвался на шаге 5 (у нас процесс убило
сигналом на 100/156, без traceback), переделывать генерацию незачем: chunks,
notes и exams уже на диске.

Пишет metrics.jsonl построчно и умеет возобновляться: при повторном запуске
пропускает note_id, которые уже посчитаны. Обрыв стоит только незаписанного
хвоста, а не всего прогона.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kstudy._compat import enable_utf8_console  # noqa: E402
from kstudy.metrics import (  # noqa: E402
    MIN_EXAM_HEADROOM,
    QA,
    chunk_is_measurable,
    exam_baseline,
    score_note,
    score_note_on_exam,
)
from kstudy.notes import verbatim_overlap  # noqa: E402
from kstudy.scoring import HFScorer  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main() -> int:
    enable_utf8_console()

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()

    out = Path(a.outdir)
    chunks = {c["chunk_id"]: c["text"] for c in load_jsonl(out / "chunks.jsonl")}
    notes = load_jsonl(out / "notes.jsonl")
    exams = {
        e["chunk_id"]: [QA(q["q"], q["a"]) for q in e["exam"]]
        for e in load_jsonl(out / "exams.jsonl")
    }
    labels = {t["chunk_id"]: t.get("triage", "?") for t in load_jsonl(out / "triage.jsonl")}

    metrics_path = out / "metrics.jsonl"
    done: set[str] = set()
    if metrics_path.exists():
        done = {r["note_id"] for r in load_jsonl(metrics_path)}
        print(f"уже посчитано: {len(done)} строк — они будут пропущены")

    print(f"чанков {len(chunks)}, заметок {len(notes)}, экзаменов {len(exams)}")
    print(f"Загружаю {a.model}…")
    scorer = HFScorer(a.model, device=a.device, dtype=a.dtype)

    # --- гейт пригодности, по модельной заметке ---------------------------
    model_note = {n["chunk_id"]: n["text"] for n in notes if n["kind"] == "model"}
    baselines: dict[str, tuple[float, int]] = {}
    unmeasurable: set[str] = set()
    for cid, exam in exams.items():
        baselines[cid] = exam_baseline(scorer, exam)
        text = model_note.get(cid)
        if text is None:
            continue
        if not chunk_is_measurable(baselines[cid][0], scorer.score(text).bits):
            unmeasurable.add(cid)
    print(f"непригодны по headroom < {MIN_EXAM_HEADROOM}: {len(unmeasurable)} чанков")

    # --- метрики ----------------------------------------------------------
    chunk_scores: dict[str, object] = {}
    written = 0
    with metrics_path.open("a", encoding="utf-8") as fh:
        for j, nr in enumerate(notes, 1):
            cid = nr["chunk_id"]
            if nr["note_id"] in done or cid in unmeasurable:
                continue
            exam = exams.get(cid)
            text = chunks.get(cid)
            if not exam or text is None:
                continue
            if cid not in chunk_scores:
                chunk_scores[cid] = scorer.score(text)

            tm = score_note_on_exam(
                scorer, nr["text"], exam, note_id=nr["note_id"], chunk_id=cid,
                kind=nr["kind"], lam=a.lam, cached_baseline=baselines[cid],
            )
            sm = score_note(
                scorer, text, nr["text"], note_id=nr["note_id"], chunk_id=cid,
                kind=nr["kind"], lam=a.lam, cached_chunk_score=chunk_scores[cid],
            )
            row = tm.to_dict()
            row.update({
                "mdl_gain_bits": sm.mdl_gain_bits,
                "savings_bits_surface": sm.savings_bits,
                "surface_savings_frac": sm.savings_frac,
                "verbatim_overlap": verbatim_overlap(nr["text"], text),
                "triage": labels.get(cid, "?"),
            })
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()  # чтобы обрыв стоил одной строки, а не всего хвоста
            written += 1
            if written % 20 == 0:
                print(f"      {written} строк ({j}/{len(notes)} заметок)")

    # Дедупликация. Файл открыт на дозапись, а множество `done` читается один
    # раз на старте — два одновременных запуска увидят пустой файл и запишут
    # каждую строку дважды. Ровно это и случилось: 264 строки вместо 132.
    # Значения совпали побитово (скоринг детерминирован), так что чистка
    # безопасна; оставляем последнюю версию каждого note_id.
    rows = load_jsonl(metrics_path)
    uniq: dict[str, dict] = {r["note_id"]: r for r in rows}
    if len(uniq) != len(rows):
        print(f"дубликаты: {len(rows)} строк -> {len(uniq)} уникальных, чищу")
        tmp = metrics_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in uniq.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(metrics_path)

    total = len(load_jsonl(metrics_path))
    print(f"\nГотово: {metrics_path} ({total} строк, дописано {written})")
    print(f"\nДальше:  python -m kstudy.analyze {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
