"""
Разметка заметок вручную. Сотня примеров — примерно час работы.

    python -m kstudy.rate data/notes.jsonl data/chunks.jsonl --out data/ratings.jsonl

Два правила, без которых разметка бесполезна:

  1. Порядок случайный, тип заметки СКРЫТ. Иначе вы будете ставить копиям
     высокие оценки просто потому, что знаете, что это копия, и корреляция
     окажется артефактом.
  2. Оценивать надо ответ на один вопрос: «если бы источник исчез, а осталась
     только эта заметка, много ли я потерял бы?» Не стиль, не полноту, не
     красоту — только это.

Прогресс сохраняется после каждой оценки, прервать можно в любой момент.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ._compat import enable_utf8_console

SCALE = """
  1  бесполезна — вода или не про этот текст
  2  почти бесполезна — пара крох, остальное потеряно
  3  наполовину — главное схвачено, деталей нет
  4  хорошо — сохраняет механизм и причины
  5  отлично — сохраняет всё существенное и короче источника

  s  пропустить    q  выйти и сохранить
"""


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    # До ArgumentParser: справка и ошибки argparse печатаются внутри
    # parse_args(), до этой строки консоль ещё в кодировке локали.
    enable_utf8_console()

    ap = argparse.ArgumentParser()
    ap.add_argument("notes_jsonl")
    ap.add_argument("chunks_jsonl")
    ap.add_argument("--out", default="data/ratings.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chunk-chars", type=int, default=1800)
    a = ap.parse_args()

    notes = load_jsonl(a.notes_jsonl)
    chunks = {c["chunk_id"]: c for c in load_jsonl(a.chunks_jsonl)}

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["note_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()

    todo = [n for n in notes if n["note_id"] not in done]
    random.Random(a.seed).shuffle(todo)   # тип заметки не должен быть предсказуем
    todo = todo[: a.limit]

    if not todo:
        print(f"Всё размечено ({len(done)} записей в {out_path})")
        return 0

    print(f"К разметке: {len(todo)}   уже готово: {len(done)}")
    print(SCALE)

    fh = out_path.open("a", encoding="utf-8")
    n_rated = 0
    try:
        for i, note in enumerate(todo, 1):
            chunk = chunks.get(note["chunk_id"])
            if not chunk:
                continue
            text = chunk["text"]
            clipped = text[: a.max_chunk_chars]

            print("\n" + "═" * 72)
            print(f"[{i}/{len(todo)}]  {chunk.get('source_path','')}  "
                  f"{chunk.get('heading','')}")
            print("═" * 72)
            print("\n--- ИСТОЧНИК " + "-" * 58)
            print(clipped + ("\n…(обрезано)" if len(text) > len(clipped) else ""))
            print("\n--- ЗАМЕТКА " + "-" * 59)
            print(note["text"])
            print("-" * 72)

            while True:
                ans = input("Если бы источник исчез, много ли потеряно? [1-5/s/q] ").strip().lower()
                if ans == "q":
                    raise KeyboardInterrupt
                if ans == "s":
                    break
                if ans in {"1", "2", "3", "4", "5"}:
                    fh.write(json.dumps({
                        "note_id": note["note_id"],
                        "chunk_id": note["chunk_id"],
                        "rating": int(ans),
                    }) + "\n")
                    fh.flush()
                    n_rated += 1
                    break
                print("  1-5, s — пропустить, q — выйти")
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано.")
    finally:
        fh.close()

    print(f"\nРазмечено за сессию: {n_rated}. Всего: {len(done) + n_rated} -> {out_path}")
    print("Дальше:  python -m kstudy.analyze data/metrics.jsonl --ratings", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
