"""
Корпус: подсистема ядра Linux, нарезанная на чанки.

Нарезка идёт по смысловым границам, а не по фиксированному числу токенов:
у .rst — по заголовкам разделов, у .c/.h — по функциям и блокам комментариев.
Это важно, потому что L_base на чанке, разрезанном посреди фразы, меряет
качество нарезки, а не новизну материала.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from ._compat import enable_utf8_console, posix_path

# --------------------------------------------------------------------------
# Пресеты подсистем
# --------------------------------------------------------------------------

SUBSYSTEMS: dict[str, dict] = {
    # Рекомендация по умолчанию: превосходная документация, известная
    # тонкость, и есть LKMM/herd7 как машинный верификатор для Э4.
    "rcu": {
        "docs": ["Documentation/RCU"],
        "code": ["kernel/rcu", "include/linux/rcupdate.h", "include/linux/rcutree.h"],
        "note": "LKMM (tools/memory-model) даёт herd7 как верификатор порядка памяти",
    },
    "block": {
        "docs": ["Documentation/block"],
        "code": ["block"],
        "note": "крупнее, ближе к практике, верификация через blktests",
    },
    "slub": {
        "docs": ["Documentation/mm/slub.rst", "Documentation/mm/slab.rst"],
        "code": ["mm/slub.c", "mm/slab_common.c", "include/linux/slab.h"],
        "note": "компактно и самодостаточно, легко собрать юнит-тесты в userspace",
    },
    "workqueue": {
        "docs": ["Documentation/core-api/workqueue.rst"],
        "code": ["kernel/workqueue.c", "include/linux/workqueue.h"],
        "note": "маленький корпус, хорош для быстрой отладки пайплайна",
    },
    "memory-model": {
        "docs": ["tools/memory-model/Documentation"],
        "code": ["tools/memory-model/litmus-tests"],
        "note": "литмус-тесты машинно проверяемы herd7 — идеально для ветки A в Э4",
    },
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_path: str
    kind: str          # doc | code
    heading: str = ""
    start_line: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Нарезка
# --------------------------------------------------------------------------

# Заголовок .rst — строка, под которой идёт линия из одинаковых знаков
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+]){3,}\s*$")
C_FUNC_START = re.compile(
    r"^(?:static\s+|inline\s+|__always_inline\s+|noinline\s+)*"
    r"[A-Za-z_][\w\s\*]*\**\s*[A-Za-z_]\w*\s*\([^;]*$"
)


def _mk_id(path: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{path}:{idx}:{text[:200]}".encode()).hexdigest()[:10]
    return f"{Path(path).stem}-{idx:04d}-{h}"


def split_rst(text: str, path: str) -> Iterator[Chunk]:
    lines = text.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    cur_head, cur_start, buf = "", 0, []

    for i, line in enumerate(lines):
        is_head = (
            i + 1 < len(lines)
            and RST_UNDERLINE.match(lines[i + 1] or "")
            and line.strip()
            and len(lines[i + 1].strip()) >= len(line.strip()) - 2
        )
        if is_head:
            if buf:
                sections.append((cur_head, cur_start, buf))
            cur_head, cur_start, buf = line.strip(), i + 1, []
        else:
            buf.append(line)
    if buf:
        sections.append((cur_head, cur_start, buf))

    for idx, (head, start, body) in enumerate(sections):
        body_text = "\n".join(body).strip()
        if len(body_text) < 200:  # огрызки склеивать не с чем — пропускаем
            continue
        full = f"{head}\n\n{body_text}" if head else body_text
        yield Chunk(_mk_id(path, idx, full), full, path, "doc", head, start)


def split_c(text: str, path: str, max_chars: int = 6000) -> Iterator[Chunk]:
    """Наивная нарезка по функциям: от сигнатуры до закрывающей скобки в нулевой колонке."""
    lines = text.splitlines()
    blocks: list[tuple[str, int, list[str]]] = []
    i, n = 0, len(lines)

    while i < n:
        if C_FUNC_START.match(lines[i]):
            start = i
            # Прихватываем комментарий-шапку над функцией — там половина смысла.
            j = i - 1
            while j >= 0 and (
                lines[j].strip().startswith(("*", "/*", "//")) or not lines[j].strip()
            ):
                if lines[j].strip().startswith("/*"):
                    start = j
                    break
                j -= 1
            # Ищем закрывающую скобку в нулевой колонке
            k, depth, opened = i, 0, False
            while k < n:
                depth += lines[k].count("{") - lines[k].count("}")
                if "{" in lines[k]:
                    opened = True
                if opened and depth <= 0:
                    break
                k += 1
            if opened:
                name = lines[i].strip()[:80]
                blocks.append((name, start, lines[start : k + 1]))
                i = k + 1
                continue
        i += 1

    for idx, (name, start, body) in enumerate(blocks):
        body_text = "\n".join(body).strip()
        if len(body_text) < 150:
            continue
        yield Chunk(
            _mk_id(path, idx, body_text),
            body_text[:max_chars],
            path,
            "code",
            name,
            start,
        )


def load_subsystem(
    kernel_root: str | Path,
    subsystem: str = "rcu",
    include_code: bool = True,
) -> list[Chunk]:
    """
    kernel_root — путь к дереву ядра. Забирается разреженно:
        python scripts/fetch_corpus.py --subsystem rcu --dest ./linux
    (обычный полный clone на Windows ломается — см. WINDOWS.md)
    """
    root = Path(kernel_root)
    if not root.exists():
        raise FileNotFoundError(f"нет дерева ядра: {root}")
    if subsystem not in SUBSYSTEMS:
        raise KeyError(f"неизвестная подсистема {subsystem!r}; есть: {list(SUBSYSTEMS)}")

    spec = SUBSYSTEMS[subsystem]
    targets = list(spec["docs"]) + (list(spec["code"]) if include_code else [])
    chunks: list[Chunk] = []

    for rel in targets:
        p = root / rel
        files = (
            [p]
            if p.is_file()
            else sorted(
                f
                for f in p.rglob("*")
                if f.is_file() and f.suffix in {".rst", ".txt", ".c", ".h", ".litmus"}
            )
            if p.is_dir()
            else []
        )
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relpath = posix_path(f, root)
            if f.suffix in {".c", ".h"}:
                chunks.extend(split_c(text, relpath))
            else:
                chunks.extend(split_rst(text, relpath))

    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


if __name__ == "__main__":
    import argparse

    enable_utf8_console()
    ap = argparse.ArgumentParser(description="Нарезать подсистему ядра на чанки")
    ap.add_argument("kernel_root")
    ap.add_argument("--subsystem", default="rcu", choices=list(SUBSYSTEMS))
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--no-code", action="store_true")
    a = ap.parse_args()

    cs = load_subsystem(a.kernel_root, a.subsystem, include_code=not a.no_code)
    save_chunks(cs, a.out)

    docs = sum(c.kind == "doc" for c in cs)
    chars = sum(len(c.text) for c in cs)
    print(f"{len(cs)} чанков ({docs} doc / {len(cs) - docs} code), "
          f"{chars / 1e6:.2f} МБ  ->  {a.out}")
    print(f"подсказка: {SUBSYSTEMS[a.subsystem]['note']}")
