#!/usr/bin/env python3
"""
Забирает только нужные каталоги ядра. Работает на Windows нативно.

    python scripts/fetch_corpus.py --subsystem rcu --dest ./linux

Зачем не обычный `git clone`. Полное дерево ядра на Windows не выкачивается
корректно по трём причинам:

  1. Коллизии регистра. В netfilter лежат пары файлов, различающиеся только
     регистром: xt_CONNMARK.h / xt_connmark.h, xt_DSCP.h / xt_dscp.h,
     xt_MARK.h / xt_mark.h и другие. На регистронезависимой NTFS второй файл
     затирает первый, и git считает дерево грязным сразу после клонирования.
  2. Длинные пути. Часть путей в ядре длиннее 260 символов.
  3. Перевод строк. autocrlf по умолчанию превращает LF в CRLF. Для нас это
     не косметика: лишний символ на строку меняет токенизацию и число бит,
     то есть портит ровно те величины, которые мы меряем.

Разреженная выборка снимает (1) и (2) целиком — netfilter просто не
скачивается, — а (3) выключается явно. Заодно вместо ~5 ГБ получается
несколько мегабайт.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kstudy._compat import enable_utf8_console  # noqa: E402
from kstudy.corpus import SUBSYSTEMS  # noqa: E402

REPO = "https://github.com/torvalds/linux.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True)
    if r.returncode != 0:
        raise SystemExit(f"команда завершилась с кодом {r.returncode}")


def main() -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsystem", default="rcu", choices=list(SUBSYSTEMS))
    ap.add_argument("--dest", default="./linux")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--force", action="store_true", help="перезаписать каталог")
    a = ap.parse_args()

    if shutil.which("git") is None:
        print("git не найден. Windows: winget install Git.Git")
        return 1

    dest = Path(a.dest).resolve()
    if dest.exists():
        if not a.force:
            print(f"{dest} уже существует. Добавьте --force, чтобы перезаписать.")
            return 1
        shutil.rmtree(dest)

    spec = SUBSYSTEMS[a.subsystem]
    paths = sorted({*spec["docs"], *spec["code"]})

    print(f"Подсистема: {a.subsystem}")
    print(f"Каталоги:   {', '.join(paths)}")
    print(f"Назначение: {dest}\n")

    # Один коммит, без блобов, без рабочего дерева на старте.
    run([
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        "--no-checkout",
        # LF сохраняем как есть — иначе поедут все замеры в битах
        "-c", "core.autocrlf=false",
        "-c", "core.eol=lf",
        # На всякий случай, если кто-то расширит выборку
        "-c", "core.longpaths=true",
        a.repo, str(dest),
    ])

    run(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=dest)
    run(["git", "checkout"], cwd=dest)

    # Проверяем, что материал действительно на месте
    print()
    total = 0
    missing = []
    for rel in paths:
        p = dest / rel
        if not p.exists():
            missing.append(rel)
            continue
        size = (
            p.stat().st_size
            if p.is_file()
            else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        )
        total += size
        print(f"  ✓ {rel:44} {size / 1024:8.0f} КБ")

    for rel in missing:
        print(f"  ✗ {rel:44} НЕ СКАЧАЛОСЬ")

    print(f"\nИтого {total / 1024 / 1024:.1f} МБ")
    if missing:
        print("Часть путей отсутствует — возможно, они переехали в свежем ядре.")
        print("Поправьте SUBSYSTEMS в kstudy/corpus.py.")
        return 1

    print(f"\nДальше:  python run_e1.py --kernel {dest} --subsystem {a.subsystem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
