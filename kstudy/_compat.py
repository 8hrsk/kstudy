"""Мелочи совместимости с Windows. Импортируется всеми точками входа."""

from __future__ import annotations

import sys
from pathlib import Path


def enable_utf8_console() -> None:
    """
    Консоль Windows по умолчанию отдаёт cp1251/cp1252, и печать рамок, стрелок
    и галочек падает с UnicodeEncodeError — особенно при перенаправлении вывода
    в файл. Вызывается в начале каждого скрипта.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def posix_path(path: Path, root: Path) -> str:
    """
    Относительный путь всегда через прямые слэши.

    Не косметика: путь входит в chunk_id через хеш. Без нормализации
    один и тот же файл даёт разные id на Windows и Linux, и результаты
    прогонов перестают сравниваться между машинами.
    """
    return path.relative_to(root).as_posix()
