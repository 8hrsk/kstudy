#!/usr/bin/env bash
# То же самое для WSL2 / Linux / macOS.
#
#   bash setup_repo.sh [имя-репозитория] [--private] [--dry-run]
#
set -euo pipefail
cd "$(dirname "$0")"

REPO="${1:-kstudy}"
[[ "${REPO}" == --* ]] && REPO="kstudy"
VIS="--public"
DRY=0
for arg in "$@"; do
    [[ "$arg" == "--private" ]] && VIS="--private"
    [[ "$arg" == "--dry-run" ]] && DRY=1
done

step() { printf '\n\033[36m=== %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m  OK  %s\033[0m\n' "$1"; }
warn() { printf '\033[33m  !!  %s\033[0m\n' "$1"; }

step "Проверяю инструменты"
command -v git >/dev/null || { echo "git не найден"; exit 1; }
ok "git $(git --version | sed 's/git version //')"
command -v gh >/dev/null || {
    echo "gh не найден."
    echo "  Ubuntu/WSL: sudo apt install gh   |   macOS: brew install gh"
    exit 1
}
if [[ $DRY -eq 0 ]]; then
    gh auth status >/dev/null 2>&1 || { warn "нужен вход в GitHub"; gh auth login; }
    ok "gh авторизован"
fi

step "Убираю мусор из песочницы"
# .git остался в нерабочем состоянии: он создавался в ФС без права unlink,
# поэтому там висит несбрасываемый index.lock.
for junk in .git .probe .DS_Store; do
    [[ -e "$junk" ]] && rm -rf "$junk" && ok "удалено: $junk"
done
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

step "Гоняю тесты"
python3 tests/test_metrics.py

step "Создаю коммит"
git init -q -b main
git config core.autocrlf false          # LF влияет на токенизацию и подсчёт бит
git config i18n.commitEncoding utf-8

# Личность автора: берём глобальную, если есть; иначе спрашиваем.
NAME="$(git config --global user.name  || true)"
EMAIL="$(git config --global user.email || true)"
if [[ -z "$EMAIL" ]]; then
    warn "глобальный git user.email не задан"
    read -rp "  ваш email для коммитов: " EMAIL
    read -rp "  ваше имя [${REPO%%-*}]: " NAME
    NAME="${NAME:-8hoursking}"
fi
git config user.name  "$NAME"
git config user.email "$EMAIL"
ok "автор: $NAME <$EMAIL>"

git add -A
git commit -q -F - <<'MSG'
Э1: метрики обучения без эталона

Стенд для одного вопроса: можно ли численно отличить заметку, в которой
модель что-то поняла, от заметки, в которой она переписала слова, — не имея
правильного ответа.

Метрика — закрытый экзамен:
    выигрыш = [L(A|Q) - L(A|Q,N)] - lambda*L(N)
Вопросы составлены по чанку, отвечать надо без чанка. lambda=1 — честный MDL,
заметка платит за себя собственной длиной кода.

Первая версия метрики (MDL на поверхности текста) не пережила тестов: дословная
копия там почти оптимальна как код — выигрыш -27 бит против -227 у осмысленной
сжатой заметки. Отсюда переход на экзамен, где польза насыщается, а цена растёт
с длиной.

Вторая находка: ответы экзамена обязаны быть перефразированы, иначе метрика
меряет запоминание, а не понимание. Детектор утечки в run_e1.py.

19 инвариантов, гоняются без GPU за секунды.
MSG
ok "коммит: $(git log --oneline -1)   файлов: $(git ls-files | wc -l)"

if [[ $DRY -eq 1 ]]; then
    step "dry-run — на GitHub ничего не создаю"
    echo "  gh repo create $REPO --source=. --push $VIS"
    exit 0
fi

step "Создаю репозиторий на GitHub"
gh repo create "$REPO" --source=. --push $VIS \
    --description "Э1: измерение того, поняла ли модель прочитанное, без эталонного ответа. MDL на закрытом экзамене."
ok "готово: $(gh repo view --json url -q .url)"

printf '\n\033[36mДальше:\033[0m\n'
echo "  python3 scripts/fetch_corpus.py --subsystem rcu --dest ./linux"
echo "  python3 scripts/smoke_gpu.py --model Qwen/Qwen3-1.7B"
