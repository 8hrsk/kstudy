<#
    Создаёт репозиторий и пушит на GitHub. Запускать в этой папке:

        powershell -ExecutionPolicy Bypass -File .\setup_repo.ps1

    Параметры:
        -RepoName   имя репозитория (по умолчанию kstudy)
        -Private    сделать приватным
        -DryRun     всё подготовить, но не создавать репозиторий на GitHub

    Почему это скриптом, а не сделано за вас: песочница, в которой собирался
    проект, смонтирована без права удаления файлов, поэтому git там не
    работает — не может снять свой же index.lock. Плюс в ней нет gh и нет
    авторизации GitHub. Здесь и то и другое есть.
#>

param(
    [string]$RepoName = "kstudy",
    [switch]$Private,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }

# --- проверки ---------------------------------------------------------------
Step "Проверяю инструменты"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git не найден.  winget install Git.Git"
}
Ok "git $((git --version) -replace 'git version ','')"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "gh не найден.  winget install GitHub.cli"
}
Ok "gh найден"

if (-not $DryRun) {
    gh auth status 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Warn "gh не авторизован. Запускаю вход — понадобится браузер."
        gh auth login
        if ($LASTEXITCODE -ne 0) { throw "авторизация не прошла" }
    }
    Ok "gh авторизован"
}

# --- уборка после песочницы -------------------------------------------------
Step "Убираю мусор из песочницы"

# .git с зависшим index.lock: инициализирован в ФС без права unlink и
# остался в нерабочем состоянии. Проще снести и создать заново.
foreach ($junk in @(".git", ".probe", ".DS_Store", "__pycache__")) {
    if (Test-Path $junk) {
        Remove-Item -Recurse -Force $junk
        Ok "удалено: $junk"
    }
}
Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# --- проверка перед коммитом ------------------------------------------------
Step "Гоняю тесты"
python tests\test_metrics.py
if ($LASTEXITCODE -ne 0) { throw "тесты не прошли — коммитить нечего" }

# --- репозиторий ------------------------------------------------------------
Step "Создаю коммит"

git init -q -b main

# Личность автора: берём глобальную, если есть; иначе спрашиваем.
$name  = (git config --global user.name)  2>$null
$email = (git config --global user.email) 2>$null
if (-not $email) {
    Warn "глобальный git user.email не задан"
    $email = Read-Host "  ваш email для коммитов"
    $name  = Read-Host "  ваше имя [8hoursking]"
    if (-not $name) { $name = "8hoursking" }
}
git config user.name  "$name"
git config user.email "$email"
Ok "автор: $name <$email>"

# LF сохраняем как есть: для этого проекта перевод строки влияет на
# токенизацию и на подсчёт бит, то есть на сами измеряемые величины.
git config core.autocrlf false
git config i18n.commitEncoding utf-8
git config i18n.logOutputEncoding utf-8

git add -A

$msg = @"
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
"@

$msgFile = Join-Path $env:TEMP "kstudy_commit_msg.txt"
[System.IO.File]::WriteAllText($msgFile, $msg, (New-Object System.Text.UTF8Encoding $false))
git commit -q -F $msgFile
Remove-Item $msgFile -Force

Ok "коммит: $(git log --oneline -1)"
Write-Host "  файлов: $((git ls-files | Measure-Object).Count)"

# --- публикация -------------------------------------------------------------
if ($DryRun) {
    Step "DryRun — на GitHub ничего не создаю"
    Write-Host "  Когда будете готовы:"
    Write-Host "    gh repo create $RepoName --source=. --push --$(if($Private){'private'}else{'public'})"
    exit 0
}

Step "Создаю репозиторий на GitHub"

$visibility = if ($Private) { "--private" } else { "--public" }
$desc = "Э1: измерение того, поняла ли модель прочитанное, без эталонного ответа. MDL на закрытом экзамене."

gh repo create $RepoName --source=. --push $visibility --description $desc
if ($LASTEXITCODE -ne 0) {
    Warn "gh repo create не отработал."
    Warn "Если репозиторий с таким именем уже есть, попробуйте:"
    Warn "  .\setup_repo.ps1 -RepoName kstudy-e1"
    throw "публикация не удалась"
}

$url = gh repo view --json url -q .url
Ok "готово: $url"

Write-Host "`nДальше:" -ForegroundColor Cyan
Write-Host "  python scripts\fetch_corpus.py --subsystem rcu --dest .\linux"
Write-Host "  python scripts\smoke_gpu.py --model Qwen/Qwen3-1.7B"
