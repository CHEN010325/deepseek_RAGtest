param(
  [string]$Dataset = "zh_int",
  [int]$Limit = 0,
  [string]$ApiBase = "http://127.0.0.1:3335",
  [string]$MeiliBase = "http://127.0.0.1:7775",
  [string]$KbLabel = "1",
  [string]$Mode = "retrieval",
  [string]$SystemName = "current",
  [string]$SearchMode = "",
  [string]$SearchModel = "",
  [string]$AnswerModel = "mimo-v2.5-pro",
  [string]$AnswerBackend = "mimo",
  [int]$OllamaNumCtx = 32768,
  [int]$OllamaTimeout = 900,
  [int]$OllamaNumPredict = 512,
  [bool]$OllamaNoThink = $true,
  [string]$MimoApiKey = "",
  [string]$DocIds = "",
  [string]$AppDir = "E:\deepseekmine",
  [string]$Python = "python",
  [switch]$InstallDeps,
  [switch]$SkipStart,
  [switch]$SkipUpload,
  [switch]$KeepKb
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Test-ApiReady {
  param([string]$BaseUrl)
  try {
    Invoke-WebRequest -Uri "$BaseUrl/api/search?query=ping&knowledgeLabel=$KbLabel" -Method GET -TimeoutSec 3 | Out-Null
    return $true
  } catch {
    return $false
  }
}

Write-Host "[eval] workdir: $ScriptDir"
Write-Host "[eval] api: $ApiBase"
Write-Host "[eval] dataset: $Dataset"
Write-Host "[eval] mode: $Mode"

if (-not $SkipStart -and -not (Test-ApiReady $ApiBase)) {
  Write-Host "[eval] app is not running, starting npm run dev:next"
  $AppLog = Join-Path $LogDir "deepseekmine-dev.log"
  $Command = "Set-Location '$AppDir'; npm run dev:next *> '$AppLog'"
  Start-Process powershell -WindowStyle Hidden -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $Command
  ) | Out-Null

  $Ready = $false
  for ($i = 1; $i -le 90; $i++) {
    Start-Sleep -Seconds 2
    if (Test-ApiReady $ApiBase) {
      $Ready = $true
      break
    }
    Write-Host "[eval] waiting for app... $($i * 2)s"
  }

  if (-not $Ready) {
    throw "app startup timeout, see log: $AppLog"
  }
}

if ($InstallDeps) {
  Write-Host "[eval] installing minimal dependencies: requests numpy tqdm"
  & $Python -m pip install requests numpy tqdm
}

$env:RAG_API_BASE = $ApiBase
$env:RAG_KB_LABEL = $KbLabel
$env:RAG_USE_ADAPTIVE = "true"
if ($SearchMode) {
  $env:RAG_SEARCH_MODE = $SearchMode
}
if ($SearchModel) {
  $env:RAG_SEARCH_MODEL = $SearchModel
}
$env:RAG_ANSWER_MODEL = $AnswerModel
$env:RAG_ANSWER_BACKEND = $AnswerBackend
$env:RAG_OLLAMA_NUM_CTX = "$OllamaNumCtx"
$env:RAG_OLLAMA_TIMEOUT = "$OllamaTimeout"
$env:RAG_OLLAMA_NUM_PREDICT = "$OllamaNumPredict"
$env:RAG_OLLAMA_NO_THINK = if ($OllamaNoThink) { "true" } else { "false" }
if ($MimoApiKey) {
  $env:MIMO_API_KEY = $MimoApiKey
}
$env:MEILI_DEV_BASE = $MeiliBase

$ArgsList = @("run_current_retrieval_eval.py", "--dataset", $Dataset, "--mode", $Mode, "--system-name", $SystemName)
if ($Limit -gt 0) {
  $ArgsList += @("--limit", "$Limit")
}
if ($SkipUpload) {
  $ArgsList += "--skip-upload"
  $ArgsList += @("--doc-ids", $DocIds)
}
if ($KeepKb) {
  $ArgsList += "--keep-kb"
}

Push-Location $ScriptDir
try {
  Write-Host "[eval] start retrieval evaluation"
  & $Python @ArgsList
  if ($LASTEXITCODE -ne 0) {
    throw "evaluation script exit code: $LASTEXITCODE"
  }
} finally {
  Pop-Location
}
