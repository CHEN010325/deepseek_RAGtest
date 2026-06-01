param(
  [string[]]$Datasets = @("zh_int", "zh_refine"),
  [string]$CurrentApiBase = "http://127.0.0.1:3335",
  [string]$BaselineApiBase = "http://127.0.0.1:3336",
  [string]$MeiliBase = "http://127.0.0.1:7775",
  [string]$KbLabel = "1",
  [string]$Mode = "qa",
  [string]$AnswerModel = "mimo-v2.5-pro",
  [string]$AnswerBackend = "mimo",
  [string]$MimoApiKey = "",
  [string]$Python = "python",
  [string]$CurrentAppDir = "E:\deepseekmine",
  [string]$BaselineAppDir = "E:\deepseekmine-baseline",
  [string]$CurrentName = "current",
  [string]$BaselineName = "baseline",
  [int]$Limit = 0,
  [switch]$SkipStart
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartScript = Join-Path $ScriptDir "start_retrieval_eval.ps1"
$LogDir = Join-Path $ScriptDir "logs"
$ResultDir = Join-Path $ScriptDir "result-zh"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

if ($AnswerBackend -eq "mimo") {
  if ($MimoApiKey) {
    $env:MIMO_API_KEY = $MimoApiKey
  }
  if (-not $env:MIMO_API_KEY) {
    throw "MIMO_API_KEY is required. Set `$env:MIMO_API_KEY first, or pass -MimoApiKey."
  }
}

function Invoke-OneEval {
  param(
    [string]$Dataset,
    [string]$SystemName,
    [string]$ApiBase,
    [string]$AppDir,
    [string]$LogPath,
    [string]$DocIds = "",
    [switch]$UploadAndKeep
  )

  $argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $StartScript,
    "-Dataset", $Dataset,
    "-Mode", $Mode,
    "-SystemName", $SystemName,
    "-ApiBase", $ApiBase,
    "-MeiliBase", $MeiliBase,
    "-KbLabel", $KbLabel,
    "-AnswerModel", $AnswerModel,
    "-AnswerBackend", $AnswerBackend,
    "-AppDir", $AppDir,
    "-Python", $Python
  )

  if ($Limit -gt 0) {
    $argsList += @("-Limit", "$Limit")
  }
  if ($SkipStart) {
    $argsList += "-SkipStart"
  }
  if ($UploadAndKeep) {
    $argsList += "-KeepKb"
  } else {
    $argsList += @("-SkipUpload", "-DocIds", $DocIds)
  }

  Write-Host ""
  Write-Host "[总控] 开始：$SystemName / $Dataset / $ApiBase"
  Write-Host "[总控] 日志：$LogPath"

  Push-Location $ScriptDir
  try {
    & powershell @argsList 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
      throw "$SystemName / $Dataset failed, exit code: $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
}

function Get-UploadedDocId {
  param(
    [string]$Dataset,
    [string]$LogPath
  )

  $text = Get-Content -LiteralPath $LogPath -Raw -Encoding UTF8
  $pattern = "current_eval_$([regex]::Escape($Dataset))_\d+\.txt"
  $matches = [regex]::Matches($text, $pattern)
  if ($matches.Count -eq 0) {
    throw "Cannot find uploaded doc id in log: $LogPath"
  }
  return $matches[$matches.Count - 1].Value
}

function Get-NewestReport {
  param(
    [string]$Dataset,
    [string]$SystemName,
    [datetime]$After
  )

  $pattern = "${SystemName}_${Mode}_eval_${Dataset}_*.json"
  $file = Get-ChildItem -LiteralPath $ResultDir -Filter $pattern |
    Where-Object { $_.LastWriteTime -ge $After } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if (-not $file) {
    throw "Cannot find report: $pattern"
  }
  return $file.FullName
}

function Read-Summary {
  param([string]$ReportPath)

  $report = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
  return [pscustomobject]@{
    Dataset = $report.dataset
    System = $report.system_name
    Accuracy = [math]::Round($report.summary.qa_accuracy * 100, 2)
    Correct = $report.summary.qa_correct
    Total = $report.summary.qa_total
    Precision = [math]::Round($report.summary.precision, 4)
    Recall = [math]::Round($report.summary.recall, 4)
    MRR = [math]::Round($report.summary.mrr, 4)
    NDCG = [math]::Round($report.summary.ndcg, 4)
    AvgK = [math]::Round($report.summary.avg_k, 2)
    Report = $ReportPath
  }
}

$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$allRows = @()

foreach ($dataset in $Datasets) {
  $datasetStart = Get-Date
  $currentLog = Join-Path $LogDir "$CurrentName-$Mode-$dataset-$runStamp.log"
  $baselineLog = Join-Path $LogDir "$BaselineName-$Mode-$dataset-$runStamp.log"

  Invoke-OneEval `
    -Dataset $dataset `
    -SystemName $CurrentName `
    -ApiBase $CurrentApiBase `
    -AppDir $CurrentAppDir `
    -LogPath $currentLog `
    -UploadAndKeep

  $docId = Get-UploadedDocId -Dataset $dataset -LogPath $currentLog
  Write-Host "[总控] 复用文档：$docId"

  $currentReport = Get-NewestReport -Dataset $dataset -SystemName $CurrentName -After $datasetStart
  $allRows += Read-Summary -ReportPath $currentReport

  Invoke-OneEval `
    -Dataset $dataset `
    -SystemName $BaselineName `
    -ApiBase $BaselineApiBase `
    -AppDir $BaselineAppDir `
    -LogPath $baselineLog `
    -DocIds $docId

  $baselineReport = Get-NewestReport -Dataset $dataset -SystemName $BaselineName -After $datasetStart
  $allRows += Read-Summary -ReportPath $baselineReport
}

$summaryPath = Join-Path $ResultDir "rag_comparison_summary_$runStamp.csv"
$allRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "[总控] 完成，汇总如下："
$allRows | Format-Table Dataset, System, Accuracy, Correct, Total, Precision, Recall, MRR, NDCG, AvgK -AutoSize
Write-Host "[总控] 汇总文件：$summaryPath"
