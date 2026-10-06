param(
  [string]$Manifest = "$PSScriptRoot\wrx80_glm53_native_series.json",
  [switch]$NoWaitForIdle
)

$ErrorActionPreference = "Stop"

function Write-JsonFile($Path, $Object) {
  $Object | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $Path
}

function Wait-ForIdle {
  if ($NoWaitForIdle) { return }
  while (Get-Process glm53 -ErrorAction SilentlyContinue) {
    Write-Host "[marshal] glm53 already active; waiting 10s"
    Start-Sleep -Seconds 10
  }
}

function Assert-Preconditions($Run) {
  if ($Run.precondition_file) {
    while (-not (Test-Path $Run.precondition_file)) {
      Write-Host "[marshal] waiting for precondition file $($Run.precondition_file)"
      Start-Sleep -Seconds 10
    }
    $text = Get-Content $Run.precondition_file -Raw
    foreach ($pattern in @($Run.precondition_require)) {
      if ($text -notmatch $pattern) { throw "precondition failed: required pattern not found: $pattern" }
    }
    foreach ($pattern in @($Run.precondition_forbid)) {
      if ($text -match $pattern) { throw "precondition failed: forbidden pattern found: $pattern" }
    }
  }
}

function Sample-Run($Process, $CsvPath, [int]$IntervalMs) {
  "timestamp,gpu_util_pct,power_w,mem_used_mib,mem_free_mib,cpu_s,working_set_bytes,private_bytes" | Set-Content -Encoding ASCII $CsvPath
  while (-not $Process.HasExited) {
    $g = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.free --format=csv,noheader,nounits) -split ","
    $Process.Refresh()
    $line = @($g[0].Trim(), $g[1].Trim(), $g[2].Trim(), $g[3].Trim(), $g[4].Trim(), $Process.CPU, $Process.WorkingSet64, $Process.PrivateMemorySize64) -join ","
    Add-Content -Encoding ASCII $CsvPath $line
    Start-Sleep -Milliseconds $IntervalMs
  }
}

$cfg = Get-Content $Manifest -Raw | ConvertFrom-Json
$repo = $cfg.repo
$model = $cfg.model
$glm = Join-Path $repo "c\glm53.exe"
$root = $cfg.results_root
New-Item -ItemType Directory -Force $root | Out-Null

$seriesStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$seriesDir = Join-Path $root "series-$seriesStamp"
New-Item -ItemType Directory -Force $seriesDir | Out-Null
Copy-Item $Manifest (Join-Path $seriesDir "manifest.json")
git -C $repo rev-parse HEAD | Set-Content -Encoding ASCII (Join-Path $seriesDir "git-head.txt")
git -C $repo status --short --branch | Set-Content -Encoding UTF8 (Join-Path $seriesDir "git-status.txt")

$seriesSummary = @()
foreach ($run in $cfg.runs) {
  if ($run.enabled -eq $false) { continue }
  Wait-ForIdle
  Assert-Preconditions $run

  $runDir = Join-Path $seriesDir $run.name
  New-Item -ItemType Directory -Force $runDir | Out-Null

  foreach ($p in $cfg.base_env.PSObject.Properties) { Set-Item -Path "Env:$($p.Name)" -Value ([string]$p.Value) }
  foreach ($p in $run.env.PSObject.Properties) { Set-Item -Path "Env:$($p.Name)" -Value ([string]$p.Value) }

  $prompt = (($cfg.prompt_unit * $cfg.prompt_repeat).Substring(0, $cfg.prompt_chars))
  $stdout = Join-Path $runDir "stdout.log"
  $stderr = Join-Path $runDir "stderr.log"
  $samples = Join-Path $runDir "telemetry.csv"
  $metadata = [ordered]@{ name=$run.name; start=(Get-Date).ToString("o"); greedy=[int]$run.greedy; env=@{} }
  foreach ($p in $cfg.base_env.PSObject.Properties) { $metadata.env[$p.Name] = [string]$p.Value }
  foreach ($p in $run.env.PSObject.Properties) { $metadata.env[$p.Name] = [string]$p.Value }
  Write-JsonFile (Join-Path $runDir "start.json") $metadata

  Write-Host "[marshal] starting $($run.name)"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $procArgs = @("--model", $model, "--prompt", $prompt, "--greedy", [string]$run.greedy)
  $proc = Start-Process -FilePath $glm -WorkingDirectory (Join-Path $repo "c") -NoNewWindow -PassThru -ArgumentList $procArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr

  Sample-Run $proc $samples ([int]$cfg.sample_interval_ms)
  $proc.WaitForExit()
  $sw.Stop()

  $allText = ""
  if (Test-Path $stdout) { $allText += (Get-Content $stdout -Raw) }
  if (Test-Path $stderr) { $allText += [Environment]::NewLine + (Get-Content $stderr -Raw) }

  $ok = ($proc.ExitCode -eq 0)
  $failures = @()
  foreach ($pattern in @($run.require)) {
    if ($allText -notmatch $pattern) { $ok = $false; $failures += "missing:$pattern" }
  }
  foreach ($pattern in @($run.forbid)) {
    if ($allText -match $pattern) { $ok = $false; $failures += "forbidden:$pattern" }
  }

  $done = [ordered]@{ name=$run.name; start=$metadata.start; end=(Get-Date).ToString("o"); elapsed_s=[math]::Round($sw.Elapsed.TotalSeconds,3); exit_code=$proc.ExitCode; ok=$ok; failures=$failures }
  Write-JsonFile (Join-Path $runDir "result.json") $done
  $seriesSummary += [pscustomobject]$done

  Write-Host "[marshal] $($run.name) exit=$($proc.ExitCode) ok=$ok elapsed=$($done.elapsed_s)s"
  if (-not $ok -and $run.stop_on_failure -ne $false) {
    Write-Host "[marshal] stopping series after failed gate"
    break
  }
}

$seriesSummary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $seriesDir "summary.json")
$seriesSummary | Format-Table -AutoSize | Out-String | Set-Content -Encoding UTF8 (Join-Path $seriesDir "summary.txt")
Write-Host "[marshal] series complete: $seriesDir"
