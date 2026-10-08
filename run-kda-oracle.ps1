# Isolated native-Windows-only numerical A/B. No gateway or inference service changes.
$ErrorActionPreference = 'Stop'
$repo = 'C:\src\colibri-kda-batch-20261008'
$api = 'http://127.0.0.1:19000/control/status'
$state = Invoke-RestMethod -Uri $api -TimeoutSec 5
if ($state.active -or $state.switching -or $state.cancel_teardown_inflight -or @($state.waiters).Count -gt 0) {
  throw 'A live native inference request or switch is running: refuse concurrent KDA oracle'
}
$py = "$repo\.oracle-clean\Scripts\python.exe"
$toolbin = 'C:\tools\w64devkit-2.10.0\w64devkit\bin'
$make = "$toolbin\make.exe"
$fixture = "$repo\c\glm53_tiny"
$reportdir = "$repo\.oracle-proof"
if (!(Test-Path $py) -or !(Test-Path $make)) { throw 'oracle dependencies unavailable' }
New-Item -Path $reportdir -ItemType Directory -Force | Out-Null
$oldPath = $env:PATH
try {
  $env:PATH = "$toolbin;$oldPath"
  if (!(Test-Path "$fixture\ref.json")) {
    & $py "$repo\c\tools\make_glm53_tiny.py" --output $fixture *> "$reportdir\generator.log"
    if ($LASTEXITCODE -ne 0) { throw 'independent GLM53 tiny fixture generation failed' }
  }
  & $make -C "$repo\c" -j2 glm53.exe *> "$reportdir\build.log"
  if ($LASTEXITCODE -ne 0) { throw 'isolated GLM53 executable build failed' }
  & $py "$repo\c\tests\test_glm53_kda_batch_equivalence.py" --binary "$repo\c\glm53.exe" --fixture $fixture *> "$reportdir\oracle-result.json"
  if ($LASTEXITCODE -ne 0) { throw 'scalar/batched numerical oracle failed' }
  Get-Content "$reportdir\oracle-result.json" | Out-String
} finally {
  $env:PATH = $oldPath
}
