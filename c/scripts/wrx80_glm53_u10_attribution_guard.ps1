param(
  [string]$Manifest = "E:\z-src\colibri-native\c\scripts\wrx80_glm53_u10_chain_attribution.json"
)

$ErrorActionPreference = "Stop"
$expected = "47A50E117F6E3CC5DDC1EC490081006F3F33FD97CBD058C8D222B5BE7CACD6BA"
$usage = @(
  "E:\z-results\glm53-native-2026-10-06\u10-attribution\usage-seed.bin",
  "E:\z-results\glm53-native-2026-10-06\u10-attribution\usage-chain0.bin",
  "E:\z-results\glm53-native-2026-10-06\u10-attribution\usage-chain1.bin"
)

foreach ($p in $usage) {
  if (-not (Test-Path $p)) { throw "missing frozen usage file: $p" }
  $h = (Get-FileHash $p -Algorithm SHA256).Hash
  if ($h -ne $expected) { throw "usage hash mismatch: $p => $h" }
}

if (Get-Process glm53 -ErrorAction SilentlyContinue) {
  throw "Windows glm53 process is active"
}

$wslGlm = wsl.exe bash -lc "pgrep -a -x glm53 || true"
if ($wslGlm) {
  throw "WSL glm53 process is active: $wslGlm"
}

$gpu = (nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits).Trim().Split(",")
$used = [int]$gpu[0].Trim()
$free = [int]$gpu[1].Trim()
$util = [int]$gpu[2].Trim()
if ($free -lt 22000) {
  throw ("GPU not clear enough for attribution run: used={0}MiB free={1}MiB util={2}%" -f $used,$free,$util)
}

Write-Host ("[u10-attribution] preflight PASS: frozen usage intact, no Windows/WSL glm53, GPU free={0}MiB" -f $free)
& "E:\z-src\colibri-native\c\scripts\wrx80_glm53_native_series.ps1" -Manifest $Manifest
