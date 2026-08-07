<#
  collect_submission.ps1 — 학습 결과 제출물 수집기 (Windows / PowerShell)

  타 하드웨어에서 돌린 학습 run 중 "제출해야 할 파일만" 골라 한 폴더에 모으고 압축한다.
  GB급 감사 로그는 자동으로 제외한다. git 없이 파일만 직접 전달할 때 쓴다.

  사용법:
    .\collect_submission.ps1 -Run "output\phase1_anneal_v1" -Alias "P1-ANNEAL" -Phase 1
    .\collect_submission.ps1 -Run "output\phase2_fixedT10"   -Alias "P2-FIXT"   -Phase 2

  checkpoint 전량을 함께 보내려면 -WithCheckpoints 를 붙인다(수백 MB).
    .\collect_submission.ps1 -Run "output\phase1_anneal_v1" -Alias "P1-ANNEAL" -Phase 1 -WithCheckpoints

  결과: _submit\<ALIAS>\ 폴더와 _submit\<ALIAS>.zip
  규칙 전문: 모델_학습_비교_절대규칙.md §4
#>

param(
  [Parameter(Mandatory=$true)][string]$Run,
  [Parameter(Mandatory=$true)][string]$Alias,
  [Parameter(Mandatory=$true)][ValidateSet(1,2)][int]$Phase,
  [switch]$WithCheckpoints,
  [int]$HeadLines = 200,
  [int]$TailLines = 2000
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Run)) { throw "학습 폴더를 찾을 수 없습니다: $Run" }

$Out = Join-Path "_submit" $Alias
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Write-Host "[수집] $Run  ->  $Out" -ForegroundColor Cyan

# ---- 1. 지표·증빙 파일 -------------------------------------------------
if ($Phase -eq 1) {
  # Phase 1: 전부 run 루트에 있다. validation/ 디렉터리가 없다.
  $Files = @(
    "validation_summary.csv",        # ★ 비교에 쓰는 유일한 산출물
    "metrics.csv",
    "summary.json",
    "run_manifest.json",             # 2026-08-03 이후 시작한 run만 존재
    "subproblem_metrics.csv",
    "training_quick_status.json"
  )
} else {
  $Files = @("run_manifest.json", "summary.json", "metrics.csv")
}

foreach ($f in $Files) {
  $src = Join-Path $Run $f
  if (Test-Path $src) {
    Copy-Item $src (Join-Path $Out $f)
    Write-Host ("  + {0,-34} {1,8:N1} MB" -f $f, ((Get-Item $src).Length/1MB))
  } else {
    Write-Host ("  - {0,-34} 없음" -f $f) -ForegroundColor DarkYellow
  }
}

# Phase 2: validation 계열 3종 x 2 디렉터리
if ($Phase -eq 2) {
  foreach ($d in @("validation", "validation_generalization")) {
    $srcDir = Join-Path $Run $d
    if (-not (Test-Path $srcDir)) { Write-Host ("  - {0,-34} 없음" -f $d) -ForegroundColor DarkYellow; continue }
    New-Item -ItemType Directory -Force -Path (Join-Path $Out $d) | Out-Null
    foreach ($f in @("grid_contract.json","validation_parent_history.csv","validation_rank_history.csv")) {
      $src = Join-Path $srcDir $f
      if (Test-Path $src) {
        Copy-Item $src (Join-Path $Out (Join-Path $d $f))
        Write-Host ("  + {0,-34} {1,8:N1} MB" -f "$d\$f", ((Get-Item $src).Length/1MB))
      } else {
        Write-Host ("  - {0,-34} 없음" -f "$d\$f") -ForegroundColor DarkYellow
      }
    }
  }
}

# ---- 2. 로그 앞/뒤 ------------------------------------------------------
# manifest 없는 run은 로그 헤더가 seed/블록범위/hidden_dim 의 유일한 근거다.
$LogCandidates = @("launch_stdout.log","launch_stderr.log","$Alias.log","train.log","stdout.log") |
  ForEach-Object { Join-Path $Run $_ }
$LogCandidates += (Get-ChildItem -Path (Split-Path $Run -Parent) -Filter "*.log" -File -ErrorAction SilentlyContinue |
  Where-Object { $_.BaseName -eq (Split-Path $Run -Leaf) } | ForEach-Object { $_.FullName })

$foundLog = $false
foreach ($lg in $LogCandidates) {
  if (Test-Path $lg) {
    $base = [IO.Path]::GetFileNameWithoutExtension($lg)
    Get-Content $lg -TotalCount $HeadLines | Set-Content (Join-Path $Out "$base.head.log") -Encoding UTF8
    Get-Content $lg -Tail     $TailLines   | Set-Content (Join-Path $Out "$base.tail.log") -Encoding UTF8
    Write-Host ("  + {0,-34} head {1} / tail {2} 줄" -f "$base.head/.tail.log", $HeadLines, $TailLines) -ForegroundColor Green
    $foundLog = $true
  }
}
if (-not $foundLog) {
  Write-Host "  ! 로그 파일을 찾지 못했습니다. 로그 앞 200줄이 없으면" -ForegroundColor Red
  Write-Host "    seed / --min-blocks / --max-blocks / hidden_dim 을 확인할 수 없어 비교가 무효입니다." -ForegroundColor Red
  Write-Host "    로그 경로를 직접 찾아 수동으로 넣어주세요." -ForegroundColor Red
}

# ---- 3. checkpoint -----------------------------------------------------
$ckDir = Join-Path $Run "checkpoints"
if ($WithCheckpoints -and (Test-Path $ckDir)) {
  $dst = Join-Path $Out "checkpoints"
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  $cks = Get-ChildItem $ckDir -Filter "*.pt" -File
  foreach ($c in $cks) { Copy-Item $c.FullName (Join-Path $dst $c.Name) }
  Write-Host ("  + checkpoints/  {0}개  {1:N0} MB" -f $cks.Count, (($cks | Measure-Object Length -Sum).Sum/1MB)) -ForegroundColor Green
} elseif (Test-Path $ckDir) {
  $n = (Get-ChildItem $ckDir -Filter "*.pt" -File).Count
  Write-Host ("  . checkpoints/ {0}개 — 제외했습니다. 전부 보내려면 -WithCheckpoints 를 붙이세요." -f $n) -ForegroundColor DarkGray
}
# 최종·best 모델은 항상 참고용으로 포함 (작다)
foreach ($m in @("phase1_pair_pointer.pt","phase1_pair_pointer_best.pt","phase2_best.pt","phase2_batch_machine_policy.pt")) {
  $src = Join-Path $Run $m
  if (Test-Path $src) { Copy-Item $src (Join-Path $Out "_selfbest_$m"); Write-Host ("  + _selfbest_{0}" -f $m) }
}

# ---- 4. INFO.txt 양식 --------------------------------------------------
$infoPath = Join-Path $Out "INFO.txt"
if (-not (Test-Path $infoPath)) {
  $gitCommit = "미상"; $gitDirty = "미상"
  try {
    $gitCommit = (git rev-parse HEAD 2>$null)
    $gitDirty  = if ((git status --porcelain 2>$null)) { "예" } else { "아니오" }
  } catch { }
@"
별칭        : $Alias
학습 경로   : $Run
최종 episode:                       완주 여부: 예/아니오
git commit  : $gitCommit
dirty       : $gitDirty
실행 CLI    :
              (실제로 실행한 명령 전체를 한 줄로. 기억나지 않으면 로그 헤더에서 옮겨 적으세요)
비고        :
"@ | Set-Content $infoPath -Encoding UTF8
  Write-Host "  + INFO.txt  ← 최종 episode / 실행 CLI 를 직접 채워주세요" -ForegroundColor Yellow
}

# ---- 5. 체크섬 + 압축 ---------------------------------------------------
Get-ChildItem $Out -Recurse -File | Where-Object { $_.Name -ne "SHA256SUMS.txt" } | ForEach-Object {
  $rel = $_.FullName.Substring((Resolve-Path $Out).Path.Length + 1)
  "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $rel
} | Set-Content (Join-Path $Out "SHA256SUMS.txt") -Encoding UTF8

$zip = "_submit\$Alias.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$Out\*" -DestinationPath $zip

$total = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum/1MB
Write-Host ""
Write-Host ("[완료] {0}  ({1:N1} MB)  ->  {2}  ({3:N1} MB)" -f $Out, $total, $zip, ((Get-Item $zip).Length/1MB)) -ForegroundColor Cyan
Write-Host "INFO.txt 의 '최종 episode' 와 '실행 CLI' 를 채운 뒤 zip 을 전달하세요." -ForegroundColor Yellow
Write-Host ""
Write-Host "제외된 것(의도적): validation_candidate_summary.csv, candidate_summary.csv," -ForegroundColor DarkGray
Write-Host "  best_action_table.jsonl, *.png, validation*/problems, validation*/evaluations" -ForegroundColor DarkGray
