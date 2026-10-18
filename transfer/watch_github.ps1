param(
    [int]$IntervalSeconds = 30,
    [switch]$Once
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$env:GIT_TERMINAL_PROMPT = '0'

function Sync-FromGitHub {
    $branch = & git -C $repo branch --show-current 2>$null
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'main') { return }

    $fetchOutput = & git -C $repo fetch --quiet origin main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CV-1 sync: fetch failed; retrying later. $fetchOutput"
        return
    }

    $localCommit = & git -C $repo rev-parse HEAD 2>$null
    $remoteCommit = & git -C $repo rev-parse origin/main 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $localCommit -or -not $remoteCommit) { return }
    if ($localCommit -eq $remoteCommit) { return }

    $changes = & git -C $repo status --porcelain --untracked-files=normal 2>$null
    if ($LASTEXITCODE -ne 0 -or $changes) {
        Write-Host 'CV-1 sync: local edits exist; keeping them and waiting to pull.'
        return
    }

    & git -C $repo merge-base --is-ancestor $localCommit $remoteCommit 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CV-1 sync: branches diverged; manual Git review needed.'
        return
    }

    $mergeOutput = & git -C $repo merge --ff-only --quiet origin/main 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "CV-1 sync: updated source to $($remoteCommit.Substring(0, 7))."
    } else {
        Write-Host "CV-1 sync: merge failed; manual Git review needed. $mergeOutput"
    }
}

Write-Host 'CV-1 sync started'
do {
    Sync-FromGitHub
    if ($Once) { break }
    Start-Sleep -Seconds ([Math]::Max(10, $IntervalSeconds))
} while ($true)
