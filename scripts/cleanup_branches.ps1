# Delete stale remote branches on Greensand321/Music_Indexer.
# git-only version - no `gh` CLI or extra auth needed, just your existing
# git credentials (the same ones you already push/pull with).
#
# Keeps: main, whatever branch you have checked out, and the branches below,
# which were tied to open PRs as of 2026-08-30. Open PRs can change between
# then and whenever you run this - eyeball
# https://github.com/Greensand321/Music_Indexer/pulls before running -Execute,
# and add any branch you want protected to $KeepExtra.
#
# Usage:
#   .\scripts\cleanup_branches.ps1            # dry run - prints what WOULD be deleted
#   .\scripts\cleanup_branches.ps1 -Execute   # actually deletes (asks you to type DELETE)

param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$BatchSize = 30

$KeepExtra = @(
    "codex/add-progress-bar-to-compression-tab",
    "codex/evaluate-implementation-timeline-for-pyqt/pyside",
    "codex/audit-duplicate-finder-preview-report",
    "revert-667-codex/enhance-track-gathering-with-post-write-refresh",
    "revert-563-codex/improve-file-cleaning-process-for-indexer",
    "codex/add-presets-to-threshold-button",
    "codex/fix-issues-with-requirements.txt",
    "codex/review-code-and-documentation-status",
    "codex/fix-hbdscan-button-layout-issue",
    "m1kpry-codex/implement-crashwatcher-with-event-recording",
    "cbxcy1-codex/fix-crash-when-playing-songs-simultaneously"
)

Write-Host "Fetching branches ..."
git fetch origin --prune --quiet

$DefaultBranch = (git remote show origin | Select-String "HEAD branch").ToString().Split(":")[1].Trim()
$CurrentBranch = git branch --show-current

Write-Host "Default branch: $DefaultBranch"
Write-Host "Current branch: $CurrentBranch"

$AllBranches = git for-each-ref --format="%(refname:short)" refs/remotes/origin |
    ForEach-Object { $_ -replace '^origin/', '' } |
    Where-Object { $_ -ne "HEAD" } |
    Sort-Object -Unique

$Keep = @($DefaultBranch, $CurrentBranch) + $KeepExtra | Sort-Object -Unique

$ToDelete = $AllBranches | Where-Object { $Keep -notcontains $_ }

Write-Host ""
Write-Host "Total branches:  $($AllBranches.Count)"
Write-Host "Keeping:         $($Keep.Count)"
Write-Host "To delete:       $($ToDelete.Count)"
Write-Host ""

if ($ToDelete.Count -eq 0) {
    Write-Host "Nothing to delete."
    exit 0
}

Write-Host "Branches slated for deletion (first 30 shown):"
$ToDelete | Select-Object -First 30 | ForEach-Object { Write-Host "  $_" }
if ($ToDelete.Count -gt 30) {
    Write-Host "  ... and $($ToDelete.Count - 30) more"
}
Write-Host ""

if (-not $Execute) {
    Write-Host "Dry run only. Re-run with -Execute to actually delete these branches."
    exit 0
}

$Confirm = Read-Host "Type DELETE to permanently delete these $($ToDelete.Count) branches"
if ($Confirm -ne "DELETE") {
    Write-Host "Aborted."
    exit 1
}

$deleted = 0
for ($i = 0; $i -lt $ToDelete.Count; $i += $BatchSize) {
    $batch = $ToDelete[$i..[Math]::Min($i + $BatchSize - 1, $ToDelete.Count - 1)]
    $refspecs = $batch | ForEach-Object { ":$_" }
    git push origin @refspecs
    if ($LASTEXITCODE -eq 0) {
        $deleted += $batch.Count
    } else {
        Write-Warning "Batch failed, some branches may not be deleted: $($batch -join ', ')"
    }
}

Write-Host ""
Write-Host "Done. Deleted approximately $deleted branches."
