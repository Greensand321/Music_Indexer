# Delete stale remote branches on Greensand321/Music_Indexer.
# git-only - no gh CLI or extra auth needed, just your existing git credentials.
#
# Keeps: main, whatever branch you have checked out, and any branch starting
# with "claude/". Deletes everything else, regardless of open PR status.
#
# Usage:
#   .\scripts\cleanup_branches.ps1            # dry run - prints what WOULD be deleted
#   .\scripts\cleanup_branches.ps1 -Execute   # actually deletes (asks you to type DELETE)

param(
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$BatchSize = 30

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

$ToDelete = $AllBranches | Where-Object {
    $_ -ne $DefaultBranch -and $_ -ne $CurrentBranch -and $_ -notlike "claude/*"
}

$Kept = $AllBranches | Where-Object { $ToDelete -notcontains $_ }

Write-Host ""
Write-Host "Total branches:  $($AllBranches.Count)"
Write-Host "Keeping:         $($Kept.Count)  ($($Kept -join ', '))"
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
