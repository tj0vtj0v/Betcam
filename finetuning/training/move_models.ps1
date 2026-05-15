$trainingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $trainingDir)
$runsRoot = Join-Path $trainingDir "runs"

if (-not (Test-Path -LiteralPath $runsRoot)) {
    throw "Runs directory not found: $runsRoot"
}

Get-ChildItem -LiteralPath $runsRoot -Directory | ForEach-Object {
    $runName = $_.Name
    $sourcePath = Join-Path $_.FullName "weights\best.pt"

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        Write-Output "Skipping $runName because best.pt was not found"
        return
    }

    $destinationPath = Join-Path $repoRoot "$runName.pt"
    Move-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    Write-Output "Moved $sourcePath to $destinationPath"
}
