param(
    [switch]$DryRun,
    [switch]$Uninstall,
    [string]$Binary = ""
)
$ErrorActionPreference = "Stop"
$menuRoot = "HKCU:\Software\Classes\Directory\shell\EpistemosInstantRecall"
if ($Uninstall) {
    if ($DryRun) { Write-Host "Would remove $menuRoot"; exit 0 }
    Remove-Item -LiteralPath $menuRoot -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Removed the per-user Explorer menu. Indexed data was left intact."
    exit 0
}
if (-not $Binary) {
    $Binary = (Resolve-Path "$PSScriptRoot\..\..\target\release\recall.exe" -ErrorAction Stop).Path
}
$command = '"' + $Binary + '" index "%V"'
if ($DryRun) {
    Write-Host "Would add Explorer > Epistemos: index this folder"
    Write-Host "Command: $command"
    exit 0
}
New-Item -Path $menuRoot -Force | Out-Null
Set-Item -Path $menuRoot -Value "Epistemos: index this folder"
New-Item -Path "$menuRoot\command" -Force | Out-Null
Set-Item -Path "$menuRoot\command" -Value $command
Write-Host "Installed a per-user Explorer folder action. No administrator access was used."
