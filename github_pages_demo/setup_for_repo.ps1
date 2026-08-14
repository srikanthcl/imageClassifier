$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$templateWorkflow = Join-Path $PSScriptRoot "workflow-template\deploy-pages.yml"
$workflowDir = Join-Path $projectRoot ".github\workflows"
$targetWorkflow = Join-Path $workflowDir "deploy-pages.yml"

if (-not (Test-Path $templateWorkflow)) {
    throw "Template workflow not found: $templateWorkflow"
}

if (-not (Test-Path $workflowDir)) {
    New-Item -Path $workflowDir -ItemType Directory | Out-Null
}

Copy-Item -Path $templateWorkflow -Destination $targetWorkflow -Force

Write-Host "GitHub Pages workflow installed at: $targetWorkflow"
Write-Host "Next steps:"
Write-Host "1) git add .github/workflows/deploy-pages.yml"
Write-Host "2) git commit -m 'Add GitHub Pages deployment workflow'"
Write-Host "3) git push"
Write-Host "4) On GitHub -> Settings -> Pages -> Source: GitHub Actions"
