# Copies the analysis outputs from a local run into this repository.
#
#   .\tools\collect_results.ps1 -RunRoot "D:\XAI_paper_Review\Code_runs\xai_audit_v3\results"
#
# Per-molecule prediction CSVs are deliberately skipped: they are large and
# reproducible from stage 2.

param(
    [Parameter(Mandatory = $true)][string]$RunRoot
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $RunRoot)) { throw "not found: $RunRoot" }

$sets = @(
    @{ From = "table_a";             Files = @("table_a_groups.csv", "table_a_rule_partitions.csv") },
    @{ From = "corpus_profile";      Files = @("corpus_motif_profile.csv", "corpus_property_summary.csv", "corpus_vs_benchmark_motifs.csv", "corpus_rule_compliance.csv", "settings.json") },
    @{ From = "table_b";             Files = @("*.csv") },
    @{ From = "table_b_attribution"; Files = @("attribution_summary.csv", "motif_importance.csv", "settings.json") },
    @{ From = "generator_gates";     Files = @("generator_gate_profile.csv") },
    @{ From = "leakage";             Files = @("leakage_summary.csv", "leakage_similarity_distribution.csv", "settings.json") },
    @{ From = "predictions";         Files = @("performance_gate.csv", "manifest.json") }
)

foreach ($set in $sets) {
    $src = Join-Path $RunRoot $set.From
    $dst = Join-Path $repo "results\$($set.From)"
    if (-not (Test-Path $src)) { Write-Warning "skipping, not found: $src"; continue }
    New-Item -ItemType Directory -Force $dst | Out-Null
    foreach ($pattern in $set.Files) {
        Get-ChildItem -Path (Join-Path $src $pattern) -ErrorAction SilentlyContinue |
            Copy-Item -Destination $dst -Force
    }
    Write-Host "copied $($set.From)"
}

# Figure 9: held-out split only. Change figures_v3 if your output directory differs.
$figSrc = Join-Path $RunRoot "figures_v3"
if (Test-Path $figSrc) {
    $figDst = Join-Path $repo "results\figures"
    New-Item -ItemType Directory -Force $figDst | Out-Null
    Copy-Item -Path (Join-Path $figSrc "*") -Destination $figDst -Force
    Write-Host "copied figures"
} else {
    Write-Warning "skipping figures, not found: $figSrc"
}

Write-Host "`nDone. Review 'git status' before committing."
