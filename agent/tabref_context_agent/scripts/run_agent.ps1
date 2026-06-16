param(
    [ValidateSet("prepare", "codex", "existing", "manual")]
    [string]$Mode = "prepare",
    [string]$ThreadId = ""
)

$AgentRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $AgentRoot ".venv\Scripts\python.exe"
$Config = Join-Path $AgentRoot "config.local.json"

if (-not (Test-Path -LiteralPath $Config)) {
    throw "Missing $Config. Copy config.example.json to config.local.json first."
}

$Arguments = @(
    "-m", "tabref_agent.cli",
    "run",
    "--config", $Config,
    "--mode", $Mode
)
if ($ThreadId) {
    $Arguments += @("--thread-id", $ThreadId)
}

& $Python @Arguments

