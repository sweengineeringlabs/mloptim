#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Setting up mloptim development environment..."
rustup update stable
cargo build
cargo test
Write-Host "Done."
