#!/usr/bin/env bash
set -euo pipefail

echo "Setting up mloptim development environment..."
rustup update stable
cargo build
cargo test
echo "Done."
