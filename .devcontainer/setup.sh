#!/bin/bash
# PLE eval environment check for GitHub Codespaces (docker-in-docker feature
# provides the daemon; the PLE kit's own `uv sync` pulls Harbor and the rest).
set -e
echo "== waiting for docker daemon =="
for i in $(seq 1 30); do
  if docker version >/dev/null 2>&1; then break; fi
  sleep 2
done
docker version --format 'docker server: {{.Server.Version}}'
echo "== installing uv =="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
echo "== machine =="
nproc; free -h | head -2; df -h / | tail -1
echo "SETUP OK — clone google-research/ple here and run its eval/kit checks."
