#!/usr/bin/env bash
# Run a ber module inside WSL with the project venv. Usage: bash ber.sh -m ber.run --stage prep
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BER_VENV:-$HOME/ber/.venv}/bin/activate"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
# LightGBM needs libgomp.so.1; without root (no apt), reuse the copy bundled with torch.
if ! ldconfig -p 2>/dev/null | grep -q libgomp.so.1; then
  export LD_LIBRARY_PATH="$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$HERE"
exec python "$@"
