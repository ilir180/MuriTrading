#!/bin/bash
# Joint Venture Boting – Runner Script
# Für launchd: com.muritrading.jv.plist

export PATH="$HOME/.pyenv/versions/3.11.9/bin:$HOME/.pyenv/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
pyenv shell 3.11.9 2>/dev/null || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$HOME/MuriTrading"
exec python src/jv/runner.py
