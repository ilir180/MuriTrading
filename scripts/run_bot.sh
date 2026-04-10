#!/bin/bash
cd /Users/ilirmurati/MuriTrading

# Use same Python as retrain pipeline (pyenv 3.11.9)
export PATH="$HOME/.pyenv/versions/3.11.9/bin:$HOME/.pyenv/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
pyenv shell 3.11.9 2>/dev/null || true

# Threading fix (PyTorch + sklearn)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec python src/bot/paper_trader.py
