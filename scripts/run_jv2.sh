#!/bin/bash
# JV Boting v2 Runner

export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$HOME/MuriTrading"
exec /Users/ilirmurati/.pyenv/versions/3.11.9/bin/python src/jv2/runner.py
