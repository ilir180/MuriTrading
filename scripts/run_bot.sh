#!/bin/bash
cd /Users/ilirmurati/MuriTrading

# Threading fix (PyTorch + sklearn)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Direkt pyenv Python 3.11.9 verwenden (gleiche Version wie Retrain)
exec /Users/ilirmurati/.pyenv/versions/3.11.9/bin/python src/bot/paper_trader.py
