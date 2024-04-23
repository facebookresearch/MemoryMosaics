#!/bin/sh

VENV=${1-$HOME/venv/mlx}

python3 -m venv $VENV

source $VENV/bin/activate

pip install --upgrade pip
pip install mlx==0.9 tqdm numpy matplotlib

