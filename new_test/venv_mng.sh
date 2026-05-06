#!/usr/bin/bash

if [ ! -d ".venv" ]; then
  uv venv .venv
fi

source .venv/bin/activate

uv pip install -r requirements.txt

uv pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu126
