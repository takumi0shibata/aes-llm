#!/bin/bash
set -e

SCRIPT=few_shot.py  # 実際のスクリプト名に合わせて変更してください

uv run "$SCRIPT" \
    --model "google/gemma-4-e4b-it" \
    --model-family "gemma"

uv run "$SCRIPT" \
    --model "Qwen/Qwen3.5-9B" \
    --model-family "qwen"
