#!/usr/bin/env python3
"""Thin wrapper around ``training.train`` so the script and module agree.

    python scripts/train_embedding.py --config configs/lora.yaml \
        --train-file /path/to/data.jsonl

    # no GPU / no 8B weights required:
    python scripts/train_embedding.py --smoke-test
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
