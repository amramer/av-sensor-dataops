"""Shrink training settings for CI (CPU runner, synthetic data). Edits params.yaml in place.

Only the values are changed (line by line), so comments and layout stay intact.
Never commit the result: CI runs on a throwaway checkout.
"""

import re
from pathlib import Path

OVERRIDES = {
    "train": {"epochs": "3", "imgsz": "320", "batch": "8", "device": "cpu", "workers": "0"},
    "gate": {"min_map50": "0.0"},  # synthetic data: the gate logic runs, the bar is not meaningful
}

text = Path("params.yaml").read_text()
for section, values in OVERRIDES.items():
    block = re.search(rf"^{section}:\n((?:[ \t]+.*\n?)+)", text, re.M)
    if not block:
        raise SystemExit(f"section {section} not found")
    body = block.group(1)
    for key, value in values.items():
        body, n = re.subn(
            rf"^(\s+{key}:\s*)([^#\n]*?)(\s*#.*)?$",
            rf"\g<1>{value}\g<3>",
            body,
            count=1,
            flags=re.M,
        )
        if n != 1:
            raise SystemExit(f"{section}.{key} not found")
    text = text[: block.start(1)] + body + text[block.end(1) :]
Path("params.yaml").write_text(text)
print("CI params applied:", OVERRIDES)
