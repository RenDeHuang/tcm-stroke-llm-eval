#!/usr/bin/env python3
"""Create a non-destructive, prompt-separated view of all run artifacts."""
from pathlib import Path
import json, re

ROOT = Path('/Users/huangrende/Desktop/王健')
RUNS = ROOT / 'out/tcm_stroke_eval/runs'
ORG = ROOT / 'out/tcm_stroke_eval/organized'
for p in [ORG / f'P{i}' for i in range(8)]:
    p.mkdir(parents=True, exist_ok=True)
(ORG / 'metadata').mkdir(exist_ok=True)

manifest = []
for src in sorted(RUNS.glob('*.jsonl')):
    m = re.search(r'__T1_(P[0-7])_(?:zh|en)__', src.name)
    if not m:
        continue
    prompt = m.group(1)
    dst = ORG / prompt / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    manifest.append({'prompt': prompt, 'file': str(src), 'view': str(dst)})

(ORG / 'metadata' / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(f'organized {len(manifest)} run files under {ORG}')
