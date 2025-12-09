"""
Prepare sequences for next-app model training.

Reads `events.jsonl` produced by the exporter and writes `sequences.jsonl`,
one JSON array of package names per line (sessionized by inactivity gap).

Usage:
  python ml/preprocessing/prepare_sequences.py --in ml/data/events.jsonl --out ml/data/sequences.jsonl
"""
import argparse
import json
from datetime import datetime


def prepare(in_path, out_path, gap_seconds=300):
    last_ts = None
    current = []
    written = 0
    with open(in_path, 'r', encoding='utf-8') as inf, open(out_path, 'w', encoding='utf-8') as outf:
        for line in inf:
            obj = json.loads(line)
            ts = obj.get('timestamp')
            pkg = obj.get('packageName') or obj.get('package') or 'unknown'
            if ts is None:
                continue
            if last_ts is None:
                current = [pkg]
            else:
                if (ts - last_ts) / 1000.0 > gap_seconds:
                    if len(current) > 1:
                        outf.write(json.dumps(current) + '\n')
                        written += 1
                    current = [pkg]
                else:
                    current.append(pkg)
            last_ts = ts
        if current and len(current) > 1:
            outf.write(json.dumps(current) + '\n')
            written += 1
    print(f"Wrote {written} sequences to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', required=True)
    p.add_argument('--out', dest='out_path', required=True)
    p.add_argument('--gap', dest='gap_seconds', type=int, default=300)
    args = p.parse_args()
    prepare(args.in_path, args.out_path, args.gap_seconds)
