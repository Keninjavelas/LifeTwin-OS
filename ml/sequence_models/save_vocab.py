"""
Save `vocab.json` from a PyTorch checkpoint that contains a `vocab` mapping.

Usage:
  python ml/sequence_models/save_vocab.py --ckpt ml/models/next_app_model.pt --out ml/models/vocab.json

This script is optional and will exit if PyTorch isn't available.
"""
import argparse
import json
import os

try:
    import torch
except Exception:
    torch = None


def save_vocab(ckpt_path, out_path):
    if torch is None:
        print("PyTorch not installed; cannot read checkpoint to extract vocab.")
        return False
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    data = torch.load(ckpt_path, map_location='cpu')
    vocab = data.get('vocab')
    if not vocab:
        print('No vocab found in checkpoint')
        return False
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f'Wrote vocab -> {out_path}')
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    save_vocab(args.ckpt, args.out)
