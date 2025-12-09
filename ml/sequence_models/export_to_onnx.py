"""
Export a trained PyTorch next-app model checkpoint to ONNX.

This helper is optional and will silently exit if PyTorch isn't installed.

Usage:
  python ml/sequence_models/export_to_onnx.py --ckpt ml/models/next_app_model.pt --out ml/models/next_app_model.onnx
"""
import argparse
import os

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        h = h[-1]
        return self.out(h)


def export(ckpt_path, out_path, seq_len=20):
    if torch is None:
        print("PyTorch not available; skipping ONNX export.")
        return False

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    data = torch.load(ckpt_path, map_location='cpu')
    vocab = data.get('vocab') or {}
    vocab_size = len(vocab)
    model = SimpleLSTM(vocab_size)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    # Dummy input (batch_size=1, seq_len-1 as training used input_ids[:-1])
    dummy = torch.zeros((1, seq_len-1), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=11,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch'}}
    )
    print(f"Exported ONNX model -> {out_path}")
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seq_len', type=int, default=20)
    args = p.parse_args()
    export(args.ckpt, args.out, args.seq_len)
