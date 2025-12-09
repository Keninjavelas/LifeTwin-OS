"""Improved training script for the sequence behavior model (next-app predictor).

Features:
- Loads exported summaries from ml/data/summaries_export.json
- Builds simple integer tokenization for apps
- Creates train/test split, batching, computes accuracy
- Optional ONNX export when available
"""

from pathlib import Path
from typing import List, Tuple
from ml.utils.load_export import load_exported_summaries
import argparse


def build_dataset(summaries) -> Tuple[List[List[int]], List[int], dict]:
    app_to_id = {}
    sequences = []
    targets = []
    for s in summaries:
        apps = s.get("top_apps") or []
        seq = []
        for a in apps:
            if a not in app_to_id:
                app_to_id[a] = len(app_to_id) + 1
            seq.append(app_to_id[a])
        if len(seq) >= 2:
            sequences.append(seq[:-1])
            targets.append(seq[-1])
    return sequences, targets, app_to_id


def pad_tensor_list(X: List[List[int]]):
    import torch
    max_len = max((len(x) for x in X), default=0)
    padded = [([0] * (max_len - len(x))) + x for x in X]
    return torch.tensor(padded, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    payload = load_exported_summaries()
    summaries = payload.get("summaries", [])
    if not summaries:
        print("No exported summaries found. Run backend/fastapi/scripts/demo_export.py first.")
        return

    X_seqs, y, app_to_id = build_dataset(summaries)
    if not X_seqs:
        print("Not enough sequence length in data to train.")
        return

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except Exception as e:
        print("PyTorch is not available. To train, install torch. Error:", e)
        return

    # construct vocab size
    vocab = set(v for seq in X_seqs for v in seq) | set(y)
    num_apps = max(vocab) + 1

    class SimpleNextAppModel(nn.Module):
        def __init__(self, num_apps: int, embed_dim: int = 32, hidden_dim: int = 64):
            super().__init__()
            self.embed = nn.Embedding(num_apps, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_apps)

        def forward(self, x):
            emb = self.embed(x)
            out, _ = self.lstm(emb)
            logits = self.fc(out[:, -1, :])
            return logits

    X_t = pad_tensor_list(X_seqs)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    # simple shuffling and split
    n = len(dataset)
    split = max(1, int(n * 0.8))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [split, n - split])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SimpleNextAppModel(num_apps=num_apps)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / max(1, len(train_loader.dataset))

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} test_acc={acc:.3f}")

    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "next_app_model.pt")
    print("Saved model to", model_dir / "next_app_model.pt")

    # write vocab mapping for deployment and device push
    try:
        from ml.utils.save_artifact import save_vocab

        save_vocab(model_dir, app_to_id, filename='vocab.json')
        print('Saved vocab to', model_dir / 'vocab.json')
    except Exception:
        pass

    if args.export_onnx:
        try:
            dummy = torch.zeros(1, X_t.size(1), dtype=torch.long)
            onnx_path = model_dir / "next_app_model.onnx"
            torch.onnx.export(model, dummy, onnx_path.as_posix(), opset_version=11)
            print("Exported ONNX to", onnx_path)
        except Exception as e:
            print("ONNX export failed:", e)

    # Write final evaluation metrics (test accuracy). If no test set, write placeholders.
    try:
        from ml.utils.save_artifact import save_metrics

        metrics = {"test_accuracy": float(acc) if 'acc' in locals() else 0.0}
        save_metrics(model_dir, 'next_app_model', metrics)
        print('Saved metrics to', model_dir / 'next_app_model.metrics.json')
    except Exception:
        pass

    # save vocab mapping for deployment and device push


if __name__ == "__main__":
    main()
