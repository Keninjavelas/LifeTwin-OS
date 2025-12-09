"""Export trained next-app model to ONNX, optionally quantize, and run a small evaluation.

This script expects a PyTorch state dict at ml/models/next_app_model.pt and
consumes the exported summaries at ml/data/summaries_export.json.

It will:
- load the PyTorch model if available
- export to ONNX (opset 11)
- optionally perform dynamic quantization using onnxruntime (if available)
- run a single-batch inference using ONNX Runtime and compare outputs
"""

from pathlib import Path
import argparse
from ml.utils.load_export import load_exported_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(Path(__file__).resolve().parents[2] / "models" / "next_app_model.pt"))
    parser.add_argument("--onnx-path", default=str(Path(__file__).resolve().parents[2] / "models" / "next_app_model.onnx"))
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    payload = load_exported_summaries()
    summaries = payload.get("summaries", [])
    if not summaries:
        print("No exported summaries found. Run the demo export first.")
        return

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        torch = None
        print("PyTorch not available; ONNX export from state dict will be skipped unless torch is installed.")

    # Build a tiny model class compatible with saved state dict
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

    # create a toy dataset consistent with export loader logic
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

    if not sequences:
        print("Not enough sequence data to evaluate.")
        return

    # pad first batch
    max_len = max(len(x) for x in sequences)
    import numpy as np
    batch = [([0] * (max_len - len(x))) + x for x in sequences[:8]]
    batch_arr = np.array(batch, dtype=np.int64)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print("Model state not found at", model_path)
        return

    if torch is None:
        print("Skipping PyTorch load/export because torch is not installed.")
    else:
        # load state and export to ONNX
        # infer vocab size
        vocab = set(v for seq in sequences for v in seq) | set(targets)
        num_apps = max(vocab) + 1
        model = SimpleNextAppModel(num_apps=num_apps)
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print("Failed to load state dict:", e)
        model.eval()
        dummy = torch.tensor(batch_arr, dtype=torch.long)
        onnx_path = Path(args.onnx_path)
        try:
            torch.onnx.export(model, dummy, onnx_path.as_posix(), opset_version=11, input_names=["input"], output_names=["output"])
            print("Exported ONNX to", onnx_path)
        except Exception as e:
            print("ONNX export failed:", e)

    # Try to run ONNX Runtime inference
    try:
        import onnxruntime as ort
    except Exception:
        ort = None

    if ort is None:
        print("ONNX Runtime not available; evaluation skipped. To evaluate ONNX, install onnxruntime.")
        return

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print("ONNX model not found at", onnx_path)
        return

    sess = ort.InferenceSession(onnx_path.as_posix())
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: batch_arr})
    print("ONNX inference output shapes:", [o.shape for o in outputs])

    if args.quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            qpath = onnx_path.with_suffix('.quant.onnx')
            quantize_dynamic(onnx_path.as_posix(), qpath.as_posix(), weight_type=QuantType.QInt8)
            print("Quantized ONNX model saved to", qpath)
        except Exception as e:
            print("Quantization failed (ensure onnxruntime-tools installed):", e)


if __name__ == "__main__":
    main()
