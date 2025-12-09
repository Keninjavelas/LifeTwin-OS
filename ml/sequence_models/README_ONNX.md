ONNX export helper
==================

This helper converts the PyTorch checkpoint produced by `train_next_app_model.py` into an ONNX file.

Usage:

```bash
# after training
python ml/sequence_models/export_to_onnx.py --ckpt ml/models/next_app_model.pt --out ml/models/next_app_model.onnx
```

Notes:
- The helper will skip if PyTorch is not installed.
- The exported ONNX model assumes integer input token IDs with shape `(batch, seq_len)`.
- You still need a lightweight runtime on-device (ONNX Runtime Mobile or similar) to run the model.
