#!/usr/bin/env bash
set -euo pipefail

# Usage: push_model_to_device.sh <app_package> <model.onnx> <vocab.json>
# Example: ./push_model_to_device.sh com.lifetwin.mlp ml/models/next_app_model.onnx ml/models/vocab.json

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <app_package> <model.onnx> <vocab.json>"
  exit 2
fi

PKG="$1"
MODEL_PATH="$2"
VOCAB_PATH="$3"

MODEL_BASENAME=$(basename "$MODEL_PATH")
VOCAB_BASENAME=$(basename "$VOCAB_PATH")

echo "Pushing $MODEL_BASENAME and $VOCAB_BASENAME to device for package $PKG"

adb push "$MODEL_PATH" /data/local/tmp/
adb push "$VOCAB_PATH" /data/local/tmp/

echo "Attempting to move files into app filesDir using run-as"
if adb shell run-as "$PKG" true 2>/dev/null; then
  adb shell "run-as $PKG cp /data/local/tmp/$MODEL_BASENAME files/ || true"
  adb shell "run-as $PKG cp /data/local/tmp/$VOCAB_BASENAME files/ || true"
  echo "Files copied to app filesDir via run-as"
else
  echo "run-as not available or failed. Trying to copy to /data/data/$PKG/files with root (requires rooted device)."
  adb shell "su -c 'mkdir -p /data/data/$PKG/files && cp /data/local/tmp/$MODEL_BASENAME /data/data/$PKG/files/ || true'" || true
  adb shell "su -c 'cp /data/local/tmp/$VOCAB_BASENAME /data/data/$PKG/files/ || true'" || true
  echo "Tried root copy; if this failed you may need to run as debuggable app or use `run-as` on emulator." 
fi

echo "Cleanup: removing from /data/local/tmp"
adb shell rm /data/local/tmp/$MODEL_BASENAME || true
adb shell rm /data/local/tmp/$VOCAB_BASENAME || true

echo "Done. Restart the app to let it pick up the new model files."
