Pushing model and vocab to Android emulator/device
=================================================

This folder includes `push_model_to_device.sh`, a small helper to push an ONNX
model file and `vocab.json` into the app's `filesDir` for local testing on an
emulator or device.

Usage example:

```bash
# from project root
./mobile/native-modules/inference/push_model_to_device.sh com.your.app ml/models/next_app_model.onnx ml/models/vocab.json
```

Notes:
- The script uses `adb run-as <package>` which works when the app is debuggable
  (typical for emulator/dev builds). If `run-as` isn't allowed, the script will
  try `su` copy which requires a rooted device and will usually fail on
  non-rooted physical devices.
- If you can't use `run-as`, you can manually copy files via Android Studio or
  include the files in your APK assets for testing.
- After copying, restart the app so `NativeModelLoader` can detect `next_app_model.onnx` and `vocab.json` in `filesDir`.

Format note:
- `vocab.json` should be a JSON object mapping app package or identifier strings to integer ids, for example:

```json
{
  "com.facebook.katana": 1,
  "com.instagram.android": 2,
  "com.whatsapp": 3
}
```

Native code prefers this mapping form (`app -> id`) when present.

Testing in the app
------------------
- Open the `NextAppPredictionExample` component (added under `mobile/app/src/components`) in the app UI and use the "Predict next app" button. The RN bridge calls `NativeCollectors.predictNextApp(history)` which will use the device files if present.
