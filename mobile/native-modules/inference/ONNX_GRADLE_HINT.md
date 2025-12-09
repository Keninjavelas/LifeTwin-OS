ONNX Runtime (Android) Gradle integration hint

This file contains a non-invasive Gradle snippet you can add to the Android `app/build.gradle` to include
ONNX Runtime Mobile. We intentionally don't add this dependency to the repo so CI and local builds don't
start pulling native AARs unless you opt-in.

Add to `dependencies` in `android/app/build.gradle` (example):

```gradle
// Optional: ONNX Runtime mobile AAR (choose matching version for your project)
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.0'
```

Notes:
- ONNX Runtime requires native libraries and increases APK size. Use the mobile flavor/ABI filtering
  to reduce binary size.
- If you add the dependency, you can remove the reflective fallback in `NativeModelLoader.kt` and
  call OrtEnvironment/OrtSession APIs directly.
