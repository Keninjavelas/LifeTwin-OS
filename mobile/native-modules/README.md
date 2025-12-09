# Mobile Native Modules (Android)

This folder contains Android native-module stubs and Room DB code used by the LifeTwin RN app during development.

## Gradle Dependencies

Add to your app module `build.gradle`:

```gradle
// Room
implementation "androidx.room:room-runtime:2.5.2"
implementation "androidx.room:room-ktx:2.5.2"
kapt "androidx.room:room-compiler:2.5.2"

// OkHttp + Gson for network and JSON
implementation "com.squareup.okhttp3:okhttp:4.11.0"
implementation "com.google.code.gson:gson:2.10.1"

// Coroutines
implementation "org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3"
implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3"

// WorkManager
implementation "androidx.work:work-runtime-ktx:2.8.1"

// ONNX Runtime Mobile (optional) - for on-device ONNX inference
// implementation "com.microsoft.onnxruntime:onnxruntime-android:1.15.1"
// Check https://github.com/microsoft/onnxruntime/releases for latest version
```

Enable KAPT in module-level build.gradle:

```gradle
apply plugin: 'kotlin-kapt'
```

## Native Modules

### NativeModelLoader (inference/)

Reflective ONNX session loader that detects ONNX Runtime at runtime without compile-time dependency:

- `loadModel(modelPath)` — detects ONNX Runtime classes, creates session, stores it.
- `unloadSession()` — releases session and environment.
- `runInference(inputName, inputData, outputName)` — runs inference with best-effort reflection.
- `copyDebugOnnxAsset(assetFileName)` — copies ONNX file from assets to internal storage for testing.

Logs detailed status (classes detected, session created, inference success/failure).

### KeystoreModule (security/)

Envelope encryption helpers using AndroidKeyStore:

- `generateWrappedDataKey()` — creates AES data key, wraps it with RSA key from Keystore, returns wrapped key + IV.
- `unwrapDataKey(wrappedKey, iv)` — unwraps data key using private key from Keystore.

JS wrapper: `mobile/app/src/services/e2eeService.ts` with Jest test.

### AutomationModule (automation/)

Rule-based automation manager:

- `evaluateRules(summaryJson)` — evaluates automation rules and returns triggered actions.
- `enableDND()` — enables Do Not Disturb mode (requires permissions).
- `suggestBreak(message)` — sends notification suggesting a break.

JS wrapper: `mobile/app/src/services/automationService.ts` with Jest test.

## Android Studio Build & Run

1. Open the Android project in Android Studio.
2. Ensure `kapt` is enabled in your `build.gradle`.
3. Sync Gradle to download dependencies.
4. Build the app (Build → Make Project).
5. Schedule `DailySummaryWorker` periodically using WorkManager APIs.

## Quick adb Commands (Emulator)

```bash
# Install and run debug build
./gradlew :app:installDebug

# Monitor logcat
adb logcat -s AppDatabase SyncManager DBHelper NativeModelLoader KeystoreModule
```

## Notes

- `SyncManager.endpoint` defaults to `http://10.0.2.2:8000/...` for emulator testing.
- Room migration 1→2 creates the `sync_queue` table (no destructive fallback).
- NativeModelLoader uses reflection to avoid compile-time ONNX dependency; add the AAR to enable full inference.

## React Native Usage Example

```js
import { NativeModules } from 'react-native'
const { AutomationModule, NativeModelLoader, KeystoreModule } = NativeModules

// Automation
AutomationModule.setServerUrl('http://192.168.1.10:8000')
AutomationModule.triggerDailySummaryExport('demo-token', 'demo-device')
  .then(resultJson => console.log('Export result:', resultJson))
  .catch(err => console.error('Export failed', err))

// ONNX Inference
NativeModelLoader.loadModel('/data/user/0/com.app/files/model.onnx')
  .then(status => console.log('Model loaded:', status))
NativeModelLoader.runInference('input', [1.0, 2.0], 'output')
  .then(result => console.log('Inference result:', result))

// Keystore E2EE
KeystoreModule.generateWrappedDataKey()
  .then(result => console.log('Wrapped key:', result))
```
