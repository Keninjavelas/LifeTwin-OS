# Keystore Native Module (Android)

This module exposes a minimal Android Keystore helper to generate a key pair from the React Native layer.

Usage (JS):

```ts
import KeystoreService from '../services/Keystore'

await KeystoreService.generateKeyPair('mlp-demo-key')
```

Notes and limitations:
- This module uses the AndroidKeyStore and is only meaningful on Android devices/emulators.
- On older Android versions some KeyGenParameterSpec options may not be available. Use API level checks if you need fine-grained compatibility.
- This library is a scaffold for E2EE demos; in production you should implement key validity checks, secure backup, and proper key usage policies.

Testing locally:
- The JS wrapper has a Jest smoke test at `mobile/app/__tests__/Keystore.test.js` which mocks `NativeModules.KeystoreModule`.
- To validate on-device, build the Android app and run the `KeystoreDebug` screen, then press `Generate Key Pair`.
