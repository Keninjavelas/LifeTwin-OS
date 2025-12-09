# LifeTwin OS Mobile App (MLP)

React Native + TypeScript mobile client for the LifeTwin OS Minimum Lovable Product.

## Included in MLP

- Home dashboard with today summary and next-app prediction (Markov model).
- App usage summary screen.
- Insights screen with simple textual insights.
- Settings screen with a toggle for summary sync.
- Permissions onboarding screen.
- Zustand store holding events and daily summary.
- TS-native bridge (`src/services/nativeCollectors.ts`) with a mock implementation.

## Debug Screens (Development)

- **ModelStatusScreen**: displays ONNX model load status, session info, and test inference results.
- **AutomationDebugScreen**: shows automation rule evaluation, DND status, and trigger logs.
- **KeystoreDebugScreen**: tests Keystore key generation, data key wrapping/unwrapping, and E2EE helpers.

## JS Service Wrappers

- `nativeInference.ts` — wrapper for NativeModelLoader (load/unload/run ONNX inference).
- `automationService.ts` — wrapper for AutomationModule (evaluate rules, enable DND, suggest breaks).
- `keystoreService.ts` — wrapper for KeystoreModule (generate/unwrap data keys).
- `e2eeService.ts` — end-to-end encryption helpers using Keystore-wrapped keys.

## Tests

Jest smoke tests for all native wrappers:

```bash
npm test
```

Tests validate module availability and basic method signatures.

## Next Steps

- Implement real Kotlin native modules for UsageStats, notifications, and screen events.
- Replace Markov predictor with an exported ONNX/TFLite model.
- Wire automation rules to Android APIs (DND, notifications, app blocking).
