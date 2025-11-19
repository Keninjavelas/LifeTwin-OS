# LifeTwin OS Mobile App (MLP)

React Native + TypeScript mobile client for the LifeTwin OS Minimum Lovable Product.

### Included in MLP

- Home dashboard with today summary and next-app prediction (Markov model).
- App usage summary screen.
- Insights screen with simple textual insights.
- Settings screen with a toggle for summary sync.
- Permissions onboarding screen.
- Zustand store holding events and daily summary.
- TS-native bridge (`src/services/nativeCollectors.ts`) with a mock implementation.

### Next steps

- Implement real Kotlin native modules for UsageStats, notifications, and screen events.
- Replace Markov predictor with an exported ONNX/TFLite model.
