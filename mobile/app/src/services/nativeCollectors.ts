// TS bridge for native collectors. In the MLP we mock this layer;
// later, Kotlin modules will implement real data collection.

import { NativeModules, Platform } from "react-native";
import { AppEvent, DailySummary } from "@models/types";

interface NativeCollectorsModule {
  requestAllPermissions(): Promise<boolean>;
  subscribeToEvents(callback: (event: AppEvent) => void): void;
  getTodaySummary(): Promise<DailySummary | null>;
  syncDailySummary(summary: DailySummary): Promise<void>;
  predictNextApp(history: string[]): Promise<string | null>;
}

const MockModule: NativeCollectorsModule = {
  async requestAllPermissions() {
    return true;
  },
  subscribeToEvents() {
    // no-op for MLP mock
  },
  async getTodaySummary() {
    const today = new Date().toISOString().slice(0, 10);
    return {
      date: today,
      totalScreenTime: 120,
      topApps: ["com.example.mail", "com.example.chat"],
      notificationCount: 20,
      sessionCount: 15,
      mostCommonHour: 20,
      predictedNextApp: null,
    };
  },
  async syncDailySummary(_summary: DailySummary) {
    try {
      const endpoint = "http://10.0.2.2:8000/sync/upload-summary"
      await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(_summary),
      })
    } catch (e) {
      // best-effort; in the app we could persist for retry
      console.warn("syncDailySummary failed", e)
    }
  },
  async predictNextApp(_history: string[]) {
    // Mock: no prediction available in MLP
    return null
  },
};

// On Android we expect a NativeCollectors module; on other platforms we fall back to the mock.
// On Android we prefer an inference module if available; fall back to the mock when not present.
const nativeModule: NativeCollectorsModule | undefined =
  Platform.OS === "android"
    ? ((NativeModules.NativeCollectors as NativeCollectorsModule | undefined) || (NativeModules.NativeInference as NativeCollectorsModule | undefined))
    : undefined;

export const NativeCollectors: NativeCollectorsModule = nativeModule ?? MockModule;
