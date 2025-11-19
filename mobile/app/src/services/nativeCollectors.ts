// TS bridge for native collectors. In the MLP we mock this layer;
// later, Kotlin modules will implement real data collection.

import { NativeModules, Platform } from "react-native";
import { AppEvent, DailySummary } from "@models/types";

interface NativeCollectorsModule {
  requestAllPermissions(): Promise<boolean>;
  subscribeToEvents(callback: (event: AppEvent) => void): void;
  getTodaySummary(): Promise<DailySummary | null>;
  syncDailySummary(summary: DailySummary): Promise<void>;
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
    // TODO: call backend FastAPI /sync/upload-summary
  },
};

// On Android we expect a NativeCollectors module; on other platforms we fall back to the mock.
const nativeModule: NativeCollectorsModule | undefined =
  Platform.OS === "android" ? (NativeModules.NativeCollectors as NativeCollectorsModule | undefined) : undefined;

export const NativeCollectors: NativeCollectorsModule = nativeModule ?? MockModule;
