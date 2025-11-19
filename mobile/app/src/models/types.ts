export type EventType = "app_open" | "app_close" | "notif" | "screen_on" | "screen_off";

export interface AppEvent {
  id: string;
  timestamp: number; // ms since epoch
  type: EventType;
  appPackage?: string; // for app_open/close
  metadata?: Record<string, any>;
}

export interface DailySummary {
  date: string; // YYYY-MM-DD
  totalScreenTime: number; // minutes
  topApps: string[];
  notificationCount: number;
  sessionCount: number;
  mostCommonHour: number; // 0-23
  predictedNextApp?: string | null;
}
