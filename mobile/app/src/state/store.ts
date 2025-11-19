import { create } from "zustand";
import { AppEvent, DailySummary } from "@models/types";

interface AppState {
  events: AppEvent[];
  dailySummary: DailySummary | null;
  markovTable: Record<string, Record<string, number>>; // prevApp -> nextApp -> count
  addEvent: (event: AppEvent) => void;
  setDailySummary: (summary: DailySummary) => void;
  predictNextApp: () => string | null;
}

export const useAppStore = create<AppState>((set, get) => ({
  events: [],
  dailySummary: null,
  markovTable: {},

  addEvent: (event) =>
    set((state) => {
      const events = [...state.events, event];
      // update simple Markov counts for app_open transitions
      if (event.type === "app_open" && events.length > 1) {
        const prevEvent = [...events].reverse().find((e) => e.type === "app_open" && e.appPackage !== event.appPackage);
        if (prevEvent && prevEvent.appPackage) {
          const prevApp = prevEvent.appPackage;
          const nextApp = event.appPackage!;
          const table = { ...state.markovTable };
          table[prevApp] = table[prevApp] || {};
          table[prevApp][nextApp] = (table[prevApp][nextApp] || 0) + 1;
          return { events, markovTable: table };
        }
      }
      return { events };
    }),

  setDailySummary: (summary) => set({ dailySummary: summary }),

  predictNextApp: () => {
    const { events, markovTable } = get();
    const lastAppEvent = [...events].reverse().find((e) => e.type === "app_open");
    if (!lastAppEvent || !lastAppEvent.appPackage) return null;
    const transitions = markovTable[lastAppEvent.appPackage];
    if (!transitions) return null;

    // choose argmax next app
    let bestApp: string | null = null;
    let bestCount = -1;
    Object.entries(transitions).forEach(([app, count]) => {
      if (count > bestCount) {
        bestApp = app;
        bestCount = count;
      }
    });
    return bestApp;
  },
}));
