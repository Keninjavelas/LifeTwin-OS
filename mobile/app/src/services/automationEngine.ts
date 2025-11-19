// Simple rule-based automation skeleton.

import { DailySummary } from "@models/types";

export type AutomationAction =
  | { type: "SUGGEST_BREAK"; reason: string }
  | { type: "ENABLE_DND"; reason: string }
  | { type: "BLOCK_APPS"; apps: string[]; reason: string };

export function evaluateRules(summary: DailySummary): AutomationAction[] {
  const actions: AutomationAction[] = [];

  // Example rule: if social apps > 60 minutes, suggest a break.
  const socialMinutes = 0; // TODO: compute using appCategories
  if (socialMinutes > 60) {
    actions.push({ type: "SUGGEST_BREAK", reason: "High social usage in last 2 hours" });
  }

  // Example rule: if sessions after 1 AM, suggest DND.
  if (summary.mostCommonHour >= 1 && summary.mostCommonHour <= 4) {
    actions.push({ type: "ENABLE_DND", reason: "Late-night phone activity" });
  }

  return actions;
}
