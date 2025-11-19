export type AppCategory = "social" | "productivity" | "entertainment" | "system" | "other";

// Simple static mapping; later can be learned or fetched.
export const APP_CATEGORY_MAP: Record<string, AppCategory> = {
  "com.facebook.katana": "social",
  "com.instagram.android": "social",
  "com.whatsapp": "social",
  "com.google.android.gm": "productivity",
};

export function getCategoryForPackage(pkg?: string): AppCategory {
  if (!pkg) return "other";
  return APP_CATEGORY_MAP[pkg] ?? "other";
}

export function isValidEventTimestamp(ts: number): boolean {
  // Very basic validation; real implementation would be stricter
  return ts > 0 && ts < Date.now() + 24 * 60 * 60 * 1000;
}
