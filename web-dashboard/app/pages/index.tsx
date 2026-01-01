import React, { useEffect, useState } from "react";
import Link from "next/link";

interface DailySummary {
  timestamp: string;
  total_screen_time: number;
  top_apps: string[];
  most_common_hour: number;
  predicted_next_app?: string | null;
  notification_count: number;
}

const DashboardPage: React.FC = () => {
  const [summaries, setSummaries] = useState<DailySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const loginRes = await fetch("http://localhost:8000/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device_id: "demo-device" }),
        });
        const { token } = await loginRes.json();
        const res = await fetch("http://localhost:8000/sync/get-summary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token, device_id: "demo-device" }),
        });
        const data = await res.json();
        setSummaries(data.summaries ?? []);
      } catch (e) {
        console.error(e);
        setError("Failed to load summaries.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const latest = summaries[summaries.length - 1];

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>LifeTwin OS — MLP Dashboard</h1>
          <p style={{ margin: 0, color: "#6B7280" }}>Overview of your device behavior and predictions</p>
        </div>
        <nav style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <Link href="/">Home</Link>
          <Link href="/timeline">Timeline</Link>
          <Link href="/analytics/trends">Trends</Link>
          <Link href="/analytics/heatmap">Heatmap</Link>
          <Link href="/simulation">Simulation</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </header>

      {loading && <p>Loading summaries…</p>}
      {error && <p style={{ color: "#DC2626" }}>{error}</p>}

      {!loading && !latest && !error && <p>No summaries synced yet.</p>}

      {latest && (
        <section style={{ marginTop: 16 }}>
          <h2>Today</h2>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div style={{ padding: 12, borderRadius: 8, background: "#0F172A", color: "#E5E7EB", minWidth: 200 }}>
              <div style={{ fontSize: 12, color: "#9CA3AF" }}>Total screen time</div>
              <div style={{ fontSize: 20, fontWeight: 600 }}>{latest.total_screen_time} min</div>
            </div>
            <div style={{ padding: 12, borderRadius: 8, background: "#0F172A", color: "#E5E7EB", minWidth: 200 }}>
              <div style={{ fontSize: 12, color: "#9CA3AF" }}>Top apps</div>
              <div>{latest.top_apps.join(", ")}</div>
            </div>
            <div style={{ padding: 12, borderRadius: 8, background: "#0F172A", color: "#E5E7EB", minWidth: 200 }}>
              <div style={{ fontSize: 12, color: "#9CA3AF" }}>Notifications</div>
              <div>{latest.notification_count}</div>
            </div>
            <div style={{ padding: 12, borderRadius: 8, background: "#0F172A", color: "#E5E7EB", minWidth: 200 }}>
              <div style={{ fontSize: 12, color: "#9CA3AF" }}>Peak usage hour</div>
              <div>{latest.most_common_hour}:00</div>
            </div>
            <div style={{ padding: 12, borderRadius: 8, background: "#0F172A", color: "#E5E7EB", minWidth: 200 }}>
              <div style={{ fontSize: 12, color: "#9CA3AF" }}>Predicted next app</div>
              <div>{latest.predicted_next_app ?? "(none)"}</div>
            </div>
          </div>
        </section>
      )}

      <section style={{ marginTop: 32 }}>
        <h2>History</h2>
        <ul>
          {summaries.map((s, idx) => (
            <li key={idx}>
              {new Date(s.timestamp).toLocaleString()} — {s.total_screen_time} min, top app: {s.top_apps[0] ?? "-"}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
};

export default DashboardPage;
