import React, { useEffect, useState } from "react";

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

  useEffect(() => {
    // MLP: simple fetch from local backend at localhost:8000
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
      }
    };
    fetchData();
  }, []);

  const latest = summaries[summaries.length - 1];

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>LifeTwin OS — MLP Dashboard</h1>
      {latest ? (
        <section style={{ marginTop: 16 }}>
          <h2>Today</h2>
          <p>Total screen time: {latest.total_screen_time} min</p>
          <p>Top apps: {latest.top_apps.join(", ")}</p>
          <p>Notifications: {latest.notification_count}</p>
          <p>Peak usage hour: {latest.most_common_hour}:00</p>
          <p>Predicted next app: {latest.predicted_next_app ?? "(none)"}</p>
        </section>
      ) : (
        <p>No summaries synced yet.</p>
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
