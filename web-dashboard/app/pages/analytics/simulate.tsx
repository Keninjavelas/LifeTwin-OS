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

interface SimulationResult {
  base: any;
  simulated: any;
}

const SimulatePage: React.FC = () => {
  const [latest, setLatest] = useState<DailySummary | null>(null);
  const [bedtimeShift, setBedtimeShift] = useState(0);
  const [socialDelta, setSocialDelta] = useState(0);
  const [result, setResult] = useState<SimulationResult | null>(null);

  useEffect(() => {
    const fetchLatest = async () => {
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
        const summaries: DailySummary[] = data.summaries ?? [];
        setLatest(summaries[summaries.length - 1] ?? null);
      } catch (e) {
        console.error(e);
      }
    };
    fetchLatest();
  }, []);

  const runSimulation = async () => {
    if (!latest) return;
    try {
      const baseHistory = {
        total_screen_time: latest.total_screen_time,
        most_common_hour: latest.most_common_hour,
        notification_count: latest.notification_count,
      };
      const res = await fetch("http://localhost:8000/simulate/what-if", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_history: baseHistory,
          bedtime_shift_hours: bedtimeShift,
          social_usage_delta_min: socialDelta,
        }),
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>What-if Simulation (Prototype)</h1>
          <p style={{ margin: 0, color: "#6B7280" }}>Experiment with bedtime and social usage (stubbed model)</p>
        </div>
        <nav style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <Link href="/">Home</Link>
          <Link href="/timeline">Timeline</Link>
          <Link href="/analytics/trends">Trends</Link>
          <Link href="/analytics/heatmap">Heatmap</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </header>
      {latest ? (
        <section style={{ marginTop: 16 }}>
          <h2>Latest Summary</h2>
          <p>Total screen time: {latest.total_screen_time} min</p>
          <p>Peak usage hour: {latest.most_common_hour}:00</p>
        </section>
      ) : (
        <p>No summaries available yet. Seed one, then reload.</p>
      )}

      <section style={{ marginTop: 24 }}>
        <h2>Scenario Controls</h2>
        <label>
          Bedtime shift (hours):
          <input
            type="number"
            value={bedtimeShift}
            onChange={(e) => setBedtimeShift(parseInt(e.target.value || "0", 10))}
            style={{ marginLeft: 8 }}
          />
        </label>
        <br />
        <label>
          Social usage delta (minutes):
          <input
            type="number"
            value={socialDelta}
            onChange={(e) => setSocialDelta(parseInt(e.target.value || "0", 10))}
            style={{ marginLeft: 8 }}
          />
        </label>
        <br />
        <button style={{ marginTop: 12 }} onClick={runSimulation} disabled={!latest}>
          Run simulation
        </button>
      </section>

      {result && (
        <section style={{ marginTop: 32 }}>
          <h2>Result (stubbed)</h2>
          <pre style={{ background: "#f4f4f4", padding: 12 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </section>
      )}
    </main>
  );
};

export default SimulatePage;
