import React from "react";
import Link from "next/link";

const TimelinePage: React.FC = () => {
  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>Timeline</h1>
          <p style={{ margin: 0, color: "#6B7280" }}>Chronological view of your summarized activity (MLP stub)</p>
        </div>
        <nav style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <Link href="/">Home</Link>
          <Link href="/analytics/trends">Trends</Link>
          <Link href="/analytics/heatmap">Heatmap</Link>
          <Link href="/analytics/simulate">Simulate</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </header>
      <p>
        This page will show a chronological view of app launches and sessions. In the MLP, the data comes from daily
        summaries only.
      </p>
    </main>
  );
};

export default TimelinePage;
