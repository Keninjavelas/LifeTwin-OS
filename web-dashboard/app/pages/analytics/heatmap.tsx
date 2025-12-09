import React from "react";
import Link from "next/link";

const HeatmapPage: React.FC = () => {
  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>Usage Heatmap</h1>
          <p style={{ margin: 0, color: "#6B7280" }}>Hourly usage and behavior clusters (MLP placeholder)</p>
        </div>
        <nav style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <Link href="/">Home</Link>
          <Link href="/timeline">Timeline</Link>
          <Link href="/analytics/trends">Trends</Link>
          <Link href="/analytics/simulate">Simulate</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </header>
      <p>Placeholder for hourly usage heatmap and behavior clusters using Recharts/D3.</p>
    </main>
  );
};

export default HeatmapPage;
