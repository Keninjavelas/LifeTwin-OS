import React from "react";
import Link from "next/link";

const SettingsPage: React.FC = () => {
  return (
    <main style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>Settings</h1>
          <p style={{ margin: 0, color: "#6B7280" }}>Manage devices and sync settings (MLP stub)</p>
        </div>
        <nav style={{ display: "flex", gap: 12, fontSize: 14 }}>
          <Link href="/">Home</Link>
          <Link href="/timeline">Timeline</Link>
          <Link href="/analytics/trends">Trends</Link>
          <Link href="/analytics/heatmap">Heatmap</Link>
          <Link href="/analytics/simulate">Simulate</Link>
        </nav>
      </header>
      <p>Here you will manage connected devices and see last sync timestamps.</p>
    </main>
  );
};

export default SettingsPage;
