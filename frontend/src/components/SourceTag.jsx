import React from "react";

const TYPE_STYLES = {
  fda: { bg: "#dcfce7", color: "#166534", icon: "✓" },
  dataset: { bg: "#dbeafe", color: "#1e40af", icon: "◉" },
  ai: { bg: "#fef3c7", color: "#92400e", icon: "✦" },
};

export default function SourceTag({ type, label }) {
  const s = TYPE_STYLES[type] || TYPE_STYLES.ai;
  return (
    <span
      className="source-tag"
      style={{ background: s.bg, color: s.color }}
    >
      {s.icon} {label}
    </span>
  );
}
