import React from "react";
import SourceTag from "./SourceTag";
import MedicineCard from "./MedicineCard";

const CONFIDENCE_BADGE = {
  high: { className: "badge-high", text: "✓ FDA Verified" },
  medium: { className: "badge-medium", text: "◉ Dataset Sourced" },
  low: { className: "badge-low", text: "⚠ Limited Data" },
};

/** Renders markdown-like bold and bullet points. */
function formatText(text) {
  if (!text) return null;

  return text.split("\n").map((line, i) => {
    // Bold: **text**
    const parts = line.split(/(\*\*[^*]+\*\*)/g).map((part, j) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={j}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });

    // Bullet points
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("• ") || trimmed.startsWith("* ")) {
      return (
        <li key={i} className="msg-list-item">
          {parts}
        </li>
      );
    }

    // Headings (###)
    if (trimmed.startsWith("### ")) {
      return <h4 key={i} className="msg-heading">{trimmed.slice(4)}</h4>;
    }
    if (trimmed.startsWith("## ")) {
      return <h3 key={i} className="msg-heading">{trimmed.slice(3)}</h3>;
    }

    return (
      <p key={i} className="msg-paragraph">
        {parts}
      </p>
    );
  });
}

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`msg-row ${isUser ? "msg-row-user" : "msg-row-bot"}`}>
      <div className="msg-avatar">{isUser ? "👤" : "💊"}</div>

      <div className={`msg-bubble ${isUser ? "msg-bubble-user" : "msg-bubble-bot"}`}>
        <div className="msg-content">{formatText(message.content)}</div>

        {/* Confidence badge */}
        {!isUser && message.confidence && message.confidence !== "N/A" && (
          <div className="msg-meta">
            <span className={`confidence-badge ${CONFIDENCE_BADGE[message.confidence]?.className || ""}`}>
              {CONFIDENCE_BADGE[message.confidence]?.text || message.confidence}
            </span>
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources?.length > 0 && (
          <div className="msg-sources">
            {message.sources.map((s, i) => (
              <SourceTag key={i} type={s.type} label={s.label} />
            ))}
          </div>
        )}

        {/* Medicine cards */}
        {!isUser && message.medicines_referenced?.length > 0 && (
          <div className="msg-medicines">
            {message.medicines_referenced.map((m, i) => (
              <MedicineCard key={i} medicine={m} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
