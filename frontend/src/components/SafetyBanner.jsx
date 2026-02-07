import React from "react";

export default function SafetyBanner() {
  return (
    <div className="safety-banner">
      <span className="safety-icon">⚕️</span>
      <p>
        <strong>Medical Disclaimer:</strong> PharmaBot is an AI assistant
        providing information for <em>educational purposes only</em>. It is{" "}
        <strong>NOT</strong> a substitute for professional medical advice,
        diagnosis, or treatment. Always consult a qualified healthcare provider.
      </p>
    </div>
  );
}
