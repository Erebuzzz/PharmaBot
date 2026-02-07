import React from "react";
import SafetyBanner from "./components/SafetyBanner";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

function App() {
  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <span className="header-icon">💊</span>
          <h1>PharmaBot</h1>
        </div>
        <span className="header-tag">AI-Powered Medicine Information</span>
      </header>

      {/* Safety disclaimer */}
      <SafetyBanner />

      {/* Main chat area */}
      <main className="app-main">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;
