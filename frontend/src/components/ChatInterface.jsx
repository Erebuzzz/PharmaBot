import React, { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import { sendChat } from "../api";

const EXAMPLE_QUERIES = [
  "What is amoxicillin used for?",
  "Side effects of ibuprofen",
  "What are symptoms of diabetes?",
  "Tell me about paracetamol dosage",
];

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const bottomRef = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(text) {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    const userMsg = { role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(msg, threadId);
      setThreadId(data.thread_id);

      const botMsg = {
        role: "assistant",
        content: data.response,
        sources: data.sources,
        confidence: data.confidence,
        medicines_referenced: data.medicines_referenced,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ Error: ${err.message}. Please check that the backend is running.`,
          sources: [],
          confidence: "N/A",
          medicines_referenced: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleNewThread() {
    setMessages([]);
    setThreadId(null);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-container">
      {/* Messages area */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">💊</div>
            <h2>Welcome to PharmaBot</h2>
            <p>Ask me about medicines, side effects, dosage, or symptoms.</p>
            <p className="chat-empty-sub">
              I use FDA-approved data and verified datasets to ensure accuracy.
            </p>
            <div className="example-queries">
              {EXAMPLE_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => handleSend(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {loading && (
          <div className="msg-row msg-row-bot">
            <div className="msg-avatar">💊</div>
            <div className="msg-bubble msg-bubble-bot">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        {threadId && (
          <button className="new-thread-btn" onClick={handleNewThread}>
            + New Conversation
          </button>
        )}
        <div className="chat-input-row">
          <textarea
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about a medicine, symptoms, or drug information…"
            rows={1}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => handleSend()}
            disabled={!input.trim() || loading}
          >
            ➤
          </button>
        </div>
        <p className="input-hint">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
