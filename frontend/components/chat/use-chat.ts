"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
// Mirrors the backend: models/chat.py MAX_MESSAGE_CHARS and the history the
// service keeps (core/guardrails.py), so we don't send what it would drop.
export const MAX_MESSAGE_CHARS = 1000;
const MAX_HISTORY_MESSAGES = 6;

export interface Citation {
  title: string;
  source_type: "journal" | "project" | "experience" | "page" | "profile" | "repo";
  url: string;
  repo_url?: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface StreamEvent {
  type: "token" | "citations" | "done" | "error";
  delta?: string;
  citations?: Citation[];
  message?: string;
}

class StreamError extends Error {}

/**
 * Chat state + SSE-over-POST streaming against the backend `/chat` endpoint.
 * Consumes the response with fetch + ReadableStream (not React Query) so token
 * deltas render as they arrive; the trailing `citations` frame is attached to
 * the final assistant message.
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Mirror of messages so sendMessage can read prior history without stale closures.
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const updateLast = useCallback((fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      const next = [...prev];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;

      setError(null);
      const history = messagesRef.current.slice(-MAX_HISTORY_MESSAGES).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      setMessages((prev) => [
        ...prev,
        { role: "user", content: trimmed },
        { role: "assistant", content: "" },
      ]);
      setIsStreaming(true);

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: trimmed, history }),
        });
        if (!res.ok || !res.body) throw new Error(`Server error ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary: number;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, boundary).trim();
            buffer = buffer.slice(boundary + 2);
            if (!frame.startsWith("data:")) continue;

            const payload = frame.slice(5).trim();
            if (!payload) continue;

            let event: StreamEvent;
            try {
              event = JSON.parse(payload);
            } catch {
              continue;
            }

            if (event.type === "token" && event.delta) {
              updateLast((m) => ({ ...m, content: m.content + event.delta }));
            } else if (event.type === "citations") {
              updateLast((m) => ({ ...m, citations: event.citations ?? [] }));
            } else if (event.type === "error") {
              throw new StreamError(event.message ?? "Sorry, I couldn't answer that right now.");
            }
          }
        }
      } catch (err) {
        setError("Something went wrong. Please try again.");
        // Stream errors carry a user-facing message (e.g. the rate limit).
        const message =
          err instanceof StreamError ? err.message : "Sorry, I couldn't answer that right now.";
        updateLast((m) => ({ ...m, content: m.content || message }));
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming, updateLast]
  );

  return { messages, isStreaming, error, sendMessage };
}
