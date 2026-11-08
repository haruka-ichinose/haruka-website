"use client";

import { useEffect, useRef, useState } from "react";
import { PaperPlaneRight, Robot, X } from "@phosphor-icons/react";

import { Button } from "@/components/ui/button";
import { ChatMessageBubble } from "./chat-message";
import { MAX_MESSAGE_CHARS, useChat } from "./use-chat";

const SUGGESTIONS = [
  "What's Haruka's tech stack?",
  "Tell me about Haruka's AI engineering work",
  "Which projects have public repos?",
  "How was this chatbot built?",
  "How do I get in touch?",
  "What has Haruka written about Next.js and FastAPI?",
];

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const { messages, isStreaming, sendMessage } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  function submit(text: string) {
    if (isStreaming || !text.trim()) return;
    sendMessage(text);
    setInput("");
    inputRef.current?.focus();
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    submit(input);
  }

  if (!open) {
    return (
      <Button
        size="icon-lg"
        onClick={() => setOpen(true)}
        aria-label="Chat with Mecha Haruka"
        className="fixed right-5 bottom-5 z-50 size-14 cursor-pointer shadow-lg"
      >
        <Robot className="size-7 group-hover/button:animate-jump" weight="fill" />
      </Button>
    );
  }

  return (
    <div className="fixed right-5 bottom-5 z-50 flex h-[80vh] max-h-[680px] w-[min(440px,calc(100vw-2.5rem))] flex-col border border-border bg-background shadow-xl">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <Robot size={16} weight="fill" className="text-primary" />
          <div className="leading-tight">
            <p className="text-sm font-semibold">Mecha Haruka</p>
            <p className="text-xs text-muted-foreground">AI persona of Haruka</p>
          </div>
        </div>
        <Button
          size="icon-xs"
          variant="ghost"
          onClick={() => setOpen(false)}
          aria-label="Close chat"
        >
          <X size={14} />
        </Button>
      </div>

      <div ref={scrollRef} className="flex flex-1 flex-col gap-3 overflow-y-auto p-3">
        {messages.length === 0 ? (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground">
              Ask about Haruka&apos;s projects, experience, skills, repos, or dev journal.
            </p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => submit(s)}
                className="border border-border bg-muted/40 px-2.5 py-2 text-left text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                {s}
              </button>
            ))}
          </div>
        ) : (
          messages.map((m, i) => <ChatMessageBubble key={i} message={m} />)
        )}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-border p-2">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={MAX_MESSAGE_CHARS}
            placeholder="Ask about Haruka…"
            className="h-9 flex-1 border border-border bg-background px-2.5 text-sm outline-none focus-visible:border-ring"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isStreaming || !input.trim()}
            aria-label="Send message"
          >
            <PaperPlaneRight size={15} weight="fill" />
          </Button>
        </div>
        <p className="mt-1.5 text-xs text-muted-foreground">
          Replies are AI-generated via Azure OpenAI. Don&apos;t share sensitive info.
        </p>
      </form>
    </div>
  );
}
