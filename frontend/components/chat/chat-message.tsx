import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowUpRight, GithubLogo } from "@phosphor-icons/react";

import { cn } from "@/lib/utils";
import type { ChatMessage, Citation } from "./use-chat";

const chipClass =
  "inline-flex items-center gap-1 border border-border bg-background px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground";

const linkClass = "text-foreground underline underline-offset-4 hover:text-muted-foreground";

// Assistant answers are markdown; site routes navigate client-side so the
// widget stays open, everything else opens in a new tab.
const markdownComponents: Components = {
  p: ({ ...props }) => <p className="mb-2 last:mb-0" {...props} />,
  ul: ({ ...props }) => <ul className="mb-2 list-disc space-y-0.5 pl-5 last:mb-0" {...props} />,
  ol: ({ ...props }) => <ol className="mb-2 list-decimal space-y-0.5 pl-5 last:mb-0" {...props} />,
  strong: ({ ...props }) => <strong className="font-semibold" {...props} />,
  code: ({ ...props }) => <code className="bg-background px-1 text-xs" {...props} />,
  a: ({ href = "", children }) =>
    href.startsWith("/") ? (
      <Link href={href} className={linkClass}>
        {children}
      </Link>
    ) : (
      <a href={href} target="_blank" rel="noopener noreferrer" className={linkClass}>
        {children}
      </a>
    ),
};

function CitationChips({ citation }: { citation: Citation }) {
  const isInternal = citation.url.startsWith("/");
  return (
    <>
      {isInternal ? (
        <Link href={citation.url} className={chipClass}>
          {citation.title}
          <ArrowUpRight size={11} />
        </Link>
      ) : (
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className={chipClass}
        >
          {citation.title}
          <ArrowUpRight size={11} />
        </a>
      )}
      {citation.repo_url && (
        <a
          href={citation.repo_url}
          target="_blank"
          rel="noopener noreferrer"
          className={chipClass}
        >
          <GithubLogo size={11} />
          Repo
        </a>
      )}
    </>
  );
}

export function ChatMessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex flex-col gap-1.5", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[85%] border px-3 py-2 text-sm leading-relaxed",
          isUser
            ? "border-transparent bg-primary text-primary-foreground"
            : "border-border bg-muted text-foreground"
        )}
      >
        {!message.content ? (
          <span className="inline-block animate-pulse text-muted-foreground">…</span>
        ) : isUser ? (
          <span className="whitespace-pre-wrap">{message.content}</span>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {message.content}
          </ReactMarkdown>
        )}
      </div>

      {message.citations && message.citations.length > 0 && (
        <div className="flex max-w-[85%] flex-wrap gap-1.5">
          {message.citations.map((c, i) => (
            <CitationChips key={i} citation={c} />
          ))}
        </div>
      )}
    </div>
  );
}
