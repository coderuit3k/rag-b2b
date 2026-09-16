"use client";

import { Bot } from "lucide-react";
import { useEffect, useRef } from "react";
import { formatDateDivider, isNewDay } from "@/lib/format";
import type { Message } from "@/lib/types";
import MessageBubble from "./MessageBubble";

export default function ChatWindow({
  messages,
  loading,
  pendingEmailNode,
  pendingCrmNode,
}: {
  messages: Message[];
  loading: boolean;
  pendingEmailNode?: React.ReactNode;
  pendingCrmNode?: React.ReactNode;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, loading, pendingEmailNode, pendingCrmNode]);

  return (
    <div className="flex-1 overflow-y-auto bg-surface px-4 py-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((m, i) => (
          <div key={i} className="flex flex-col gap-4">
            {isNewDay(messages[i - 1]?.ts, m.ts) && (
              <div className="flex justify-center">
                <span className="rounded-full bg-panel px-3 py-1 text-[11px] font-medium text-ink-soft ring-1 ring-line">
                  {formatDateDivider(m.ts!)}
                </span>
              </div>
            )}
            <MessageBubble message={m} />
          </div>
        ))}
        {pendingEmailNode}
        {pendingCrmNode}
        {loading && (
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand text-white">
              <Bot size={16} />
            </div>
            <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm border border-line bg-panel px-4 py-3">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-soft [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-soft [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-soft" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
