"use client";

import { useEffect, useRef } from "react";
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
    <div className="flex-1 overflow-y-auto bg-zinc-50 px-4 py-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {pendingEmailNode}
        {pendingCrmNode}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-400" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
