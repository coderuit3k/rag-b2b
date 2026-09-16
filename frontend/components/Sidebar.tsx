"use client";

import { UserButton } from "@clerk/nextjs";
import { Search, Sparkles, SquarePen } from "lucide-react";
import { useState } from "react";
import { formatRelative } from "@/lib/format";
import type { Conversations } from "@/lib/types";

function lastMessagePreview(conversations: Conversations, id: string) {
  const messages = conversations[id]?.messages ?? [];
  const last = messages[messages.length - 1];
  if (!last) return { snippet: "", ts: undefined as number | undefined };
  const prefix = last.role === "user" ? "Bạn: " : "";
  return { snippet: `${prefix}${last.content.slice(0, 60)}`, ts: last.ts };
}

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNewChat,
}: {
  conversations: Conversations;
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}) {
  const [query, setQuery] = useState("");

  // Mới nhất lên đầu — Object giữ thứ tự chèn (bảo bảo bởi backend chat_store.py).
  const ids = Object.keys(conversations)
    .reverse()
    .filter((id) => conversations[id].title.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-line bg-surface">
      <div className="flex items-center justify-between px-4 pt-4 pb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-brand text-white">
            <Sparkles size={16} />
          </div>
          <span className="text-[15px] font-semibold text-ink">RAG B2B</span>
        </div>
        <button
          onClick={onNewChat}
          title="Hội thoại mới"
          aria-label="Hội thoại mới"
          className="flex h-8 w-8 items-center justify-center rounded-full text-ink-soft transition-colors hover:bg-panel hover:text-brand"
        >
          <SquarePen size={18} />
        </button>
      </div>

      <div className="px-3 pb-2">
        <div className="flex items-center gap-2 rounded-full bg-panel px-3 py-2 ring-1 ring-line focus-within:ring-2 focus-within:ring-brand">
          <Search size={15} className="shrink-0 text-ink-soft" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Tìm hội thoại..."
            className="w-full bg-transparent text-sm text-ink placeholder:text-ink-soft focus:outline-none"
          />
        </div>
      </div>

      <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
        {ids.length === 0 && (
          <p className="px-3 py-2 text-xs text-ink-soft">
            {query ? "Không tìm thấy hội thoại nào." : "Chưa có hội thoại nào."}
          </p>
        )}
        {ids.map((id) => {
          const active = id === activeId;
          const { snippet, ts } = lastMessagePreview(conversations, id);
          return (
            <button
              key={id}
              onClick={() => onSelect(id)}
              className={`relative flex w-full items-center gap-3 rounded-xl px-2.5 py-2.5 text-left transition-colors ${
                active ? "bg-brand-tint" : "hover:bg-panel"
              }`}
            >
              {active && <span className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-brand" />}
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-panel text-brand ring-1 ring-line">
                <Sparkles size={18} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className={`truncate text-sm ${active ? "font-semibold text-brand-strong" : "font-medium text-ink"}`}>
                    {conversations[id].title}
                  </span>
                  {ts && <span className="shrink-0 text-[11px] text-ink-soft">{formatRelative(ts)}</span>}
                </div>
                <p className="truncate text-xs text-ink-soft">{snippet || "Chưa có tin nhắn"}</p>
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2 border-t border-line px-3 py-3">
        <UserButton showName />
      </div>
    </aside>
  );
}
