"use client";

import { UserButton } from "@clerk/nextjs";
import { Pencil, Search, Sparkles, SquarePen, Trash2 } from "lucide-react";
import { useRef, useState } from "react";
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
  onRename,
  onDelete,
}: {
  conversations: Conversations;
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const cancelRef = useRef(false);

  function startEdit(id: string) {
    setEditingId(id);
    setEditValue(conversations[id].title);
  }

  function commitEdit(id: string) {
    const title = editValue.trim();
    setEditingId(null);
    if (title && title !== conversations[id].title) onRename(id, title);
  }

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
          const editing = editingId === id;
          const { snippet, ts } = lastMessagePreview(conversations, id);
          return (
            <div
              key={id}
              className={`group relative flex w-full items-center rounded-xl px-2.5 py-2.5 transition-colors ${
                active ? "bg-brand-tint" : "hover:bg-panel"
              }`}
            >
              {active && <span className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-brand" />}
              <div
                role="button"
                tabIndex={0}
                onClick={editing ? undefined : () => onSelect(id)}
                onKeyDown={(e) => {
                  if (!editing && e.key === "Enter") onSelect(id);
                }}
                className="flex min-w-0 flex-1 cursor-pointer items-center gap-3"
              >
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-panel text-brand ring-1 ring-line">
                  <Sparkles size={18} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    {editing ? (
                      <input
                        autoFocus
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onKeyDown={(e) => {
                          e.stopPropagation();
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitEdit(id);
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            cancelRef.current = true;
                            setEditingId(null);
                          }
                        }}
                        onBlur={() => {
                          if (cancelRef.current) {
                            cancelRef.current = false;
                            return;
                          }
                          commitEdit(id);
                        }}
                        maxLength={60}
                        className="w-full rounded-md bg-panel px-1.5 py-0.5 text-sm text-ink ring-2 ring-brand focus:outline-none"
                      />
                    ) : (
                      <span
                        className={`truncate pr-12 text-sm ${
                          active ? "font-semibold text-brand-strong" : "font-medium text-ink"
                        }`}
                      >
                        {conversations[id].title}
                      </span>
                    )}
                    {!editing && ts && <span className="shrink-0 text-[11px] text-ink-soft">{formatRelative(ts)}</span>}
                  </div>
                  {!editing && <p className="truncate text-xs text-ink-soft">{snippet || "Chưa có tin nhắn"}</p>}
                </div>
              </div>
              {!editing && (
                <div className="absolute top-1/2 right-2 flex -translate-y-1/2 shrink-0 items-center gap-0.5">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startEdit(id);
                    }}
                    aria-label="Đổi tên hội thoại"
                    title="Đổi tên"
                    className="flex h-7 w-7 items-center justify-center rounded-full text-ink-soft opacity-0 transition-opacity hover:bg-line hover:text-ink group-hover:opacity-100 focus:opacity-100"
                  >
                    <Pencil size={13} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm(`Xoá hội thoại "${conversations[id].title}"? Không thể hoàn tác.`)) {
                        onDelete(id);
                      }
                    }}
                    aria-label="Xoá hội thoại"
                    title="Xoá"
                    className="flex h-7 w-7 items-center justify-center rounded-full text-ink-soft opacity-0 transition-opacity hover:bg-red-100 hover:text-red-600 group-hover:opacity-100 focus:opacity-100"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2 border-t border-line px-3 py-3">
        <UserButton showName />
      </div>
    </aside>
  );
}
