"use client";

import { UserButton } from "@clerk/nextjs";
import { MessageSquare, Plus } from "lucide-react";
import type { Conversations } from "@/lib/types";

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
  // Mới nhất lên đầu — Object giữ thứ tự chèn (bảo bảo bởi backend chat_store.py).
  const ids = Object.keys(conversations).reverse();

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-200 bg-white">
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
        >
          <Plus size={16} /> Hội thoại mới
        </button>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-2">
        {ids.length === 0 && (
          <p className="px-3 py-2 text-xs text-zinc-400">Chưa có hội thoại nào.</p>
        )}
        {ids.map((id) => (
          <button
            key={id}
            onClick={() => onSelect(id)}
            className={`flex w-full items-center gap-2 truncate rounded-lg px-3 py-2 text-left text-sm transition-colors ${
              id === activeId ? "bg-blue-50 text-blue-700" : "text-zinc-700 hover:bg-zinc-100"
            }`}
          >
            <MessageSquare size={15} className="shrink-0" />
            <span className="truncate">{conversations[id].title}</span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2 border-t border-zinc-200 p-3">
        <UserButton showName />
      </div>
    </aside>
  );
}
