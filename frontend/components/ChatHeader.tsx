"use client";

import { Sparkles } from "lucide-react";
import { FORCE_ROUTES, type ForceRoute } from "@/lib/types";

export default function ChatHeader({
  conversationTitle,
  forceRoute,
  onForceRouteChange,
}: {
  conversationTitle?: string;
  forceRoute: ForceRoute;
  onForceRouteChange: (route: ForceRoute) => void;
}) {
  return (
    <header className="flex items-center justify-between border-b border-line bg-panel px-5 py-3">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand text-white">
          <Sparkles size={18} />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-ink">
            Trợ lý B2B{conversationTitle ? ` · ${conversationTitle}` : ""}
          </p>
          <p className="flex items-center gap-1.5 text-xs text-ink-soft">
            <span className="h-1.5 w-1.5 rounded-full bg-online" />
            Đang hoạt động
          </p>
        </div>
      </div>

      <select
        value={forceRoute}
        onChange={(e) => onForceRouteChange(e.target.value as ForceRoute)}
        title="Ép luồng xử lý thay vì để Auto-Router tự chọn"
        className="rounded-full bg-surface px-3 py-1.5 text-xs font-medium text-ink-soft ring-1 ring-line focus:outline-none focus:ring-2 focus:ring-brand"
      >
        {FORCE_ROUTES.map((r) => (
          <option key={r.value} value={r.value}>
            {r.label}
          </option>
        ))}
      </select>
    </header>
  );
}
