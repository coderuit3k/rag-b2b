"use client";

import { SendHorizontal } from "lucide-react";
import { useRef, useState } from "react";
import { FORCE_ROUTES, type ForceRoute } from "@/lib/types";
import VoiceButton from "./VoiceButton";

export default function ChatInput({
  onSend,
  onSendVoice,
  disabled,
  forceRoute,
  onForceRouteChange,
}: {
  onSend: (question: string) => void;
  onSendVoice: (audioBlob: Blob) => void;
  disabled: boolean;
  forceRoute: ForceRoute;
  onForceRouteChange: (route: ForceRoute) => void;
}) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function autoResize(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  function submit() {
    const question = text.trim();
    if (!question || disabled) return;
    onSend(question);
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  return (
    <div className="border-t border-zinc-200 bg-white p-3">
      <div className="mx-auto flex max-w-3xl flex-col gap-2">
        <select
          value={forceRoute}
          onChange={(e) => onForceRouteChange(e.target.value as ForceRoute)}
          className="w-fit rounded-md border border-zinc-200 bg-zinc-50 px-2 py-1 text-xs text-zinc-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
          title="Ép luồng xử lý thay vì để Auto-Router tự chọn"
        >
          {FORCE_ROUTES.map((r) => (
            <option key={r.value} value={r.value}>
              {r.label}
            </option>
          ))}
        </select>
        <div className="flex items-end gap-2">
          <VoiceButton onRecorded={onSendVoice} disabled={disabled} />
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              autoResize(e.target);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Nhập câu hỏi... (Shift+Enter xuống dòng)"
            disabled={disabled}
            className="max-h-[200px] flex-1 resize-none rounded-xl border border-zinc-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-zinc-50 disabled:text-zinc-400"
          />
          <button
            onClick={submit}
            disabled={disabled || !text.trim()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white transition-colors hover:bg-blue-700 disabled:bg-zinc-300"
            aria-label="Gửi"
          >
            <SendHorizontal size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
