"use client";

import { SendHorizontal } from "lucide-react";
import { useRef, useState } from "react";
import VoiceButton from "./VoiceButton";

export default function ChatInput({
  onSend,
  onSendVoice,
  disabled,
}: {
  onSend: (question: string) => void;
  onSendVoice: (audioBlob: Blob) => void;
  disabled: boolean;
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
    <div className="border-t border-line bg-panel p-3">
      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-3xl bg-surface px-2 py-2 ring-1 ring-line focus-within:ring-2 focus-within:ring-brand">
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
          placeholder="Nhập tin nhắn... (Shift+Enter xuống dòng)"
          disabled={disabled}
          className="max-h-[200px] flex-1 resize-none bg-transparent px-2.5 py-2 text-sm text-ink placeholder:text-ink-soft focus:outline-none disabled:text-ink-soft"
        />
        {text.trim() ? (
          <button
            onClick={submit}
            disabled={disabled}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-brand text-white transition-colors hover:bg-brand-strong disabled:bg-ink-soft"
            aria-label="Gửi"
          >
            <SendHorizontal size={18} />
          </button>
        ) : (
          <VoiceButton onRecorded={onSendVoice} disabled={disabled} />
        )}
      </div>
    </div>
  );
}
