import { Mail } from "lucide-react";
import ReactMarkdown from "react-markdown";

export default function EmailConfirmCard({
  recipient,
  preview,
  onConfirm,
  onCancel,
  pending,
}: {
  recipient: string;
  preview: string;
  onConfirm: () => void;
  onCancel: () => void;
  pending: boolean;
}) {
  return (
    <div className="message-rise-in flex flex-row gap-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-tint text-amber">
        <Mail size={16} />
      </div>
      <div className="max-w-[75%] rounded-2xl rounded-tl-sm border border-[#f0930b4d] bg-amber-tint px-4 py-3 text-sm">
        <p className="mb-2 font-medium text-ink">
          Sẽ gửi tới <span className="font-semibold">{recipient}</span>:
        </p>
        <div className="prose prose-sm max-w-none prose-p:my-1">
          <ReactMarkdown>{preview}</ReactMarkdown>
        </div>
        <div className="mt-3 flex gap-2">
          <button
            onClick={onConfirm}
            disabled={pending}
            className="rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-strong disabled:opacity-50"
          >
            Xác nhận gửi
          </button>
          <button
            onClick={onCancel}
            disabled={pending}
            className="rounded-lg bg-panel px-3 py-1.5 text-xs font-medium text-ink-soft ring-1 ring-line hover:text-ink disabled:opacity-50"
          >
            Huỷ
          </button>
        </div>
      </div>
    </div>
  );
}
