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
    <div className="flex flex-row gap-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-700">
        <Mail size={16} />
      </div>
      <div className="max-w-[75%] rounded-2xl rounded-tl-sm border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
        <p className="mb-2 font-medium text-amber-900">
          📧 Sẽ gửi tới <span className="font-semibold">{recipient}</span>:
        </p>
        <div className="prose prose-sm max-w-none prose-p:my-1">
          <ReactMarkdown>{preview}</ReactMarkdown>
        </div>
        <div className="mt-3 flex gap-2">
          <button
            onClick={onConfirm}
            disabled={pending}
            className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            ✅ Xác nhận gửi
          </button>
          <button
            onClick={onCancel}
            disabled={pending}
            className="rounded-lg bg-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-300 disabled:opacity-50"
          >
            ❌ Huỷ
          </button>
        </div>
      </div>
    </div>
  );
}
