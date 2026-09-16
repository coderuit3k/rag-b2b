import { Building2 } from "lucide-react";

export default function CrmConfirmCard({
  preview,
  onConfirm,
  onCancel,
  pending,
}: {
  preview: string;
  onConfirm: () => void;
  onCancel: () => void;
  pending: boolean;
}) {
  return (
    <div className="message-rise-in flex flex-row gap-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-amber-tint text-amber">
        <Building2 size={16} />
      </div>
      <div className="max-w-[75%] rounded-2xl rounded-tl-sm border border-[#f0930b4d] bg-amber-tint px-4 py-3 text-sm">
        <p className="mb-2 font-medium text-ink">Cập nhật CRM (HubSpot):</p>
        <p className="text-ink-soft">{preview}</p>
        <div className="mt-3 flex gap-2">
          <button
            onClick={onConfirm}
            disabled={pending}
            className="rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-strong disabled:opacity-50"
          >
            Duyệt
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
