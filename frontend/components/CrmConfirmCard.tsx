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
    <div className="flex flex-row gap-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-orange-100 text-orange-700">
        <Building2 size={16} />
      </div>
      <div className="max-w-[75%] rounded-2xl rounded-tl-sm border border-orange-200 bg-orange-50 px-4 py-3 text-sm">
        <p className="mb-2 font-medium text-orange-900">🏢 Cập nhật CRM (HubSpot):</p>
        <p className="text-zinc-700">{preview}</p>
        <div className="mt-3 flex gap-2">
          <button
            onClick={onConfirm}
            disabled={pending}
            className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            ✅ Duyệt
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
