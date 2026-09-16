const clockFmt = new Intl.DateTimeFormat("vi-VN", { hour: "2-digit", minute: "2-digit" });
const dateFmt = new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });

function isSameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

/** Giờ:phút của 1 tin nhắn — hiển thị nhỏ dưới mỗi bubble, giống Zalo/Telegram. */
export function formatClock(ts?: number | null): string {
  if (!ts) return "";
  return clockFmt.format(ts * 1000);
}

/** Giờ tương đối cho preview trong sidebar (danh sách hội thoại). */
export function formatRelative(ts?: number | null): string {
  if (!ts) return "";
  const diffMin = Math.floor((Date.now() - ts * 1000) / 60000);
  if (diffMin < 1) return "Vừa xong";
  if (diffMin < 60) return `${diffMin} phút`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} giờ`;
  const d = new Date(ts * 1000);
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (isSameDay(d, yesterday)) return "Hôm qua";
  return dateFmt.format(d);
}

/** Nhãn dải phân cách ngày giữa các tin nhắn (Hôm nay / Hôm qua / dd/mm/yyyy). */
export function formatDateDivider(ts: number): string {
  const d = new Date(ts * 1000);
  const today = new Date();
  if (isSameDay(d, today)) return "Hôm nay";
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (isSameDay(d, yesterday)) return "Hôm qua";
  return dateFmt.format(d);
}

export function isNewDay(prevTs?: number | null, ts?: number | null): boolean {
  if (!prevTs || !ts) return false;
  return !isSameDay(new Date(prevTs * 1000), new Date(ts * 1000));
}
