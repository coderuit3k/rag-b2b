import { auth } from "@clerk/nextjs/server";
import ChatApp from "@/components/ChatApp";

// Resource-based check TRỰC TIẾP trên trang (khuyến nghị hiện tại của Clerk, thay cho chỉ dựa vào
// proxy.ts::createRouteMatcher — Clerk đã cảnh báo deprecated: matcher theo path có thể lệch với
// cách Next.js thực sự route request, để lọt route được bảo vệ). Giữ CẢ HAI (proxy.ts vẫn cần —
// clerkMiddleware() là điều kiện bắt buộc để auth() hoạt động) — lớp này đảm bảo chắc chắn trang
// chat không bao giờ lộ ra ngoài dù matcher ở proxy.ts có lỗ hổng gì.
export default async function Page() {
  await auth.protect();
  return <ChatApp />;
}
