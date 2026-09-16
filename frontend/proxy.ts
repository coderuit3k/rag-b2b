// Next.js 16 đổi tên file "middleware.ts" -> "proxy.ts" (hành vi/API giữ nguyên, chỉ đổi tên file +
// tên hàm export — xem node_modules/next/dist/docs/01-app/03-api-reference/03-file-conventions/proxy.md).
// Khoá TOÀN BỘ app: chỉ /sign-in và /sign-up công khai, còn lại bắt buộc đăng nhập Clerk.
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isPublicRoute = createRouteMatcher(["/sign-in(.*)", "/sign-up(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  if (!isPublicRoute(req)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // Bỏ qua file tĩnh/Next internals, áp dụng cho mọi route còn lại + luôn chạy cho /api.
    "/((?!_next|.*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
