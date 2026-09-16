# RAG(B2B) — Frontend (Next.js)

Giao diện chat kiểu Telegram/Zalo cho hệ thống RAG(B2B), thay thế/song song với app Streamlit
(`../src/rag_b2b/app.py`). Next.js 16 (App Router) + Tailwind CSS v4 + Clerk (auth) + recharts +
react-markdown + lucide-react.

## Kiến trúc

```
Next.js (frontend/, port 3000)  --HTTP-->  FastAPI (rag_b2b.api, port 8000)  -->  pipeline.py (Python)
       |
       +-- Clerk (đăng nhập, khoá toàn bộ app)
```

Next.js/React không gọi thẳng được hàm Python trong pipeline hiện có (nó chạy trong tiến trình
Streamlit) — `src/rag_b2b/api.py` (FastAPI, ở thư mục gốc repo) là cầu nối HTTP, dùng lại NGUYÊN
`pipeline.py`/`chat_store.py` đã có, không viết lại logic nghiệp vụ.

## Chạy thử (2 bước)

**1. Backend** (từ thư mục gốc repo, không phải `frontend/`):

```bash
cd ..
source .venv/bin/activate   # hoặc .venv/bin/uvicorn trực tiếp
uvicorn rag_b2b.api:app --reload --port 8000
```

**2. Frontend:**

```bash
cp .env.local.example .env.local   # rồi điền key thật (xem bên dưới)
npm install
npm run dev
```

Mở http://localhost:3000.

## Cần điền vào `.env.local`

| Biến | Lấy ở đâu |
|---|---|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | Tạo project tại https://dashboard.clerk.com → API Keys. Chưa điền key thật → app báo lỗi "Publishable key not valid" ngay khi mở (đã kiểm chứng, đây là hành vi ĐÚNG, không phải bug). |
| `NEXT_PUBLIC_API_URL` | URL của FastAPI backend ở bước 1 (mặc định `http://localhost:8000`, đổi khi deploy). |

## Cấu trúc chính

- `proxy.ts` — khoá toàn bộ app bằng Clerk (Next.js 16 đổi tên quy ước file từ `middleware.ts` ->
  `proxy.ts`, xem comment trong file). Chỉ `/sign-in`, `/sign-up` công khai.
- `app/page.tsx` — Server Component, gọi `auth.protect()` trực tiếp trên trang (resource-based
  check, khuyến nghị hiện tại của Clerk — bổ sung THÊM cho `proxy.ts`, không thay thế, vì Clerk đã
  đánh dấu deprecated việc chỉ dựa vào middleware path-matching).
- `components/ChatApp.tsx` — component chính (Client), quản lý state hội thoại + gọi API.
- `components/Sidebar.tsx` — danh sách hội thoại + nút hội thoại mới + `<UserButton/>`.
- `components/ChatWindow.tsx`, `MessageBubble.tsx`, `ChartRenderer.tsx` — luồng tin nhắn, markdown,
  biểu đồ Cột/Tròn/Đường (recharts) khi nhánh Graph trả về dữ liệu dạng bảng.
- `components/ChatInput.tsx` — textarea tự giãn dòng + dropdown "Force Route".
- `components/EmailConfirmCard.tsx` — xem "Về nhánh email" bên dưới.

## Phân quyền theo vai trò (RBAC, 2026-09-15)

Câu hỏi mơ hồ (không mã khách, không sản phẩm cụ thể) được thiên vị chọn nhánh theo vai trò của
người hỏi — vd nhân viên Sale hỏi chung chung sẽ nghiêng về nhánh `leads` (tìm khách tiềm năng) thay
vì `predict`; CEO hỏi chung chung về tình hình kinh doanh sẽ nghiêng về `graph` (số liệu/biểu đồ
doanh thu vĩ mô toàn hệ thống). Câu hỏi đã RÕ RÀNG (có mã khách, tên sản phẩm...) thì route như cũ,
không bị vai trò ghi đè.

**Gán vai trò**: vào https://dashboard.clerk.com → chọn project → Users → chọn user → **Public
metadata** → thêm `{"role": "sales"}` hoặc `{"role": "ceo"}`. Không cần sửa code — frontend tự đọc
`user.publicMetadata.role` (xem `ChatApp.tsx`) và gửi kèm mỗi lần chat. Vai trò lạ/không set thì bỏ
qua, hành vi y hệt trước khi có RBAC.

**Giới hạn cần biết**:
- Đây là tuỳ biến TRẢI NGHIỆM, KHÔNG PHẢI ranh giới bảo mật — `role` là giá trị client tự gửi lên,
  chưa xác thực JWT (cùng giới hạn đã ghi trong `api.py`, xem docstring). Đừng dùng để chặn/lộ dữ
  liệu nhạy cảm dựa hoàn toàn vào giá trị này.
- Route đúng không đảm bảo TRẢ LỜI hay — đã test thật: role Sale với câu cực mơ hồ ("Tôi nên tiếp
  cận khách hàng nào hôm nay?") route đúng sang `leads`, nhưng bản thân công cụ `leads` cần TÊN sản
  phẩm/thương hiệu cụ thể để tìm khách nên vẫn trả "không tìm thấy sản phẩm phù hợp" — cần nêu rõ
  sản phẩm/thương hiệu trong câu hỏi mới ra kết quả hữu ích. Role CEO với câu mơ hồ tương tự thì
  hoạt động tốt (graph tự tính được số liệu toàn hệ thống không cần thực thể cụ thể nào).

## Về nhánh "email"

`FORCE_ROUTES` (lib/types.ts) **cố ý không có "email"**, và backend (`api.py`) chặn nó luôn ở kiểu
dữ liệu (Pydantic `Literal`) — ép route email từ dropdown sẽ gửi Gmail thật ngay lập tức, không qua
bước xác nhận nào. Nhưng người dùng vẫn có thể gõ tự nhiên "gửi email cho tôi ..." — trường hợp đó
API tự nhận ra ý định và trả về preview (`needs_email_confirm`) thay vì gửi ngay, `ChatApp.tsx` hiện
`EmailConfirmCard` với 2 nút Xác nhận/Huỷ — cùng mức an toàn với app Streamlit đã có, không tự động
gửi mail không qua xác nhận.

## Đã test

`next build` + `eslint` sạch, đăng nhập/đăng ký thật qua Clerk hoạt động đúng (kể cả redirect về
`/sign-in` tự viết trong app thay vì Account Portal mặc định — cần 2 biến
`NEXT_PUBLIC_CLERK_SIGN_IN_URL`/`SIGN_UP_URL`, xem `.env.local.example`), chat end-to-end qua API
thật đã kiểm.
