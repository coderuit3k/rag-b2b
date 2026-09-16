export type ChartRow = Record<string, string | number>;

export type Message = {
  role: "user" | "assistant";
  content: string;
  chart?: ChartRow[] | null;
};

export type Conversation = {
  title: string;
  messages: Message[];
};

export type Conversations = Record<string, Conversation>;

// Phải khớp _ROUTES trong src/rag_b2b/api.py. Cố ý KHÔNG có "email" — nhánh đó gửi mail thật ngay
// (send_result_email) không qua bước xác nhận nào ở giao diện này (khác app Streamlit đã có preview
// + nút xác nhận) — ép route email từ đây sẽ gửi mail ngoài ý muốn.
export const FORCE_ROUTES = [
  { value: "", label: "Tự động (Auto-Router)" },
  { value: "predict", label: "Predict — dự đoán sản phẩm khách sẽ mua" },
  { value: "leads", label: "Leads — tìm khách tiềm năng cho sản phẩm" },
  { value: "graph", label: "Graph — số liệu chính xác từ Neo4j" },
  { value: "vector", label: "Vector — chân dung/hành vi khách hàng" },
  { value: "web", label: "Web — tra cứu ngoài hệ thống" },
  { value: "multi_step", label: "Multi-step — phối hợp nhiều công cụ nối tiếp (Plan & Execute)" },
] as const;

export type ForceRoute = (typeof FORCE_ROUTES)[number]["value"];
