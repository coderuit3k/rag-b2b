import type { Conversations, ForceRoute } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    throw new Error(`API ${path} lỗi ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function fetchConversations(userId: string): Promise<Conversations> {
  return api(`/api/conversations?user_id=${encodeURIComponent(userId)}`);
}

export function createConversation(
  userId: string
): Promise<{ conv_id: string; conversation: Conversations[string] }> {
  return api(`/api/conversations?user_id=${encodeURIComponent(userId)}`, { method: "POST" });
}

export type ChatResult =
  | { needs_email_confirm: true; needs_crm_confirm?: false; question: string; preview: string; recipient: string; title: string }
  | { needs_email_confirm?: false; needs_crm_confirm: true; customer_id: string; status: string; preview: string; title: string }
  | { needs_email_confirm?: false; needs_crm_confirm?: false; answer: string; chart: Record<string, string | number>[] | null; title: string };

export function sendChat(params: {
  userId: string;
  convId: string;
  question: string;
  forceTopic?: ForceRoute;
  role?: string | null;
}): Promise<ChatResult> {
  return api(`/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      user_id: params.userId,
      conv_id: params.convId,
      question: params.question,
      force_topic: params.forceTopic || null,
      role: params.role || null,
    }),
  });
}

export type VoiceChatResult =
  | { needs_email_confirm: true; needs_crm_confirm?: false; question: string; preview: string; recipient: string; title: string }
  | { needs_email_confirm?: false; needs_crm_confirm: true; question: string; customer_id: string; status: string; preview: string; title: string }
  | {
      needs_email_confirm?: false;
      needs_crm_confirm?: false;
      question: string;
      answer: string;
      chart: Record<string, string | number>[] | null;
      title: string;
      audio_base64: string | null;
    };

export async function sendVoiceChat(params: {
  userId: string;
  convId: string;
  audioBlob: Blob;
  role?: string | null;
}): Promise<VoiceChatResult> {
  // multipart/form-data — KHÔNG dùng helper api() ở trên vì nó luôn set Content-Type: application/json;
  // trình duyệt phải tự set boundary cho multipart, set tay sẽ sai định dạng.
  const form = new FormData();
  form.append("user_id", params.userId);
  form.append("conv_id", params.convId);
  if (params.role) form.append("role", params.role);
  form.append("audio", params.audioBlob, "voice.webm");
  const res = await fetch(`${API_URL}/api/voice-chat`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`API /api/voice-chat lỗi ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export function renameConversation(params: {
  userId: string;
  convId: string;
  title: string;
}): Promise<{ conv_id: string; title: string }> {
  return api(`/api/conversations/${encodeURIComponent(params.convId)}/rename`, {
    method: "POST",
    body: JSON.stringify({ user_id: params.userId, title: params.title }),
  });
}

export function deleteConversation(params: { userId: string; convId: string }): Promise<{ conv_id: string }> {
  return api(
    `/api/conversations/${encodeURIComponent(params.convId)}?user_id=${encodeURIComponent(params.userId)}`,
    { method: "DELETE" }
  );
}

export function confirmEmail(params: {
  userId: string;
  convId: string;
  question: string;
  answer: string;
  send: boolean;
}): Promise<{ result: string }> {
  return api(`/api/chat/confirm-email`, {
    method: "POST",
    body: JSON.stringify({
      user_id: params.userId,
      conv_id: params.convId,
      question: params.question,
      answer: params.answer,
      send: params.send,
    }),
  });
}

export function confirmCrmUpdate(params: {
  userId: string;
  convId: string;
  customerId: string;
  status: string;
  send: boolean;
}): Promise<{ result: string }> {
  return api(`/api/chat/confirm-crm-update`, {
    method: "POST",
    body: JSON.stringify({
      user_id: params.userId,
      conv_id: params.convId,
      customer_id: params.customerId,
      status: params.status,
      send: params.send,
    }),
  });
}
