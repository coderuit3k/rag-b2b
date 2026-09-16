"use client";

import { useUser } from "@clerk/nextjs";
import { useCallback, useEffect, useState } from "react";
import ChatHeader from "./ChatHeader";
import ChatInput from "./ChatInput";
import ChatWindow from "./ChatWindow";
import CrmConfirmCard from "./CrmConfirmCard";
import EmailConfirmCard from "./EmailConfirmCard";
import Sidebar from "./Sidebar";
import {
  confirmCrmUpdate,
  confirmEmail,
  createConversation,
  deleteConversation,
  fetchConversations,
  renameConversation,
  sendChat,
  sendVoiceChat,
} from "@/lib/api";
import type { Conversations, ForceRoute } from "@/lib/types";

type PendingEmail = { question: string; preview: string; recipient: string };
type PendingCrm = { customerId: string; status: string; preview: string };

export default function ChatApp() {
  const { user, isLoaded } = useUser();
  const [conversations, setConversations] = useState<Conversations>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [forceRoute, setForceRoute] = useState<ForceRoute>("");
  const [pendingEmail, setPendingEmail] = useState<PendingEmail | null>(null);
  const [emailPending, setEmailPending] = useState(false);
  const [pendingCrm, setPendingCrm] = useState<PendingCrm | null>(null);
  const [crmPending, setCrmPending] = useState(false);

  const startNewChat = useCallback(async (userId: string) => {
    const { conv_id, conversation } = await createConversation(userId);
    setConversations((prev) => ({ ...prev, [conv_id]: conversation }));
    setActiveId(conv_id);
  }, []);

  useEffect(() => {
    if (!user) return;
    fetchConversations(user.id).then((convs) => {
      setConversations(convs);
      const ids = Object.keys(convs);
      if (ids.length > 0) {
        setActiveId(ids[ids.length - 1]);
      } else {
        startNewChat(user.id);
      }
    });
  }, [user, startNewChat]);

  // Poll nhẹ (Cảnh báo Chủ động, watcher.py) — job cron ngoài tiến trình này viết thẳng cảnh báo
  // vào chat_store (Postgres), không có kênh push/WebSocket nào báo cho tab đang mở biết ngay -
  // phải tự hỏi lại định kỳ mới thấy cảnh báo mà KHÔNG cần người dùng bấm gì/tải lại trang. Bỏ qua
  // lúc đang gửi tin (`loading`) để tránh đè optimistic update giữa chừng.
  useEffect(() => {
    if (!user) return;
    const id = setInterval(() => {
      if (loading) return;
      fetchConversations(user.id).then(setConversations);
    }, 20000);
    return () => clearInterval(id);
  }, [user, loading]);

  async function handleSend(question: string) {
    if (!user || !activeId) return;
    setPendingEmail(null);
    setPendingCrm(null);
    setConversations((prev) => ({
      ...prev,
      [activeId]: { ...prev[activeId], messages: [...prev[activeId].messages, { role: "user", content: question }] },
    }));
    setLoading(true);
    try {
      const role = typeof user.publicMetadata?.role === "string" ? user.publicMetadata.role : null;
      const res = await sendChat({ userId: user.id, convId: activeId, question, forceTopic: forceRoute, role });
      if (res.needs_email_confirm) {
        setPendingEmail({ question: res.question, preview: res.preview, recipient: res.recipient });
        setConversations((prev) => ({ ...prev, [activeId]: { ...prev[activeId], title: res.title } }));
      } else if (res.needs_crm_confirm) {
        setPendingCrm({ customerId: res.customer_id, status: res.status, preview: res.preview });
        setConversations((prev) => ({ ...prev, [activeId]: { ...prev[activeId], title: res.title } }));
      } else {
        setConversations((prev) => ({
          ...prev,
          [activeId]: {
            title: res.title,
            messages: [...prev[activeId].messages, { role: "assistant", content: res.answer, chart: res.chart }],
          },
        }));
      }
    } catch (e) {
      setConversations((prev) => ({
        ...prev,
        [activeId]: {
          ...prev[activeId],
          messages: [
            ...prev[activeId].messages,
            { role: "assistant", content: `Xin lỗi, có lỗi xảy ra: ${(e as Error).message}` },
          ],
        },
      }));
    } finally {
      setLoading(false);
    }
  }

  // Tin nhắn thoại (Voice AI, 2026-09-15): khác handleSend() ở chỗ câu hỏi CHƯA BIẾT trước khi gọi
  // API (Whisper phiên âm ở backend) — nên chỉ thêm tin nhắn "user" vào sau khi có kết quả, không
  // optimistic-update như văn bản gõ tay.
  async function handleSendVoice(audioBlob: Blob) {
    if (!user || !activeId) return;
    setPendingEmail(null);
    setPendingCrm(null);
    setLoading(true);
    try {
      const role = typeof user.publicMetadata?.role === "string" ? user.publicMetadata.role : null;
      const res = await sendVoiceChat({ userId: user.id, convId: activeId, audioBlob, role });
      if (res.needs_email_confirm) {
        setConversations((prev) => ({
          ...prev,
          [activeId]: {
            title: res.title,
            messages: [...prev[activeId].messages, { role: "user", content: res.question }],
          },
        }));
        setPendingEmail({ question: res.question, preview: res.preview, recipient: res.recipient });
      } else if (res.needs_crm_confirm) {
        setConversations((prev) => ({
          ...prev,
          [activeId]: {
            title: res.title,
            messages: [...prev[activeId].messages, { role: "user", content: res.question }],
          },
        }));
        setPendingCrm({ customerId: res.customer_id, status: res.status, preview: res.preview });
      } else {
        setConversations((prev) => ({
          ...prev,
          [activeId]: {
            title: res.title,
            messages: [
              ...prev[activeId].messages,
              { role: "user", content: res.question },
              { role: "assistant", content: res.answer, chart: res.chart },
            ],
          },
        }));
        if (res.audio_base64) {
          new Audio(`data:audio/mpeg;base64,${res.audio_base64}`).play().catch(() => {});
        }
      }
    } catch (e) {
      setConversations((prev) => ({
        ...prev,
        [activeId]: {
          ...prev[activeId],
          messages: [
            ...prev[activeId].messages,
            { role: "assistant", content: `Xin lỗi, có lỗi xảy ra: ${(e as Error).message}` },
          ],
        },
      }));
    } finally {
      setLoading(false);
    }
  }

  const handleRename = useCallback(
    async (convId: string, title: string) => {
      if (!user) return;
      try {
        const { title: saved } = await renameConversation({ userId: user.id, convId, title });
        setConversations((prev) => ({ ...prev, [convId]: { ...prev[convId], title: saved } }));
      } catch {
        // lỗi mạng — bỏ qua, tiêu đề cũ vẫn hiển thị nguyên (app này không có hệ thống toast)
      }
    },
    [user]
  );

  const handleDelete = useCallback(
    async (convId: string) => {
      if (!user) return;
      try {
        await deleteConversation({ userId: user.id, convId });
      } catch {
        return; // lỗi mạng — không xoá khỏi state nếu backend chưa xoá thành công
      }
      setConversations((prev) => {
        const next = { ...prev };
        delete next[convId];
        return next;
      });
      if (activeId !== convId) return;
      const remaining = Object.keys(conversations).filter((id) => id !== convId);
      if (remaining.length > 0) {
        setActiveId(remaining[remaining.length - 1]);
      } else {
        startNewChat(user.id);
      }
    },
    [user, activeId, conversations, startNewChat]
  );

  async function handleEmailDecision(send: boolean) {
    if (!user || !activeId || !pendingEmail) return;
    setEmailPending(true);
    try {
      const { result } = await confirmEmail({
        userId: user.id,
        convId: activeId,
        question: pendingEmail.question,
        answer: pendingEmail.preview,
        send,
      });
      setConversations((prev) => ({
        ...prev,
        [activeId]: { ...prev[activeId], messages: [...prev[activeId].messages, { role: "assistant", content: result }] },
      }));
      setPendingEmail(null);
    } finally {
      setEmailPending(false);
    }
  }

  async function handleCrmDecision(send: boolean) {
    if (!user || !activeId || !pendingCrm) return;
    setCrmPending(true);
    try {
      const { result } = await confirmCrmUpdate({
        userId: user.id,
        convId: activeId,
        customerId: pendingCrm.customerId,
        status: pendingCrm.status,
        send,
      });
      setConversations((prev) => ({
        ...prev,
        [activeId]: { ...prev[activeId], messages: [...prev[activeId].messages, { role: "assistant", content: result }] },
      }));
      setPendingCrm(null);
    } finally {
      setCrmPending(false);
    }
  }

  if (!isLoaded) {
    return <div className="flex h-dvh items-center justify-center bg-surface text-sm text-ink-soft">Đang tải...</div>;
  }

  const activeConversation = activeId ? conversations[activeId] : undefined;

  return (
    <div className="flex h-dvh">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNewChat={() => user && startNewChat(user.id)}
        onRename={handleRename}
        onDelete={handleDelete}
      />
      <div className="flex flex-1 flex-col">
        <ChatHeader
          conversationTitle={activeConversation?.title}
          forceRoute={forceRoute}
          onForceRouteChange={setForceRoute}
        />
        <ChatWindow
          messages={activeConversation?.messages ?? []}
          loading={loading}
          pendingEmailNode={
            pendingEmail && (
              <EmailConfirmCard
                recipient={pendingEmail.recipient}
                preview={pendingEmail.preview}
                pending={emailPending}
                onConfirm={() => handleEmailDecision(true)}
                onCancel={() => handleEmailDecision(false)}
              />
            )
          }
          pendingCrmNode={
            pendingCrm && (
              <CrmConfirmCard
                preview={pendingCrm.preview}
                pending={crmPending}
                onConfirm={() => handleCrmDecision(true)}
                onCancel={() => handleCrmDecision(false)}
              />
            )
          }
        />
        <ChatInput
          onSend={handleSend}
          onSendVoice={handleSendVoice}
          disabled={loading || !activeId || !!pendingEmail || !!pendingCrm}
        />
      </div>
    </div>
  );
}
