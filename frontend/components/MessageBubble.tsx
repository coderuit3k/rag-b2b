import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { formatClock } from "@/lib/format";
import type { Message } from "@/lib/types";
import ChartRenderer from "./ChartRenderer";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`message-rise-in flex gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-ink text-white" : "bg-brand text-white"
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "rounded-tr-sm bg-brand text-white"
            : "rounded-tl-sm border border-line bg-panel text-ink"
        }`}
      >
        <div
          className={`prose prose-sm max-w-none ${isUser ? "prose-invert" : ""} prose-p:my-1 prose-img:rounded-lg`}
        >
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {!isUser && <ChartRenderer data={message.chart} />}
        {message.ts && (
          <p className={`mt-1 text-right text-[10px] ${isUser ? "text-white/70" : "text-ink-soft"}`}>
            {formatClock(message.ts)}
          </p>
        )}
      </div>
    </div>
  );
}
