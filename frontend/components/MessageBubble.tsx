import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Message } from "@/lib/types";
import ChartRenderer from "./ChartRenderer";

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-blue-600 text-white" : "bg-zinc-200 text-zinc-700"
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "rounded-tr-sm bg-blue-600 text-white"
            : "rounded-tl-sm bg-zinc-100 text-zinc-900"
        }`}
      >
        <div
          className={`prose prose-sm max-w-none ${isUser ? "prose-invert" : ""} prose-p:my-1 prose-img:rounded-lg`}
        >
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {!isUser && <ChartRenderer data={message.chart} />}
      </div>
    </div>
  );
}
