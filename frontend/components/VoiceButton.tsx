"use client";

import { Mic, Square } from "lucide-react";
import { useRef, useState } from "react";

export default function VoiceButton({
  onRecorded,
  disabled,
}: {
  onRecorded: (blob: Blob) => void;
  disabled: boolean;
}) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        onRecorded(new Blob(chunksRef.current, { type: "audio/webm" }));
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch {
      // Từ chối quyền mic hoặc trình duyệt không hỗ trợ — im lặng bỏ qua, nút quay lại trạng thái
      // chưa ghi âm (không có hệ thống toast/notification trong app này để báo lỗi đẹp hơn).
      setRecording(false);
    }
  }

  function stop() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <button
      type="button"
      onClick={recording ? stop : start}
      disabled={disabled}
      aria-label={recording ? "Dừng ghi âm" : "Ghi âm câu hỏi bằng giọng nói"}
      title={recording ? "Dừng ghi âm" : "Ghi âm câu hỏi bằng giọng nói"}
      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition-colors disabled:bg-line disabled:text-ink-soft ${
        recording
          ? "animate-pulse bg-red-600 text-white"
          : "text-ink-soft hover:bg-line hover:text-ink"
      }`}
    >
      {recording ? <Square size={16} /> : <Mic size={18} />}
    </button>
  );
}
