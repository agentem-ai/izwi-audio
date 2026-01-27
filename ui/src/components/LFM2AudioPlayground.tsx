import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AudioWaveform,
  Mic,
  MicOff,
  Download,
  RotateCcw,
  Loader2,
  Send,
  MessageSquare,
} from "lucide-react";
import { api } from "../api";
import clsx from "clsx";

interface Message {
  id: string;
  role: "user" | "assistant";
  text?: string;
  audioUrl?: string;
  audioBlob?: Blob;
}

export function LFM2AudioPlayground() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        await processAudioInput(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError("Could not access microphone. Please grant permission.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const processAudioInput = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError(null);

    const userAudioUrl = URL.createObjectURL(audioBlob);
    const userMessageId = `user-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        role: "user",
        audioUrl: userAudioUrl,
        audioBlob: audioBlob,
      },
    ]);

    try {
      const reader = new FileReader();
      const audioBase64 = await new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(",")[1];
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
      });

      const response = await api.lfm2AudioChat({
        audio_base64: audioBase64,
      });

      const assistantMessageId = `assistant-${Date.now()}`;
      let assistantAudioUrl: string | undefined;

      if (response.audio_base64) {
        const binaryString = atob(response.audio_base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const audioBlob = new Blob([bytes], { type: "audio/wav" });
        assistantAudioUrl = URL.createObjectURL(audioBlob);
      }

      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          role: "assistant",
          text: response.text,
          audioUrl: assistantAudioUrl,
        },
      ]);

      if (assistantAudioUrl && audioRef.current) {
        audioRef.current.src = assistantAudioUrl;
        audioRef.current.play();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process audio");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTextSubmit = async () => {
    if (!textInput.trim() || isProcessing) return;

    setIsProcessing(true);
    setError(null);

    const userMessageId = `user-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        role: "user",
        text: textInput,
      },
    ]);

    const inputText = textInput;
    setTextInput("");

    try {
      const response = await api.lfm2AudioChat({
        text: inputText,
      });

      const assistantMessageId = `assistant-${Date.now()}`;
      let assistantAudioUrl: string | undefined;

      if (response.audio_base64) {
        const binaryString = atob(response.audio_base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const audioBlob = new Blob([bytes], { type: "audio/wav" });
        assistantAudioUrl = URL.createObjectURL(audioBlob);
      }

      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          role: "assistant",
          text: response.text,
          audioUrl: assistantAudioUrl,
        },
      ]);

      if (assistantAudioUrl && audioRef.current) {
        audioRef.current.src = assistantAudioUrl;
        audioRef.current.play();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    messages.forEach((msg) => {
      if (msg.audioUrl) {
        URL.revokeObjectURL(msg.audioUrl);
      }
    });
    setMessages([]);
    setError(null);
    setTextInput("");
  };

  const handleDownloadAudio = (audioUrl: string, index: number) => {
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `lfm2-audio-${index}-${Date.now()}.wav`;
    a.click();
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/20">
            <AudioWaveform className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Audio Chat</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Talk with LFM2-Audio using voice or text
            </p>
          </div>
        </div>

        {messages.length > 0 && (
          <button onClick={handleReset} className="btn btn-ghost text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="space-y-3 mb-4 max-h-96 overflow-y-auto">
        {messages.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <AudioWaveform className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Start a conversation</p>
            <p className="text-xs mt-1">Record audio or type a message</p>
          </div>
        )}

        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={clsx(
                "p-3 rounded-lg",
                message.role === "user"
                  ? "bg-blue-500/10 border border-blue-500/20 ml-8"
                  : "bg-[#1a1a1a] border border-[#2a2a2a] mr-8",
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <span
                    className={clsx(
                      "text-[10px] font-medium uppercase tracking-wider",
                      message.role === "user"
                        ? "text-blue-400"
                        : "text-purple-400",
                    )}
                  >
                    {message.role === "user" ? "You" : "LFM2-Audio"}
                  </span>

                  {message.text && (
                    <p className="text-sm text-gray-300 mt-1">{message.text}</p>
                  )}

                  {message.audioUrl && (
                    <div className="mt-2">
                      <audio
                        src={message.audioUrl}
                        controls
                        className="w-full h-8"
                      />
                    </div>
                  )}
                </div>

                {message.audioUrl && (
                  <button
                    onClick={() =>
                      handleDownloadAudio(message.audioUrl!, index)
                    }
                    className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-gray-400 text-sm p-3"
          >
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </motion.div>
        )}
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs mb-4"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input area */}
      <div className="space-y-3">
        {/* Text input */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <MessageSquare className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleTextSubmit()}
              placeholder="Type a message..."
              disabled={isProcessing || isRecording}
              className="input pl-10 text-sm"
            />
          </div>
          <button
            onClick={handleTextSubmit}
            disabled={!textInput.trim() || isProcessing || isRecording}
            className="btn btn-primary px-4"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Audio controls */}
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            className={clsx(
              "p-4 rounded-full transition-all",
              isRecording
                ? "bg-red-500 hover:bg-red-600 animate-pulse"
                : "bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2a2a2a]",
            )}
          >
            {isRecording ? (
              <MicOff className="w-6 h-6 text-white" />
            ) : (
              <Mic className="w-6 h-6 text-gray-300" />
            )}
          </button>

          {isRecording && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-2 text-red-400 text-sm"
            >
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Recording...
            </motion.div>
          )}
        </div>

        <p className="text-center text-xs text-gray-500">
          {isRecording
            ? "Click the microphone to stop recording"
            : "Click the microphone to start recording, or type a message"}
        </p>
      </div>

      {/* Hidden audio element for playback */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
