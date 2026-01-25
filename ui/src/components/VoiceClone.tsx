import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Mic, Square, Play, Check, X } from "lucide-react";
import clsx from "clsx";

interface VoiceCloneProps {
  onVoiceCloneReady: (audioBase64: string, transcript: string) => void;
  onClear: () => void;
}

export function VoiceClone({ onVoiceCloneReady, onClear }: VoiceCloneProps) {
  const [mode, setMode] = useState<"upload" | "record" | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("audio/")) {
      setError("Please upload an audio file");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB");
      return;
    }

    setError(null);
    setAudioBlob(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setMode("upload");
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setMode("record");
      setError(null);
    } catch (err) {
      setError("Microphone access denied. Please allow microphone access.");
      console.error("Recording error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handlePlay = () => {
    audioRef.current?.play();
  };

  const handleClear = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioBlob(null);
    setAudioUrl(null);
    setTranscript("");
    setMode(null);
    setError(null);
    onClear();
  };

  const handleConfirm = async () => {
    if (!audioBlob || !transcript.trim()) {
      setError("Please provide both audio and transcript");
      return;
    }

    try {
      // Convert blob to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        const base64Audio = base64.split(",")[1];
        onVoiceCloneReady(base64Audio, transcript.trim());
      };
      reader.readAsDataURL(audioBlob);
    } catch (err) {
      setError("Failed to process audio");
      console.error(err);
    }
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-white">Voice Cloning</h3>
          <p className="text-xs text-gray-600 mt-0.5">
            Upload or record a voice sample
          </p>
        </div>
        {audioBlob && (
          <button onClick={handleClear} className="btn btn-ghost text-xs">
            <X className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {!audioBlob ? (
        <div className="grid grid-cols-2 gap-2">
          {/* Upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center gap-2 p-4 rounded-lg border border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a] transition-colors"
          >
            <Upload className="w-5 h-5 text-gray-400" />
            <span className="text-xs text-gray-400">Upload Audio</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* Record button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={clsx(
              "flex flex-col items-center gap-2 p-4 rounded-lg border transition-colors",
              isRecording
                ? "border-red-500/50 bg-red-950/20"
                : "border-[#2a2a2a] bg-[#161616] hover:bg-[#1a1a1a]",
            )}
          >
            {isRecording ? (
              <>
                <Square className="w-5 h-5 text-red-400" />
                <span className="text-xs text-red-400">Stop Recording</span>
              </>
            ) : (
              <>
                <Mic className="w-5 h-5 text-gray-400" />
                <span className="text-xs text-gray-400">Record Voice</span>
              </>
            )}
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Audio player */}
          <div className="p-3 rounded-lg bg-[#161616] border border-[#2a2a2a]">
            <div className="flex items-center gap-2 mb-2">
              <button
                onClick={handlePlay}
                className="p-1.5 rounded bg-[#1f1f1f] hover:bg-[#2a2a2a]"
              >
                <Play className="w-3.5 h-3.5 text-white" />
              </button>
              <div className="flex-1 text-xs text-gray-500">
                {mode === "upload" ? "Uploaded audio" : "Recorded audio"}
              </div>
              <div className="text-xs text-gray-600">
                {(audioBlob.size / 1024).toFixed(0)} KB
              </div>
            </div>
            <audio
              ref={audioRef}
              src={audioUrl || ""}
              className="w-full h-8"
              controls
            />
          </div>

          {/* Transcript input */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">
              Transcript
              <span className="text-red-400 ml-1">*</span>
            </label>
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="Enter what is said in the audio..."
              rows={3}
              className="textarea text-sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Provide an accurate transcript of the audio for best results
            </p>
          </div>

          {/* Confirm button */}
          <button
            onClick={handleConfirm}
            disabled={!transcript.trim()}
            className="btn btn-primary w-full text-sm"
          >
            <Check className="w-4 h-4" />
            Use This Voice
          </button>
        </div>
      )}
    </div>
  );
}
