import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Volume2,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  Loader2,
} from "lucide-react";
import { api } from "../api";
import { LFM2_VOICES, SAMPLE_TEXTS } from "../types";
import clsx from "clsx";

export function LFM2TTSPlayground() {
  const [text, setText] = useState("");
  const [voice, setVoice] = useState<"us_male" | "us_female" | "uk_male" | "uk_female">("us_female");
  const [showVoiceSelect, setShowVoiceSelect] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const selectedVoice = LFM2_VOICES.find((v) => v.id === voice);

  const handleGenerate = async () => {
    if (!text.trim()) {
      setError("Please enter some text");
      return;
    }

    try {
      setGenerating(true);
      setError(null);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const blob = await api.lfm2GenerateTTS({
        text: text.trim(),
        voice: voice,
      });

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      setTimeout(() => {
        audioRef.current?.play();
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const handleDownload = () => {
    if (audioUrl) {
      const a = document.createElement("a");
      a.href = audioUrl;
      a.download = `lfm2-tts-${voice}-${Date.now()}.wav`;
      a.click();
    }
  };

  const handleReset = () => {
    setText("");
    setError(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    textareaRef.current?.focus();
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/20">
            <Volume2 className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">LFM2-Audio TTS</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Generate speech with Liquid AI's LFM2-Audio
            </p>
          </div>
        </div>

        {/* Voice selector */}
        <div className="relative">
          <button
            onClick={() => setShowVoiceSelect(!showVoiceSelect)}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#1f1f1f] text-sm"
          >
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-[10px] font-medium text-white">
              {selectedVoice?.name.charAt(0)}
            </div>
            <span className="text-white">{selectedVoice?.name || voice}</span>
            <ChevronDown
              className={clsx(
                "w-3.5 h-3.5 text-gray-500 transition-transform",
                showVoiceSelect && "rotate-180"
              )}
            />
          </button>

          <AnimatePresence>
            {showVoiceSelect && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="absolute right-0 mt-1 w-56 p-1 rounded bg-[#1a1a1a] border border-[#2a2a2a] shadow-xl z-50"
              >
                {LFM2_VOICES.map((v) => (
                  <button
                    key={v.id}
                    onClick={() => {
                      setVoice(v.id as typeof voice);
                      setShowVoiceSelect(false);
                    }}
                    className={clsx(
                      "w-full px-3 py-2 rounded text-left transition-colors flex items-center gap-3",
                      voice === v.id ? "bg-white/10" : "hover:bg-[#2a2a2a]"
                    )}
                  >
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-xs font-medium text-white flex-shrink-0">
                      {v.name.charAt(0)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={clsx(
                          "text-sm font-medium",
                          voice === v.id ? "text-white" : "text-gray-300"
                        )}
                      >
                        {v.name}
                      </div>
                      <div className="text-[10px] text-gray-500 truncate">
                        {v.description}
                      </div>
                    </div>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Text input */}
      <div className="space-y-3">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to synthesize..."
            rows={6}
            disabled={generating}
            className="textarea text-sm"
          />
          <div className="absolute bottom-2 right-2">
            <span
              className={clsx(
                "text-xs",
                text.length > 500 ? "text-red-400" : "text-gray-600"
              )}
            >
              {text.length}
            </span>
          </div>
        </div>

        {/* Sample texts */}
        <div className="flex flex-wrap gap-2">
          {SAMPLE_TEXTS.english.map((sample, i) => (
            <button
              key={i}
              onClick={() => setText(sample)}
              className="text-xs px-2 py-1 rounded bg-[#1a1a1a] hover:bg-[#1f1f1f] text-gray-500 hover:text-gray-300 border border-[#2a2a2a]"
            >
              Sample {i + 1}
            </button>
          ))}
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

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleGenerate}
            disabled={generating}
            className={clsx(
              "btn flex-1",
              generating ? "btn-secondary" : "btn-primary"
            )}
          >
            {generating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating...
              </>
            ) : (
              "Generate"
            )}
          </button>

          {audioUrl && (
            <>
              <button onClick={handleStop} className="btn btn-secondary">
                <Square className="w-4 h-4" />
              </button>
              <button onClick={handleDownload} className="btn btn-secondary">
                <Download className="w-4 h-4" />
              </button>
              <button onClick={handleReset} className="btn btn-ghost">
                <RotateCcw className="w-4 h-4" />
              </button>
            </>
          )}
        </div>

        <p className="text-xs text-gray-400">
          LFM2-Audio supports 4 voices: US Male, US Female, UK Male, UK Female
        </p>
      </div>

      {/* Audio player */}
      <AnimatePresence>
        {audioUrl && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="mt-4 p-3 rounded bg-[#1a1a1a] border border-[#2a2a2a]"
          >
            <audio ref={audioRef} src={audioUrl} className="w-full" controls />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
