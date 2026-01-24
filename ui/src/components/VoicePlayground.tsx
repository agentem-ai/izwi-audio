import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Square,
  Download,
  RotateCcw,
  Sparkles,
  ChevronDown,
  Volume2,
  Clock,
  Zap,
} from "lucide-react";
import { api } from "../api";
import { Waveform } from "./ui/Waveform";
import { VoiceOrb } from "./ui/VoiceOrb";
import clsx from "clsx";

interface VoicePlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

const SPEAKERS = [
  { id: "Vivian", name: "Vivian", gender: "Female", language: "Multi" },
  { id: "Serena", name: "Serena", gender: "Female", language: "Multi" },
  { id: "Ryan", name: "Ryan", gender: "Male", language: "Multi" },
  { id: "Aiden", name: "Aiden", gender: "Male", language: "Multi" },
  { id: "Dylan", name: "Dylan", gender: "Male", language: "Multi" },
  { id: "Eric", name: "Eric", gender: "Male", language: "Multi" },
  { id: "Sohee", name: "Sohee", gender: "Female", language: "Korean" },
  { id: "Ono_anna", name: "Anna", gender: "Female", language: "Japanese" },
  { id: "Uncle_fu", name: "Uncle Fu", gender: "Male", language: "Chinese" },
];

const SAMPLE_TEXTS = [
  {
    text: "Hello! Welcome to Izwi, your personal AI voice playground. Let's create something amazing together.",
    label: "Welcome",
  },
  {
    text: "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    label: "Pangram",
  },
  {
    text: "In a world where technology evolves rapidly, the ability to generate natural-sounding speech has become increasingly important for accessibility and communication.",
    label: "Tech",
  },
  {
    text: "Once upon a time, in a land far away, there lived a curious inventor who dreamed of giving machines the gift of voice.",
    label: "Story",
  },
];

type GenerationStatus = "idle" | "generating" | "complete" | "error";

export function VoicePlayground({
  selectedModel,
  onModelRequired,
}: VoicePlaygroundProps) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("Vivian");
  const [showSpeakerSelect, setShowSpeakerSelect] = useState(false);
  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [generationTime, setGenerationTime] = useState<number | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const handleLoadedMetadata = () => setAudioDuration(audio.duration);

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [audioUrl]);

  const handleGenerate = async () => {
    if (!selectedModel) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      setError("Please enter some text to generate speech");
      return;
    }

    try {
      setStatus("generating");
      setError(null);
      setGenerationTime(null);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const startTime = performance.now();
      const blob = await api.generateTTS({
        text: text.trim(),
        speaker,
      });
      const endTime = performance.now();

      setGenerationTime((endTime - startTime) / 1000);

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setStatus("complete");

      setTimeout(() => {
        audioRef.current?.play();
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
      setStatus("error");
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
      a.download = `izwi-${speaker.toLowerCase()}-${Date.now()}.wav`;
      a.click();
    }
  };

  const handleReset = () => {
    setText("");
    setStatus("idle");
    setError(null);
    setGenerationTime(null);
    setAudioDuration(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    textareaRef.current?.focus();
  };

  const selectedSpeaker = SPEAKERS.find((s) => s.id === speaker) || SPEAKERS[0];

  return (
    <div className="glass-card p-6 lg:p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">
              Voice Playground
            </h2>
            <p className="text-sm text-gray-500">
              Generate natural speech from text
            </p>
          </div>
        </div>

        {/* Speaker selector */}
        <div className="relative">
          <button
            onClick={() => setShowSpeakerSelect(!showSpeakerSelect)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.05] border border-white/[0.1] hover:bg-white/[0.08] transition-colors"
          >
            <Volume2 className="w-4 h-4 text-indigo-400" />
            <span className="text-sm text-white">{selectedSpeaker.name}</span>
            <ChevronDown
              className={clsx(
                "w-4 h-4 text-gray-400 transition-transform",
                showSpeakerSelect && "rotate-180",
              )}
            />
          </button>

          <AnimatePresence>
            {showSpeakerSelect && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute right-0 mt-2 w-56 p-2 rounded-xl bg-gray-900 border border-white/[0.1] shadow-xl z-50"
              >
                {SPEAKERS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSpeaker(s.id);
                      setShowSpeakerSelect(false);
                    }}
                    className={clsx(
                      "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors",
                      speaker === s.id
                        ? "bg-indigo-500/20 text-white"
                        : "hover:bg-white/[0.05] text-gray-300",
                    )}
                  >
                    <div
                      className={clsx(
                        "w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium",
                        s.gender === "Female"
                          ? "bg-pink-500/20 text-pink-400"
                          : "bg-blue-500/20 text-blue-400",
                      )}
                    >
                      {s.name[0]}
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium">{s.name}</div>
                      <div className="text-xs text-gray-500">{s.language}</div>
                    </div>
                    {speaker === s.id && (
                      <div className="w-2 h-2 rounded-full bg-indigo-400" />
                    )}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Main content */}
      <div className="grid lg:grid-cols-[1fr,auto] gap-6">
        {/* Text input area */}
        <div className="space-y-4">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste your text here..."
              rows={6}
              disabled={status === "generating"}
              className="textarea text-base leading-relaxed"
            />
            <div className="absolute bottom-3 right-3 flex items-center gap-2">
              <span
                className={clsx(
                  "text-xs",
                  text.length > 500 ? "text-amber-400" : "text-gray-500",
                )}
              >
                {text.length} / 500
              </span>
            </div>
          </div>

          {/* Sample texts */}
          <div className="flex flex-wrap gap-2">
            {SAMPLE_TEXTS.map((sample, i) => (
              <button
                key={i}
                onClick={() => setText(sample.text)}
                className="text-xs px-3 py-1.5 rounded-full bg-white/[0.05] hover:bg-white/[0.1] text-gray-400 hover:text-white transition-colors border border-white/[0.05]"
              >
                {sample.label}
              </button>
            ))}
          </div>

          {/* Error message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleGenerate}
              disabled={status === "generating" || !selectedModel}
              className={clsx(
                "btn flex-1 sm:flex-none",
                status === "generating"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                  : "btn-primary",
              )}
            >
              {status === "generating" ? (
                <>
                  <motion.div
                    className="w-4 h-4 border-2 border-indigo-300 border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  />
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Generate Speech
                </>
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

          {/* No model warning */}
          {!selectedModel && (
            <p className="text-sm text-amber-400/80">
              ⚠️ Please load a model first to generate speech
            </p>
          )}
        </div>

        {/* Voice orb visualization */}
        <div className="hidden lg:flex flex-col items-center justify-center px-8">
          <VoiceOrb
            isActive={isPlaying}
            isGenerating={status === "generating"}
            size="lg"
          />
          {status === "generating" && (
            <p className="mt-4 text-sm text-gray-400 animate-pulse">
              Synthesizing voice...
            </p>
          )}
        </div>
      </div>

      {/* Audio player */}
      <AnimatePresence>
        {audioUrl && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="mt-6 p-4 rounded-xl bg-white/[0.03] border border-white/[0.08]"
          >
            {/* Waveform visualization */}
            <Waveform isPlaying={isPlaying} className="mb-4" />

            {/* Audio element */}
            <audio
              ref={audioRef}
              src={audioUrl}
              className="w-full h-10 opacity-80"
              controls
            />

            {/* Stats */}
            <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
              {generationTime && (
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  Generated in {generationTime.toFixed(1)}s
                </div>
              )}
              {audioDuration && (
                <div className="flex items-center gap-1">
                  <Volume2 className="w-3 h-3" />
                  {audioDuration.toFixed(1)}s audio
                </div>
              )}
              {generationTime && audioDuration && (
                <div className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  {(audioDuration / generationTime).toFixed(2)}x realtime
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
