import { motion, AnimatePresence } from "framer-motion";
import { Download, Play, Square, Check, Cpu, HardDrive } from "lucide-react";
import { ModelInfo } from "../api";
import { ProgressRing } from "./ui/ProgressRing";
import { MiniWaveform } from "./ui/Waveform";
import clsx from "clsx";

interface ModelManagerProps {
  models: ModelInfo[];
  selectedModel: string | null;
  onDownload: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onSelect: (variant: string) => void;
  downloadProgress: Record<string, number>;
}

const MODEL_INFO: Record<
  string,
  { name: string; description: string; size: string; badge?: string }
> = {
  "Qwen3-TTS-12Hz-0.6B-Base": {
    name: "Qwen3 0.6B Base",
    description: "Voice cloning with reference audio",
    size: "1.2 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    name: "Qwen3 0.6B Custom",
    description: "9 built-in voices, fast generation",
    size: "1.2 GB",
    badge: "Recommended",
  },
  "Qwen3-TTS-12Hz-1.7B-Base": {
    name: "Qwen3 1.7B Base",
    description: "Higher quality voice cloning",
    size: "3.4 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    name: "Qwen3 1.7B Custom",
    description: "9 voices, best quality",
    size: "3.4 GB",
    badge: "Best Quality",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    name: "Qwen3 1.7B Design",
    description: "Create voices from text descriptions",
    size: "3.4 GB",
    badge: "Creative",
  },
};

const formatBytes = (bytes: number | null): string => {
  if (bytes === null) return "";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)} MB`;
};

export function ModelManager({
  models,
  selectedModel,
  onDownload,
  onLoad,
  onUnload,
  onSelect,
  downloadProgress,
}: ModelManagerProps) {
  const ttsModels = models.filter((m) => !m.variant.includes("Tokenizer"));

  return (
    <div className="space-y-3">
      <AnimatePresence mode="popLayout">
        {ttsModels.map((model, index) => {
          const info = MODEL_INFO[model.variant] || {
            name: model.variant,
            description: "",
            size: formatBytes(model.size_bytes),
          };
          const isSelected = selectedModel === model.variant;
          const isDownloading = model.status === "downloading";
          const isLoading = model.status === "loading";
          const isReady = model.status === "ready";
          const isDownloaded = model.status === "downloaded";
          const progress =
            downloadProgress[model.variant] || model.download_progress || 0;

          return (
            <motion.div
              key={model.variant}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.05 }}
              className={clsx(
                "relative p-4 rounded-xl border transition-all duration-200 cursor-pointer group",
                isSelected
                  ? "bg-indigo-500/10 border-indigo-500/50 shadow-lg shadow-indigo-500/10"
                  : "bg-white/[0.02] border-white/[0.08] hover:bg-white/[0.05] hover:border-white/[0.15]",
              )}
              onClick={() => isReady && onSelect(model.variant)}
            >
              {/* Badge */}
              {info.badge && (
                <div className="absolute -top-2 right-3">
                  <span
                    className={clsx(
                      "text-[10px] font-semibold px-2 py-0.5 rounded-full",
                      info.badge === "Recommended" &&
                        "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
                      info.badge === "Best Quality" &&
                        "bg-purple-500/20 text-purple-400 border border-purple-500/30",
                      info.badge === "Creative" &&
                        "bg-amber-500/20 text-amber-400 border border-amber-500/30",
                    )}
                  >
                    {info.badge}
                  </span>
                </div>
              )}

              <div className="flex items-center gap-4">
                {/* Status indicator / Progress */}
                <div className="flex-shrink-0">
                  {isDownloading ? (
                    <ProgressRing
                      progress={progress}
                      size={44}
                      strokeWidth={3}
                    />
                  ) : isLoading ? (
                    <div className="w-11 h-11 rounded-full bg-indigo-500/20 flex items-center justify-center">
                      <motion.div
                        className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          ease: "linear",
                        }}
                      />
                    </div>
                  ) : isReady ? (
                    <div className="w-11 h-11 rounded-full bg-emerald-500/20 flex items-center justify-center">
                      {isSelected ? (
                        <MiniWaveform isActive />
                      ) : (
                        <Check className="w-5 h-5 text-emerald-400" />
                      )}
                    </div>
                  ) : isDownloaded ? (
                    <div className="w-11 h-11 rounded-full bg-blue-500/20 flex items-center justify-center">
                      <HardDrive className="w-5 h-5 text-blue-400" />
                    </div>
                  ) : (
                    <div className="w-11 h-11 rounded-full bg-white/[0.05] flex items-center justify-center">
                      <Cpu className="w-5 h-5 text-gray-500" />
                    </div>
                  )}
                </div>

                {/* Model info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-white truncate">
                      {info.name}
                    </h3>
                    {isSelected && (
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-indigo-500 text-white">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 truncate">
                    {info.description}
                  </p>

                  {/* Download progress text */}
                  {isDownloading && (
                    <p className="text-xs text-amber-400 mt-1">
                      Downloading... {progress.toFixed(0)}%
                    </p>
                  )}
                  {isLoading && (
                    <p className="text-xs text-indigo-400 mt-1">
                      Loading model into memory...
                    </p>
                  )}
                </div>

                {/* Size badge */}
                <div className="hidden sm:block text-xs text-gray-500 bg-white/[0.05] px-2 py-1 rounded">
                  {info.size}
                </div>

                {/* Action button */}
                <div className="flex-shrink-0">
                  {model.status === "not_downloaded" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDownload(model.variant);
                      }}
                      className="btn btn-secondary text-sm py-2 px-3"
                    >
                      <Download className="w-4 h-4" />
                      <span className="hidden sm:inline">Download</span>
                    </button>
                  )}
                  {isDownloaded && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onLoad(model.variant);
                      }}
                      className="btn btn-primary text-sm py-2 px-3"
                    >
                      <Play className="w-4 h-4" />
                      <span className="hidden sm:inline">Load</span>
                    </button>
                  )}
                  {isReady && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onUnload(model.variant);
                      }}
                      className="btn btn-danger text-sm py-2 px-3"
                    >
                      <Square className="w-4 h-4" />
                      <span className="hidden sm:inline">Unload</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Progress bar for downloading */}
              {isDownloading && (
                <div className="mt-3 progress-bar">
                  <motion.div
                    className="progress-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
