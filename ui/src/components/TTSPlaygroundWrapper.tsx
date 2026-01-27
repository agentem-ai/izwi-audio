import { useState } from "react";
import { motion } from "framer-motion";
import clsx from "clsx";
import { CustomVoicePlayground } from "./CustomVoicePlayground";
import { LFM2TTSPlayground } from "./LFM2TTSPlayground";

type TTSEngine = "qwen3" | "lfm2";

interface TTSPlaygroundWrapperProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function TTSPlaygroundWrapper({
  selectedModel,
  onModelRequired,
}: TTSPlaygroundWrapperProps) {
  const [engine, setEngine] = useState<TTSEngine>("qwen3");

  return (
    <div className="space-y-4">
      {/* Engine selector */}
      <div className="flex items-center gap-2 p-1 bg-[#0d0d0d] rounded-lg border border-[#2a2a2a]">
        <button
          onClick={() => setEngine("qwen3")}
          className={clsx(
            "flex-1 px-4 py-2 rounded-md text-sm font-medium transition-all relative",
            engine === "qwen3"
              ? "text-white"
              : "text-gray-400 hover:text-gray-300"
          )}
        >
          {engine === "qwen3" && (
            <motion.div
              layoutId="ttsEngineIndicator"
              className="absolute inset-0 bg-[#1a1a1a] rounded-md border border-[#2a2a2a]"
              transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
            />
          )}
          <span className="relative z-10">Qwen3-TTS</span>
        </button>
        <button
          onClick={() => setEngine("lfm2")}
          className={clsx(
            "flex-1 px-4 py-2 rounded-md text-sm font-medium transition-all relative",
            engine === "lfm2"
              ? "text-white"
              : "text-gray-400 hover:text-gray-300"
          )}
        >
          {engine === "lfm2" && (
            <motion.div
              layoutId="ttsEngineIndicator"
              className="absolute inset-0 bg-[#1a1a1a] rounded-md border border-[#2a2a2a]"
              transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
            />
          )}
          <span className="relative z-10">LFM2-Audio</span>
        </button>
      </div>

      {/* Engine description */}
      <p className="text-xs text-gray-500">
        {engine === "qwen3"
          ? "Qwen3-TTS: 9 built-in voices with speaking style instructions"
          : "LFM2-Audio: Liquid AI's end-to-end audio model with 4 voices (US/UK Male/Female)"}
      </p>

      {/* Playground based on selected engine */}
      {engine === "qwen3" ? (
        <CustomVoicePlayground
          selectedModel={selectedModel}
          onModelRequired={onModelRequired}
        />
      ) : (
        <LFM2TTSPlayground />
      )}
    </div>
  );
}
