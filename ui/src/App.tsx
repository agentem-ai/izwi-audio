import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, AlertCircle, X, Github, Waves, ChevronRight } from "lucide-react";
import { ModelManager } from "./components/ModelManager";
import { VoicePlayground } from "./components/VoicePlayground";
import { api, ModelInfo } from "./api";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, number>
  >({});
  const [showModels, setShowModels] = useState(true);

  const loadModels = useCallback(async () => {
    try {
      const response = await api.listModels();
      setModels(response.models);

      // Auto-select first ready model
      const readyModel = response.models.find((m) => m.status === "ready");
      if (readyModel && !selectedModel) {
        setSelectedModel(readyModel.variant);
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  }, [selectedModel]);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await loadModels();
      setLoading(false);
    };
    init();

    // Poll for model status updates
    const interval = setInterval(loadModels, 5000);
    return () => clearInterval(interval);
  }, [loadModels]);

  const handleDownload = async (variant: string) => {
    try {
      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "downloading" as const } : m,
        ),
      );

      // Simulate progress updates (real implementation would use SSE/WebSocket)
      const progressInterval = setInterval(() => {
        setDownloadProgress((prev) => {
          const current = prev[variant] || 0;
          if (current >= 95) {
            clearInterval(progressInterval);
            return prev;
          }
          return {
            ...prev,
            [variant]: Math.min(current + Math.random() * 15, 95),
          };
        });
      }, 500);

      await api.downloadModel(variant);

      clearInterval(progressInterval);
      setDownloadProgress((prev) => ({ ...prev, [variant]: 100 }));

      await loadModels();

      // Clear progress after a delay
      setTimeout(() => {
        setDownloadProgress((prev) => {
          const { [variant]: _, ...rest } = prev;
          return rest;
        });
      }, 1000);
    } catch (err) {
      console.error("Download failed:", err);
      setError("Failed to download model. Please try again.");
      await loadModels();
    }
  };

  const handleLoad = async (variant: string) => {
    try {
      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "loading" as const } : m,
        ),
      );

      await api.loadModel(variant);
      await loadModels();
      setSelectedModel(variant);
    } catch (err) {
      console.error("Load failed:", err);
      setError("Failed to load model. Please try again.");
      await loadModels();
    }
  };

  const handleUnload = async (variant: string) => {
    try {
      await api.unloadModel(variant);
      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Unload failed:", err);
    }
  };

  const readyModelsCount = models.filter((m) => m.status === "ready").length;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                  <Waves className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#0a0a0f]" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">
                  Izwi
                </h1>
                <p className="text-xs text-gray-500">AI Voice Playground</p>
              </div>
            </div>

            {/* Status */}
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.05] border border-white/[0.08]">
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                <span className="text-xs text-gray-400">Qwen3-TTS</span>
                {readyModelsCount > 0 && (
                  <span className="text-xs text-emerald-400">
                    • {readyModelsCount} loaded
                  </span>
                )}
              </div>
              <a
                href="https://github.com/QwenLM/Qwen3-TTS"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg hover:bg-white/[0.05] transition-colors"
              >
                <Github className="w-5 h-5 text-gray-400 hover:text-white" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Error toast */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-20 left-1/2 -translate-x-1/2 z-50"
          >
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 backdrop-blur-xl shadow-xl">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <span className="text-sm text-red-200">{error}</span>
              <button
                onClick={() => setError(null)}
                className="p-1 rounded-lg hover:bg-white/[0.1] transition-colors"
              >
                <X className="w-4 h-4 text-red-400" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
        <div className="grid lg:grid-cols-[380px,1fr] gap-6">
          {/* Models sidebar */}
          <div className="lg:block">
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
                  Models
                </h2>
                <button
                  onClick={() => setShowModels(!showModels)}
                  className="lg:hidden p-1 rounded hover:bg-white/[0.1]"
                >
                  <ChevronRight
                    className={`w-4 h-4 text-gray-400 transition-transform ${showModels ? "rotate-90" : ""}`}
                  />
                </button>
              </div>

              <AnimatePresence>
                {(showModels || window.innerWidth >= 1024) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                  >
                    {loading ? (
                      <div className="flex flex-col items-center justify-center py-12 gap-3">
                        <motion.div
                          className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full"
                          animate={{ rotate: 360 }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            ease: "linear",
                          }}
                        />
                        <p className="text-sm text-gray-500">
                          Loading models...
                        </p>
                      </div>
                    ) : (
                      <ModelManager
                        models={models}
                        selectedModel={selectedModel}
                        onDownload={handleDownload}
                        onLoad={handleLoad}
                        onUnload={handleUnload}
                        onSelect={setSelectedModel}
                        downloadProgress={downloadProgress}
                      />
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Voice playground */}
          <div>
            <VoicePlayground
              selectedModel={selectedModel}
              onModelRequired={() =>
                setError("Please load a model first to generate speech")
              }
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.05] py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-gray-600">
              Powered by Qwen3-TTS • Built with ❤️ for the open-source community
            </p>
            <div className="flex items-center gap-4">
              <a
                href="#"
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                Documentation
              </a>
              <a
                href="#"
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                API Reference
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
