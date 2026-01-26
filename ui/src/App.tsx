import { useState, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle, X, Github, Waves, ChevronRight } from "lucide-react";
import { ModelManager } from "./components/ModelManager";
import { ViewSwitcher } from "./components/ViewSwitcher";
import { CustomVoicePlayground } from "./components/CustomVoicePlayground";
import { VoiceClonePlayground } from "./components/VoiceClonePlayground";
import { VoiceDesignPlayground } from "./components/VoiceDesignPlayground";
import { api, ModelInfo } from "./api";
import { ViewMode, VIEW_CONFIGS } from "./types";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, number>
  >({});
  const [showModels, setShowModels] = useState(true);
  const [currentView, setCurrentView] = useState<ViewMode>("custom-voice");

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

  const handleDelete = async (variant: string) => {
    try {
      // Unload if loaded
      if (models.find((m) => m.variant === variant)?.status === "ready") {
        await api.unloadModel(variant);
      }

      // TODO: Add delete endpoint to API
      // For now, just show error
      setError("Delete functionality requires backend API endpoint");

      await loadModels();
      if (selectedModel === variant) {
        setSelectedModel(null);
      }
    } catch (err) {
      console.error("Delete failed:", err);
      setError("Failed to delete model");
    }
  };

  const readyModelsCount = models.filter((m) => m.status === "ready").length;

  const currentViewConfig = VIEW_CONFIGS[currentView];

  const modelCounts = useMemo(() => {
    const counts: Record<ViewMode, { total: number; ready: number }> = {
      "custom-voice": { total: 0, ready: 0 },
      "voice-clone": { total: 0, ready: 0 },
      "voice-design": { total: 0, ready: 0 },
    };

    models
      .filter((m) => !m.variant.includes("Tokenizer"))
      .forEach((model) => {
        Object.entries(VIEW_CONFIGS).forEach(([viewId, config]) => {
          if (config.modelFilter(model.variant)) {
            counts[viewId as ViewMode].total++;
            if (model.status === "ready") {
              counts[viewId as ViewMode].ready++;
            }
          }
        });
      });

    return counts;
  }, [models]);

  const relevantSelectedModel = useMemo(() => {
    if (!selectedModel) return null;
    if (currentViewConfig.modelFilter(selectedModel)) {
      return selectedModel;
    }
    const readyModel = models.find(
      (m) => m.status === "ready" && currentViewConfig.modelFilter(m.variant),
    );
    return readyModel?.variant || null;
  }, [selectedModel, currentView, models, currentViewConfig]);

  const handleViewChange = (view: ViewMode) => {
    setCurrentView(view);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0d0d0d]">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-[#2a2a2a] bg-[#0d0d0d]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-white rounded flex items-center justify-center">
                <Waves className="w-4 h-4 text-black" />
              </div>
              <div>
                <h1 className="text-base font-semibold text-white">
                  Izwi Audio
                </h1>
                <p className="text-xs text-gray-400">TTS Playground</p>
              </div>
            </div>

            {/* Status */}
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-2 text-xs text-gray-500">
                <span>Qwen3-TTS</span>
                {readyModelsCount > 0 && (
                  <span className="text-gray-400">
                    • {readyModelsCount} loaded
                  </span>
                )}
              </div>
              <a
                href="https://github.com/QwenLM/Qwen3-TTS"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded hover:bg-[#1a1a1a] transition-colors"
              >
                <Github className="w-4 h-4 text-gray-500 hover:text-white" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Error toast */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed top-16 left-1/2 -translate-x-1/2 z-50"
          >
            <div className="flex items-center gap-3 px-4 py-2.5 rounded bg-[#1a1a1a] border border-red-900/50">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-red-200">{error}</span>
              <button
                onClick={() => setError(null)}
                className="p-1 rounded hover:bg-[#2a2a2a] transition-colors"
              >
                <X className="w-3.5 h-3.5 text-gray-500" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8">
        {/* View Switcher */}
        <div className="mb-6">
          <ViewSwitcher
            currentView={currentView}
            onViewChange={handleViewChange}
            modelCounts={modelCounts}
          />
        </div>

        <div className="grid lg:grid-cols-[380px,1fr] gap-6">
          {/* Models sidebar */}
          <div className="lg:block">
            <div className="card p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-sm font-medium text-white">Models</h2>
                  <p className="text-[10px] text-gray-400 mt-0.5">
                    {currentViewConfig.label} compatible
                  </p>
                </div>
                <button
                  onClick={() => setShowModels(!showModels)}
                  className="lg:hidden p-1 rounded hover:bg-[#1a1a1a]"
                >
                  <ChevronRight
                    className={`w-4 h-4 text-gray-500 transition-transform ${showModels ? "rotate-90" : ""}`}
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
                      <div className="flex flex-col items-center justify-center py-12 gap-2">
                        <motion.div
                          className="w-6 h-6 border-2 border-white border-t-transparent rounded-full"
                          animate={{ rotate: 360 }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            ease: "linear",
                          }}
                        />
                        <p className="text-xs text-gray-400">Loading...</p>
                      </div>
                    ) : (
                      <ModelManager
                        models={models}
                        selectedModel={relevantSelectedModel}
                        onDownload={handleDownload}
                        onLoad={handleLoad}
                        onUnload={handleUnload}
                        onDelete={handleDelete}
                        onSelect={setSelectedModel}
                        downloadProgress={downloadProgress}
                        modelFilter={currentViewConfig.modelFilter}
                        emptyStateTitle={currentViewConfig.emptyStateTitle}
                        emptyStateDescription={
                          currentViewConfig.emptyStateDescription
                        }
                      />
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Playground area */}
          <div>
            <AnimatePresence mode="wait">
              {currentView === "custom-voice" && (
                <motion.div
                  key="custom-voice"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <CustomVoicePlayground
                    selectedModel={relevantSelectedModel}
                    onModelRequired={() =>
                      setError("Please load a CustomVoice model first")
                    }
                  />
                </motion.div>
              )}

              {currentView === "voice-clone" && (
                <motion.div
                  key="voice-clone"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <VoiceClonePlayground
                    selectedModel={relevantSelectedModel}
                    onModelRequired={() =>
                      setError("Please load a Base model first")
                    }
                  />
                </motion.div>
              )}

              {currentView === "voice-design" && (
                <motion.div
                  key="voice-design"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                >
                  <VoiceDesignPlayground
                    selectedModel={relevantSelectedModel}
                    onModelRequired={() =>
                      setError("Please load the VoiceDesign model first")
                    }
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.05] py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-gray-400">
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
