import { useState, useEffect } from "react";
import { Volume2, Download, Loader2, AlertCircle } from "lucide-react";
import { ModelCard } from "./components/ModelCard";
import { TTSPanel } from "./components/TTSPanel";
import { api, ModelInfo } from "./api";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const response = await api.listModels();
      setModels(response.models);

      // Auto-select first ready model
      const readyModel = response.models.find((m) => m.status === "ready");
      if (readyModel) {
        setSelectedModel(readyModel.variant);
      }
    } catch (err) {
      setError("Failed to load models");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (variant: string) => {
    try {
      // Update UI to show downloading
      setModels((prev) =>
        prev.map((m) =>
          m.variant === variant ? { ...m, status: "downloading" as const } : m,
        ),
      );

      await api.downloadModel(variant);
      await loadModels();
    } catch (err) {
      console.error("Download failed:", err);
      setError("Failed to download model");
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
      setError("Failed to load model");
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

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Volume2 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">Izwi</h1>
                <p className="text-sm text-gray-400">
                  Qwen3-TTS Inference Engine
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-400">Apple Silicon • MLX</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-200">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Models Panel */}
          <div className="lg:col-span-1">
            <div className="card">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Download className="w-5 h-5" />
                Models
              </h2>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
                </div>
              ) : (
                <div className="space-y-3">
                  {models
                    .filter((m) => !m.variant.includes("tokenizer"))
                    .map((model) => (
                      <ModelCard
                        key={model.variant}
                        model={model}
                        isSelected={selectedModel === model.variant}
                        onDownload={() => handleDownload(model.variant)}
                        onLoad={() => handleLoad(model.variant)}
                        onUnload={() => handleUnload(model.variant)}
                        onSelect={() => setSelectedModel(model.variant)}
                      />
                    ))}
                </div>
              )}

              {/* Tokenizer section */}
              <div className="mt-6 pt-6 border-t border-gray-800">
                <h3 className="text-sm font-medium text-gray-400 mb-3">
                  Audio Codec
                </h3>
                {models
                  .filter((m) => m.variant.includes("tokenizer"))
                  .map((model) => (
                    <ModelCard
                      key={model.variant}
                      model={model}
                      isSelected={false}
                      onDownload={() => handleDownload(model.variant)}
                      onLoad={() => handleLoad(model.variant)}
                      onUnload={() => handleUnload(model.variant)}
                      onSelect={() => {}}
                      compact
                    />
                  ))}
              </div>
            </div>
          </div>

          {/* TTS Panel */}
          <div className="lg:col-span-2">
            <TTSPanel
              selectedModel={selectedModel}
              onModelRequired={() => setError("Please load a model first")}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <p className="text-center text-sm text-gray-500">
            Izwi TTS Engine • Powered by Qwen3-TTS and MLX
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
