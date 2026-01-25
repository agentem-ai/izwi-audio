const API_BASE = "/api/v1";

export interface ModelInfo {
  variant: string;
  status:
    | "not_downloaded"
    | "downloading"
    | "downloaded"
    | "loading"
    | "ready"
    | "error";
  local_path: string | null;
  size_bytes: number | null;
  download_progress: number | null;
  error_message: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface TTSRequest {
  text: string;
  speaker?: string;
  voice_description?: string;
  reference_audio?: string;
  reference_text?: string;
  format?: "wav" | "raw_f32" | "raw_i16";
  temperature?: number;
  speed?: number;
}

export interface TTSResponse {
  request_id: string;
  audio: string; // base64
  format: string;
  sample_rate: number;
  duration_secs: number;
  stats: {
    tokens_generated: number;
    generation_time_ms: number;
    rtf: number;
  };
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "Request failed" } }));
      throw new Error(error.error?.message || "Request failed");
    }

    return response.json();
  }

  async listModels(): Promise<ModelsResponse> {
    return this.request("/models");
  }

  async getModelInfo(variant: string): Promise<ModelInfo> {
    return this.request(`/models/${variant}`);
  }

  async downloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/download`, { method: "POST" });
  }

  async loadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/load`, { method: "POST" });
  }

  async unloadModel(
    variant: string,
  ): Promise<{ status: string; message: string }> {
    return this.request(`/models/${variant}/unload`, { method: "POST" });
  }

  async generateTTS(request: TTSRequest): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/tts/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ...request, format: "wav" }),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS generation failed" } }));
      throw new Error(error.error?.message || "TTS generation failed");
    }

    return response.blob();
  }

  async generateTTSStream(request: TTSRequest): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/tts/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "TTS streaming failed" } }));
      throw new Error(error.error?.message || "TTS streaming failed");
    }

    return response;
  }
}

export const api = new ApiClient();
