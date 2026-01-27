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

export interface LFM2TTSRequest {
  text: string;
  voice?: "us_male" | "us_female" | "uk_male" | "uk_female";
  max_new_tokens?: number;
  audio_temperature?: number;
  audio_top_k?: number;
}

export interface LFM2ASRRequest {
  audio_base64: string;
  max_new_tokens?: number;
}

export interface LFM2AudioChatRequest {
  audio_base64?: string;
  text?: string;
  max_new_tokens?: number;
  audio_temperature?: number;
  audio_top_k?: number;
}

export interface LFM2TTSResponse {
  audio_base64: string;
  sample_rate: number;
  format: string;
}

export interface LFM2ASRResponse {
  transcription: string;
}

export interface LFM2AudioChatResponse {
  text: string;
  audio_base64: string | null;
  sample_rate: number;
  format: string;
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

  async lfm2GenerateTTS(request: LFM2TTSRequest): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/lfm2/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ error: { message: "LFM2 TTS generation failed" } }));
      throw new Error(error.error?.message || "LFM2 TTS generation failed");
    }

    const data: LFM2TTSResponse = await response.json();
    const binaryString = atob(data.audio_base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  async lfm2TranscribeAudio(request: LFM2ASRRequest): Promise<LFM2ASRResponse> {
    return this.request("/lfm2/asr", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async lfm2AudioChat(
    request: LFM2AudioChatRequest,
  ): Promise<LFM2AudioChatResponse> {
    return this.request("/lfm2/chat", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async lfm2Status(): Promise<{
    status: string;
    device: string;
    cached_models: string[];
    voices: string[];
  }> {
    return this.request("/lfm2/status");
  }
}

export const api = new ApiClient();
