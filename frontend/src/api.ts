import type {
  ExportKind,
  GeminiTranscriptionProgress,
  GeminiTranscriptResult,
  JobRecord,
  Language,
  Mode,
  OpenAITranscribeModel,
  OptimizedPackageResult,
  OptimizerAnalysisResponse,
  OptimizerCodec,
  OptimizerDestination,
  OptimizerRecommendationResponse,
  PreparedAudioResult,
  QuickScanResult,
  RuntimeProfile,
  Transcript,
  TranscriptionWorkflowStart,
  TranscriptionWorkflowStatus
} from "./types";

export interface UploadOptions {
  file: File;
  mode: Mode;
  language: Language;
  speakers: string;
  minSpeakers: string;
  maxSpeakers: string;
  glossary: string;
  denoise: boolean;
  loudnessNormalize: boolean;
  trimSilence: boolean;
}

export interface PrepareAudioOptions {
  file: File;
  removeSilence: boolean;
  maxMinutes: string;
  bitrateKbps: string;
}

export interface OptimizerOptions {
  file?: File;
  destination: OptimizerDestination;
  openaiModel: OpenAITranscribeModel;
  wordTimestamps: boolean;
  codec: OptimizerCodec | "";
  bitrateKbps: string;
  chunkMinutes: string;
  removeSilence: boolean;
  loudnorm: boolean;
  speechFilter: boolean;
  denoise: boolean;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
let apiAccessToken: string | null = null;

export function clearApiAccessToken(): void {
  apiAccessToken = null;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(init?.headers);
    if (apiAccessToken && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${apiAccessToken}`);
    }
    response = await fetch(apiUrl(url), { ...init, headers });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        "서버 연결이 잠시 끊겼습니다. 파일은 Google로 전송되지 않았습니다. 잠시 후 다시 시도하세요."
      );
    }
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? response.statusText);
  }
  return (await response.json()) as T;
}

export function listJobs(): Promise<JobRecord[]> {
  return request<JobRecord[]>("/api/jobs");
}

export function getJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/api/jobs/${jobId}`);
}

export function getTranscript(jobId: string): Promise<Transcript> {
  return request<Transcript>(`/api/jobs/${jobId}/transcript`);
}

export function getRuntime(): Promise<RuntimeProfile> {
  return request<RuntimeProfile>("/api/runtime?client=share-v1", { cache: "no-store" });
}

export function quickScanFile(file: File, language: Language): Promise<QuickScanResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  return request<QuickScanResult>("/api/files/quick-scan", { method: "POST", body: form });
}

export function prepareAudio(options: PrepareAudioOptions): Promise<PreparedAudioResult> {
  const form = new FormData();
  form.append("file", options.file);
  form.append("remove_silence", String(options.removeSilence));
  if (options.maxMinutes.trim()) form.append("max_minutes", options.maxMinutes.trim());
  form.append("bitrate_kbps", options.bitrateKbps);
  return request<PreparedAudioResult>("/api/files/prepare-audio", { method: "POST", body: form });
}

export function recommendOptimizer(
  options: OptimizerOptions
): Promise<OptimizerRecommendationResponse> {
  return request<OptimizerRecommendationResponse>("/api/optimizer/recommend", {
    method: "POST",
    body: optimizerForm(options)
  });
}

export function analyzeOptimizer(
  options: OptimizerOptions,
  language: Language,
  quickScan = true
): Promise<OptimizerAnalysisResponse> {
  const form = optimizerForm(options);
  form.append("language", language);
  form.append("quick_scan", String(quickScan));
  return request<OptimizerAnalysisResponse>("/api/optimizer/analyze", {
    method: "POST",
    body: form
  });
}

export function createOptimizerPackage(
  options: OptimizerOptions,
  uploadId?: string
): Promise<OptimizedPackageResult> {
  const form = optimizerForm(options, !uploadId);
  if (uploadId) form.append("upload_id", uploadId);
  return request<OptimizedPackageResult>("/api/optimizer/package", {
    method: "POST",
    body: form
  });
}

export function transcribeGeminiPackage(
  packageId: string,
  apiKey?: string,
  sharePasscode?: string
): Promise<GeminiTranscriptResult> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-Gemini-API-Key"] = apiKey;
  if (sharePasscode) headers["X-LocalMeetScribe-Passcode"] = sharePasscode;
  return request<GeminiTranscriptResult>(
    `/api/optimizer/packages/${packageId}/gemini-transcribe`,
    { method: "POST", headers: Object.keys(headers).length ? headers : undefined }
  );
}

export function startTranscriptionWorkflow(
  options: OptimizerOptions,
  input: {
    uploadId?: string;
    packageId?: string;
    apiKey?: string;
    sharePasscode?: string;
  }
): Promise<TranscriptionWorkflowStart> {
  const form = optimizerForm(options, false);
  if (input.uploadId) form.append("upload_id", input.uploadId);
  if (input.packageId) form.append("package_id", input.packageId);
  const headers: Record<string, string> = {};
  if (input.apiKey) headers["X-Gemini-API-Key"] = input.apiKey;
  if (input.sharePasscode) {
    headers["X-LocalMeetScribe-Passcode"] = input.sharePasscode;
  }
  return request<TranscriptionWorkflowStart>("/api/workflows", {
    method: "POST",
    headers: Object.keys(headers).length ? headers : undefined,
    body: form
  });
}

export function getTranscriptionWorkflow(
  workflowId: string
): Promise<TranscriptionWorkflowStatus> {
  return request<TranscriptionWorkflowStatus>(`/api/workflows/${workflowId}`, {
    cache: "no-store"
  });
}

export function verifyGeminiSharePasscode(
  sharePasscode: string
): Promise<{ valid: boolean; key_ready: boolean; expires_in: number }> {
  clearApiAccessToken();
  return request<{
    valid: boolean;
    key_ready: boolean;
    access_token: string;
    expires_in: number;
  }>("/api/gemini-share/verify", {
    method: "POST",
    headers: { "X-LocalMeetScribe-Passcode": sharePasscode }
  }).then((result) => {
    apiAccessToken = result.access_token;
    return result;
  });
}

export function configureGeminiShareKey(
  apiKey: string,
  sharePasscode: string
): Promise<{ configured: boolean }> {
  const form = new FormData();
  form.append("api_key", apiKey);
  return request<{ configured: boolean }>("/api/admin/gemini-share-key", {
    method: "POST",
    headers: { "X-LocalMeetScribe-Passcode": sharePasscode },
    body: form
  });
}

export function getGeminiTranscriptionProgress(
  packageId: string
): Promise<GeminiTranscriptionProgress> {
  return request<GeminiTranscriptionProgress>(
    `/api/optimizer/packages/${packageId}/gemini-transcribe/progress`
  );
}

function optimizerForm(options: OptimizerOptions, includeFile = true): FormData {
  const form = new FormData();
  if (includeFile) {
    if (!options.file) throw new Error("최적화할 녹음 파일이 필요합니다.");
    form.append("file", options.file);
  }
  form.append("destination", options.destination);
  form.append("openai_model", options.openaiModel);
  form.append("word_timestamps", String(options.wordTimestamps));
  form.append("remove_silence", String(options.removeSilence));
  form.append("loudnorm", String(options.loudnorm));
  form.append("speech_filter", String(options.speechFilter));
  form.append("denoise", String(options.denoise));
  if (options.codec) form.append("codec", options.codec);
  if (options.bitrateKbps.trim()) form.append("bitrate_kbps", options.bitrateKbps.trim());
  if (options.chunkMinutes.trim()) form.append("chunk_minutes", options.chunkMinutes.trim());
  return form;
}

export function uploadJob(options: UploadOptions): Promise<JobRecord> {
  const form = new FormData();
  form.append("file", options.file);
  form.append("mode", options.mode);
  form.append("language", options.language);
  form.append("glossary", options.glossary);
  form.append("denoise", String(options.denoise));
  form.append("loudness_normalize", String(options.loudnessNormalize));
  form.append("trim_silence", String(options.trimSilence));
  if (options.speakers) form.append("speakers", options.speakers);
  if (options.minSpeakers) form.append("min_speakers", options.minSpeakers);
  if (options.maxSpeakers) form.append("max_speakers", options.maxSpeakers);
  return request<JobRecord>("/api/jobs", { method: "POST", body: form });
}

export function saveTranscript(transcript: Transcript): Promise<Transcript> {
  return request<Transcript>(`/api/jobs/${transcript.id}/transcript`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speakers: transcript.speakers.map((speaker) => ({
        id: speaker.id,
        display_name: speaker.display_name
      })),
      segments: transcript.segments.map((segment) => ({
        id: segment.id,
        text_clean: segment.text_clean
      }))
    })
  });
}

export function exportUrl(jobId: string, kind: ExportKind): string {
  return apiUrl(`/api/jobs/${jobId}/exports/${kind}`);
}

export function audioUrl(jobId: string): string {
  return apiUrl(`/api/jobs/${jobId}/audio`);
}

export async function downloadApiFile(url: string, downloadName: string): Promise<void> {
  const headers = new Headers();
  if (apiAccessToken) headers.set("Authorization", `Bearer ${apiAccessToken}`);
  const response = await fetch(apiUrl(withDownloadName(url, downloadName)), {
    headers,
    cache: "no-store"
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? response.statusText);
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = downloadName;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

function apiUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return `${API_BASE_URL}${normalized}`;
}

function withDownloadName(url: string, downloadName: string): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}download_name=${encodeURIComponent(downloadName)}`;
}
