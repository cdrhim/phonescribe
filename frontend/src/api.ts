import type {
  CloudRecordingReady,
  CloudUploadDescriptor,
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
const API_ACCESS_TOKEN_STORAGE_KEY = "local-meetscribe.remote-session.v1";
const MAX_CLOUD_PART_SIZE_BYTES = 6 * 1024 * 1024;
const CLOUD_UPLOAD_RETRY_DELAYS_MS = [0, 1000, 3000, 5000, 10000, 20000];
const CLOUD_UPLOAD_ATTEMPT_TIMEOUT_MS = 120000;
const API_NETWORK_ERROR_MESSAGE =
  "서버 연결이 잠시 끊겼습니다. 파일은 Google로 전송되지 않았습니다. 잠시 후 다시 시도하세요.";
let apiAccessToken: string | null = restoreApiAccessToken();

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

export class ApiNetworkError extends Error {
  constructor(message = API_NETWORK_ERROR_MESSAGE) {
    super(message);
    this.name = "ApiNetworkError";
  }
}

export function hasApiAccessToken(): boolean {
  return Boolean(apiAccessToken);
}

export function isApiAuthenticationError(error: unknown): boolean {
  return error instanceof ApiRequestError && error.status === 401;
}

export function isApiTransientError(error: unknown): boolean {
  return (
    error instanceof TypeError ||
    error instanceof ApiNetworkError ||
    (error instanceof Error && error.message === API_NETWORK_ERROR_MESSAGE) ||
    (error instanceof ApiRequestError && isTransientUploadStatus(error.status))
  );
}

export function clearApiAccessToken(): void {
  apiAccessToken = null;
  try {
    window.sessionStorage.removeItem(API_ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    // Some privacy modes disable session storage; in-memory access still works.
  }
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
      throw new ApiNetworkError();
    }
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiRequestError(String(body.detail ?? response.statusText), response.status);
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
  quickScan = true,
  signal?: AbortSignal
): Promise<OptimizerAnalysisResponse> {
  const form = optimizerForm(options);
  form.append("language", language);
  form.append("quick_scan", String(quickScan));
  return request<OptimizerAnalysisResponse>("/api/optimizer/analyze", {
    method: "POST",
    body: form,
    signal
  });
}

export function createCloudUploadDescriptor(file: File): Promise<CloudUploadDescriptor> {
  return request<CloudUploadDescriptor>("/api/cloud-recordings/upload-descriptor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      content_type: file.type || "application/octet-stream",
      size_bytes: file.size
    })
  });
}

export async function uploadCloudRecording(
  file: File,
  descriptor: CloudUploadDescriptor,
  onProgress?: (uploadedBytes: number, totalBytes: number) => void,
  signal?: AbortSignal,
  onRetry?: (partNumber: number, attempt: number, delayMs: number) => void
): Promise<void> {
  validateCloudUploadDescriptor(file, descriptor);
  const parts = [...descriptor.parts].sort(
    (left, right) => left.part_number - right.part_number
  );
  let completedBytes = 0;
  onProgress?.(0, file.size);

  for (const part of parts) {
    if (signal?.aborted) throw abortError();
    const payload = file.slice(part.byte_start, part.byte_end, file.type);
    await uploadSignedPart(payload, descriptor.content_type, part, signal, (attempt, delayMs) => {
      onRetry?.(part.part_number, attempt, delayMs);
    });
    completedBytes += part.size_bytes;
    onProgress?.(Math.min(file.size, completedBytes), file.size);
  }
}

export function completeCloudRecordingUpload(
  recordingId: string
): Promise<CloudRecordingReady> {
  return request<CloudRecordingReady>(
    `/api/cloud-recordings/${encodeURIComponent(recordingId)}/complete`,
    { method: "POST" }
  );
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
    cloudRecordingId?: string;
    apiKey?: string;
    sharePasscode?: string;
  }
): Promise<TranscriptionWorkflowStart> {
  const form = optimizerForm(options, false);
  if (input.uploadId) form.append("upload_id", input.uploadId);
  if (input.packageId) form.append("package_id", input.packageId);
  if (input.cloudRecordingId) {
    form.append("cloud_recording_id", input.cloudRecordingId);
  }
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

function validateCloudUploadDescriptor(
  file: File,
  descriptor: CloudUploadDescriptor
): void {
  if (!descriptor.recording_id || !descriptor.bucket_id || !descriptor.parts.length) {
    throw new Error("클라우드 업로드 정보가 올바르지 않습니다.");
  }
  const parts = [...descriptor.parts].sort(
    (left, right) => left.part_number - right.part_number
  );
  let expectedStart = 0;
  for (const [expectedPartNumber, part] of parts.entries()) {
    if (
      part.upload.protocol !== "signed-put" ||
      !isHttpsUrl(part.upload.url) ||
      part.part_number !== expectedPartNumber ||
      part.byte_start !== expectedStart ||
      part.byte_end <= part.byte_start ||
      part.size_bytes !== part.byte_end - part.byte_start ||
      part.size_bytes > MAX_CLOUD_PART_SIZE_BYTES ||
      !part.object_path
    ) {
      throw new Error("클라우드 업로드 조각 정보가 올바르지 않습니다.");
    }
    expectedStart = part.byte_end;
  }
  if (expectedStart !== file.size) {
    throw new Error("클라우드 업로드 크기가 녹음 파일과 일치하지 않습니다.");
  }
}

async function uploadSignedPart(
  payload: Blob,
  contentType: string,
  part: CloudUploadDescriptor["parts"][number],
  signal?: AbortSignal,
  onRetry?: (attempt: number, delayMs: number) => void
): Promise<void> {
  let lastFailure: unknown = null;
  for (const [attemptIndex, delayMs] of CLOUD_UPLOAD_RETRY_DELAYS_MS.entries()) {
    if (delayMs > 0) {
      onRetry?.(attemptIndex + 1, delayMs);
      await abortableDelay(delayMs, signal);
    }
    if (signal?.aborted) throw abortError();
    const attemptController = new AbortController();
    let timedOut = false;
    const handleUserAbort = () => attemptController.abort();
    signal?.addEventListener("abort", handleUserAbort, { once: true });
    if (signal?.aborted) handleUserAbort();
    const timeout = window.setTimeout(() => {
      timedOut = true;
      attemptController.abort();
    }, CLOUD_UPLOAD_ATTEMPT_TIMEOUT_MS);
    try {
      const headers = new Headers(part.upload.headers);
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", contentType || "application/octet-stream");
      }
      const response = await fetch(part.upload.url, {
        method: "PUT",
        headers,
        body: payload,
        signal: attemptController.signal
      });
      if (response.ok) return;
      lastFailure = new ApiRequestError(
        `클라우드 업로드가 거절되었습니다. (HTTP ${response.status})`,
        response.status
      );
      if (!isTransientUploadStatus(response.status)) throw lastFailure;
    } catch (error) {
      if (signal?.aborted) throw abortError();
      if (timedOut) {
        lastFailure = new ApiNetworkError("클라우드 업로드 응답 시간이 초과되었습니다.");
        continue;
      }
      if (isAbortException(error)) throw abortError();
      lastFailure = error;
      if (!(error instanceof TypeError)) throw error;
    } finally {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", handleUserAbort);
    }
  }
  throw lastFailure instanceof Error
    ? lastFailure
    : new Error("클라우드 업로드를 완료하지 못했습니다.");
}

function isHttpsUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && Boolean(url.hostname);
  } catch {
    return false;
  }
}

function isTransientUploadStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function isAbortException(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function abortableDelay(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort);
      resolve();
    }, delayMs);
    const handleAbort = () => {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", handleAbort);
      reject(abortError());
    };
    signal?.addEventListener("abort", handleAbort, { once: true });
  });
}

function abortError(): DOMException {
  return new DOMException("Upload aborted", "AbortError");
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
    saveApiAccessToken(result.access_token, result.expires_in);
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
    throw new ApiRequestError(String(body.detail ?? response.statusText), response.status);
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

function saveApiAccessToken(token: string, expiresIn: number): void {
  apiAccessToken = token;
  try {
    window.sessionStorage.setItem(
      API_ACCESS_TOKEN_STORAGE_KEY,
      JSON.stringify({
        token,
        expiresAt: Date.now() + Math.max(0, expiresIn) * 1000
      })
    );
  } catch {
    // Some privacy modes disable session storage; in-memory access still works.
  }
}

function restoreApiAccessToken(): string | null {
  try {
    const stored = window.sessionStorage.getItem(API_ACCESS_TOKEN_STORAGE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored) as { token?: unknown; expiresAt?: unknown };
    if (
      typeof parsed.token !== "string" ||
      typeof parsed.expiresAt !== "number" ||
      parsed.expiresAt <= Date.now()
    ) {
      window.sessionStorage.removeItem(API_ACCESS_TOKEN_STORAGE_KEY);
      return null;
    }
    return parsed.token;
  } catch {
    return null;
  }
}
