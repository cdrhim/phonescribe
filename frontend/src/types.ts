export type JobStatus = "queued" | "running" | "completed" | "failed";
export type Mode = "accurate" | "fast" | "cpu";
export type Language = "auto" | "ko" | "en";
export type ExportKind = "json" | "md" | "txt" | "srt" | "vtt" | "docx" | "minutes_md";
export type OptimizerDestination = "gemini" | "openai" | "optimize";
export type OpenAITranscribeModel =
  | "gpt-4o-transcribe"
  | "gpt-4o-mini-transcribe"
  | "whisper-1";
export type OptimizerCodec = "mp3" | "m4a" | "ogg";

export interface RuntimeProfile {
  device: "cpu" | "cuda";
  cuda: boolean;
  fast_model: string;
  accurate_model: string;
  gemini_transcription_enabled: boolean;
  gemini_api_key_configured: boolean;
  gemini_model: string;
  gemini_share_enabled: boolean;
  gemini_share_ready: boolean;
  local_admin: boolean;
  cloud_upload_enabled?: boolean;
}

export interface CloudUploadPart {
  part_number: number;
  byte_start: number;
  byte_end: number;
  size_bytes: number;
  object_path: string;
  upload: {
    protocol: "signed-put";
    url: string;
    headers: Record<string, string>;
  };
}

export interface CloudUploadDescriptor {
  recording_id: string;
  bucket_id: string;
  object_path: string;
  content_type: string;
  parts: CloudUploadPart[];
  expires_in: number;
}

export interface CloudRecordingReady {
  recording_id: string;
  status: "ready";
}

export interface QuickScanResult {
  glossary: string[];
  preview_text: string;
  detected_language: Language | "unknown";
  scan_seconds: number;
  warning: string | null;
}

export interface PreparedAudioResult {
  id: string;
  filename: string;
  download_url: string;
  original_duration_sec: number;
  prepared_duration_sec: number;
  original_bytes: number;
  prepared_bytes: number;
  compression_ratio: number;
  remove_silence: boolean;
  max_minutes: number | null;
  bitrate_kbps: number;
}

export interface SourceInfo {
  filename: string;
  duration_sec: number;
  sample_rate: number;
  channels: number;
}

export interface OptimizerRecommendation {
  destination: OptimizerDestination;
  provider_label: string;
  model: string | null;
  codec: OptimizerCodec;
  sample_rate_hz: number;
  channels: number;
  bitrate_kbps: number;
  chunk_count: number;
  chunk_minutes: number | null;
  projected_size_mb: number;
  projected_chunk_mb: number;
  estimated_tokens: number | null;
  estimated_cost_usd: number | null;
  delivery: string;
  rationale: string;
  warnings: string[];
  prompt: string;
}

export interface OptimizerRecommendationResponse {
  source: SourceInfo;
  original_bytes: number;
  recommendation: OptimizerRecommendation;
}

export interface OptimizerAnalysisResponse extends OptimizerRecommendationResponse {
  upload_id: string;
  quick_scan: QuickScanResult;
}

export interface OptimizedChunk {
  filename: string;
  download_url: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  bytes: number;
}

export interface OptimizedPackageResult {
  id: string;
  source: SourceInfo;
  recommendation: OptimizerRecommendation;
  chunks: OptimizedChunk[];
  manifest_url: string;
  package_url: string;
}

export interface GeminiTranscriptChunk {
  filename: string;
  start_sec: number;
  end_sec: number;
  delivery: "inline" | "files_api";
  mime_type: string;
}

export interface GeminiTranscriptResult {
  provider: "gemini";
  model: string;
  text: string;
  suggested_filename: string;
  chunk_count: number;
  chunks: GeminiTranscriptChunk[];
  txt_url: string;
  json_url: string;
}

export interface GeminiTranscriptionProgress {
  status: "idle" | "transcribing" | "complete" | "failed";
  completed_chunks: number;
  total_chunks: number;
  current_chunk: number | null;
  progress: number;
  elapsed_sec: number;
  eta_sec: number | null;
}

export type TranscriptionWorkflowStage =
  | "queued"
  | "optimizing"
  | "transcribing"
  | "complete"
  | "failed";

export interface TranscriptionWorkflowStart {
  workflow_id: string;
  package_id: string;
  status: TranscriptionWorkflowStage;
}

export interface TranscriptionWorkflowStatus extends TranscriptionWorkflowStart {
  error: string | null;
  auto_exported?: boolean | null;
  auto_export_error?: string | null;
  package?: OptimizedPackageResult;
  transcription_progress?: GeminiTranscriptionProgress;
  transcript?: GeminiTranscriptResult;
}

export interface JobRecord {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  source_path?: string | null;
  output_dir?: string | null;
  transcript_path?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface Word {
  word: string;
  start: number;
  end: number;
  confidence: number | null;
  speaker: string;
}

export interface Segment {
  id: string;
  start: number;
  end: number;
  speaker: string;
  language: "ko" | "en" | "mixed" | "unknown";
  text_raw: string;
  text_clean: string;
  confidence: number | null;
  needs_review: boolean;
  overlap: boolean;
  words: Word[];
}

export interface Speaker {
  id: string;
  display_name: string;
  total_sec: number;
}

export interface Transcript {
  id: string;
  source: {
    filename: string;
    duration_sec: number;
    sample_rate: number;
    channels: number;
  };
  config: {
    mode: Mode;
    asr_engine: string;
    asr_model: string;
    diarization_engine: string;
    language: Language;
    alignment_engine: string;
    vad_engine: string;
    formatter_engine: string;
  };
  speakers: Speaker[];
  segments: Segment[];
  exports: Record<ExportKind, string | null>;
  created_at: string;
}
