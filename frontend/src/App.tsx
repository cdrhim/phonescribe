import {
  AlertCircle,
  Check,
  CheckCircle2,
  Clipboard,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  FileAudio,
  FilePenLine,
  KeyRound,
  Loader2,
  Mic,
  MonitorSmartphone,
  Play,
  RefreshCw,
  ShieldCheck,
  Smartphone,
  Square
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  type OptimizerOptions,
  analyzeOptimizer,
  clearApiAccessToken,
  completeCloudRecordingUpload,
  configureGeminiShareKey,
  createCloudUploadDescriptor,
  createOptimizerPackage,
  downloadApiFile,
  getTranscriptionWorkflow,
  getRuntime,
  hasApiAccessToken,
  isApiAuthenticationError,
  isApiTransientError,
  startTranscriptionWorkflow,
  uploadCloudRecording,
  verifyGeminiSharePasscode
} from "./api";
import type {
  GeminiTranscriptionProgress,
  GeminiTranscriptResult,
  OptimizedPackageResult,
  OptimizerRecommendationResponse,
  QuickScanResult,
  RuntimeProfile,
  TranscriptionWorkflowStatus
} from "./types";

type WorkflowStage =
  | "idle"
  | "analyzing"
  | "ready"
  | "optimizing"
  | "transcribing"
  | "complete"
  | "failed";

type NamingMode = "original" | "recommended" | "custom";
type RecordingState = "idle" | "starting" | "recording" | "stopping";
type WakeLockStatus = "idle" | "requesting" | "active" | "unavailable" | "released";

const workflowSteps = ["분석", "최적화", "전사", "완료"];
const ACTIVE_WORKFLOW_STORAGE_KEY = "local-meetscribe.active-workflow.v1";
const LAST_AUTO_DOWNLOADED_WORKFLOW_KEY =
  "local-meetscribe.last-auto-downloaded-workflow.v1";
const AUTO_DOWNLOAD_RETRY_DELAYS_MS = [1000, 3000, 10000, 30000];
const RECORDING_UPLOAD_RETRY_DELAYS_MS = [1000, 3000, 10000, 30000];

interface PersistedWorkflow {
  version: 1;
  workflowId: string | null;
  stagedUploadId: string | null;
  cloudRecordingId: string | null;
  sourceName: string;
  sourceBytes: number;
  recommendation: OptimizerRecommendationResponse | null;
  optimizedPackage: OptimizedPackageResult | null;
  saveBaseName: string;
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [sourceBytes, setSourceBytes] = useState(0);
  const [runtime, setRuntime] = useState<RuntimeProfile | null>(null);
  const [runtimeChecked, setRuntimeChecked] = useState(false);
  const [scan, setScan] = useState<QuickScanResult | null>(null);
  const [recommendation, setRecommendation] =
    useState<OptimizerRecommendationResponse | null>(null);
  const [stagedUploadId, setStagedUploadId] = useState<string | null>(null);
  const [cloudRecordingId, setCloudRecordingId] = useState<string | null>(null);
  const [cloudUploadProgress, setCloudUploadProgress] = useState<number | null>(null);
  const [cloudUploadNotice, setCloudUploadNotice] = useState<string | null>(null);
  const [recordingRetryNotice, setRecordingRetryNotice] = useState<string | null>(null);
  const [recordingRetryScheduled, setRecordingRetryScheduled] = useState(false);
  const [sameRecordingRetryAvailable, setSameRecordingRetryAvailable] =
    useState(false);
  const [optimizedPackage, setOptimizedPackage] =
    useState<OptimizedPackageResult | null>(null);
  const [transcript, setTranscript] = useState<GeminiTranscriptResult | null>(null);
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [sharePasscode, setSharePasscode] = useState("");
  const [shareAccessReady, setShareAccessReady] = useState(hasApiAccessToken);
  const [shareStatus, setShareStatus] = useState<string | null>(null);
  const [verifyingShareAccess, setVerifyingShareAccess] = useState(false);
  const [accessNeedsReconnect, setAccessNeedsReconnect] = useState(false);
  const [showKeySetup, setShowKeySetup] = useState(false);
  const [savingShareKey, setSavingShareKey] = useState(false);
  const [cloudConsent, setCloudConsent] = useState(true);
  const [stage, setStage] = useState<WorkflowStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [namingMode, setNamingMode] = useState<NamingMode>("original");
  const [saveBaseName, setSaveBaseName] = useState("");
  const [originalAudioUrl, setOriginalAudioUrl] = useState<string | null>(null);
  const [transcriptionProgress, setTranscriptionProgress] =
    useState<GeminiTranscriptionProgress | null>(null);
  const [wakeLockStatus, setWakeLockStatus] = useState<WakeLockStatus>("idle");
  const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(null);
  const [autoDownloadStatus, setAutoDownloadStatus] = useState<string | null>(null);
  const [serverExportStatus, setServerExportStatus] = useState<string | null>(null);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [recordingElapsedSec, setRecordingElapsedSec] = useState(0);
  const workflowTimerRef = useRef<number | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef<number | null>(null);
  const workflowStartingRef = useRef(false);
  const analysisStartingRef = useRef(false);
  const uploadAbortRef = useRef<AbortController | null>(null);
  const recordingRetryTimerRef = useRef<number | null>(null);
  const recordingRetryAttemptRef = useRef(0);
  const autoStartSuppressedRef = useRef(false);
  // Switching recordings detaches the UI, never cancels a server-side workflow.
  const selectionVersionRef = useRef(0);
  const pollingVersionRef = useRef(0);
  const pollingInFlightRef = useRef<number | null>(null);

  useEffect(() => {
    void getRuntime()
      .then(setRuntime)
      .catch(() => setRuntime(null))
      .finally(() => setRuntimeChecked(true));
  }, []);

  useEffect(() => {
    if (!file) {
      setOriginalAudioUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setOriginalAudioUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(
    () => () => {
      selectionVersionRef.current += 1;
      uploadAbortRef.current?.abort();
      if (recordingRetryTimerRef.current !== null) {
        window.clearTimeout(recordingRetryTimerRef.current);
        recordingRetryTimerRef.current = null;
      }
      stopProgressPolling();
      void wakeLockRef.current?.release();
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        recorder.onerror = null;
        recorder.stop();
      }
      stopRecordingStream();
    },
    []
  );

  useEffect(() => {
    const saved = loadPersistedWorkflow();
    const resumableWorkflowId = saved?.workflowId || workflowIdFromUrl();
    if (saved) {
      setSourceName(saved.sourceName);
      setSourceBytes(saved.sourceBytes);
      setRecommendation(saved.recommendation);
      setStagedUploadId(saved.stagedUploadId);
      setCloudRecordingId(saved.cloudRecordingId);
      setOptimizedPackage(saved.optimizedPackage);
      setSaveBaseName(saved.saveBaseName);
    }
    if (resumableWorkflowId) {
      setActiveWorkflowId(resumableWorkflowId);
      setStage("optimizing");
      beginWorkflowPolling(resumableWorkflowId);
    } else if (saved) {
      setStage("ready");
    }
  }, []);

  const shareMode = Boolean(runtime?.gemini_share_enabled);
  const serverKeyReady = Boolean(
    !shareMode &&
      runtime?.gemini_transcription_enabled &&
      runtime.gemini_api_key_configured
  );
  const keyReady = shareMode
    ? shareAccessReady
    : serverKeyReady || geminiApiKey.trim().length > 0;
  const busy =
    stage === "analyzing" || stage === "optimizing" || stage === "transcribing";
  const recordingActive = recordingState !== "idle";
  const keepScreenAwake =
    stage === "analyzing" || recordingRetryScheduled || recordingActive;
  const wakeLockActive = wakeLockStatus === "active";
  const recordingSupported = Boolean(
    typeof MediaRecorder !== "undefined" && navigator.mediaDevices?.getUserMedia
  );
  const canStart = Boolean(
    (stagedUploadId || cloudRecordingId || optimizedPackage) &&
      (recommendation || cloudRecordingId) &&
      keyReady &&
      cloudConsent &&
      !busy
  );
  const displaySourceName =
    file?.name || sourceName || optimizedPackage?.source.filename || "";
  const displaySourceBytes = file?.size || sourceBytes;
  const hasSource = Boolean(displaySourceName);
  const safeSaveBaseName =
    sanitizeDownloadBaseName(saveBaseName) ||
    baseNameFromFile(displaySourceName || "transcript");

  useEffect(() => {
    if (
      !runtimeChecked ||
      !file ||
      recommendation ||
      stage !== "idle" ||
      analysisStartingRef.current ||
      (shareMode && !shareAccessReady) ||
      (Boolean(runtime?.cloud_upload_enabled) && (!keyReady || !cloudConsent))
    ) {
      return;
    }
    void analyzeSelectedFile(file);
  }, [
    cloudConsent,
    file,
    keyReady,
    recommendation,
    runtime?.cloud_upload_enabled,
    runtimeChecked,
    shareAccessReady,
    shareMode,
    stage
  ]);

  useEffect(() => {
    if (
      !shareMode ||
      stage !== "ready" ||
      !canStart ||
      autoStartSuppressedRef.current
    ) {
      return;
    }
    const timer = window.setTimeout(() => void startTranscription(), 0);
    return () => window.clearTimeout(timer);
  }, [canStart, shareMode, stage]);

  useEffect(() => {
    if (stage !== "complete" || !transcript || !activeWorkflowId) return;
    const statusPrefix = serverExportStatus ? `${serverExportStatus} · ` : "";
    if (wasWorkflowAutoDownloaded(activeWorkflowId)) {
      setAutoDownloadStatus(`${statusPrefix}이 기기의 TXT 다운로드도 준비되었습니다.`);
      return;
    }

    let cancelled = false;
    let timer: number | null = null;
    let downloadStarting = false;
    let finished = false;
    let retryIndex = 0;
    const clearRetryTimer = () => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    };
    const scheduleDownload = (delayMs: number) => {
      if (cancelled || finished || document.visibilityState !== "visible") return;
      clearRetryTimer();
      timer = window.setTimeout(() => {
        timer = null;
        void downloadWhenVisible();
      }, delayMs);
    };
    const downloadWhenVisible = async () => {
      if (cancelled || finished) return;
      if (document.visibilityState !== "visible") {
        setAutoDownloadStatus(
          `${statusPrefix}화면을 다시 켜면 이 기기에도 TXT가 자동 다운로드됩니다.`
        );
        return;
      }
      if (downloadStarting) return;
      downloadStarting = true;
      try {
        await downloadApiFile(transcript.txt_url, `${safeSaveBaseName}.txt`);
        if (cancelled) return;
        markWorkflowAutoDownloaded(activeWorkflowId);
        finished = true;
        clearRetryTimer();
        setAutoDownloadStatus(`${statusPrefix}이 기기에도 TXT를 자동 다운로드했습니다.`);
      } catch (downloadError) {
        if (!cancelled) {
          if (isApiAuthenticationError(downloadError)) {
            finished = true;
            clearRetryTimer();
            requireAccessReconnect();
            setAutoDownloadStatus("비밀번호를 다시 확인하면 TXT를 자동 다운로드합니다.");
          } else if (
            isApiTransientError(downloadError) &&
            retryIndex < AUTO_DOWNLOAD_RETRY_DELAYS_MS.length
          ) {
            const retryDelay = AUTO_DOWNLOAD_RETRY_DELAYS_MS[retryIndex];
            retryIndex += 1;
            setAutoDownloadStatus(
              `TXT 자동 다운로드 연결을 다시 확인합니다. ${Math.ceil(retryDelay / 1000)}초 후 재시도합니다.`
            );
            scheduleDownload(retryDelay);
          } else {
            finished = true;
            clearRetryTimer();
            setAutoDownloadStatus(
              "TXT 자동 다운로드를 완료하지 못했습니다. TXT 버튼을 눌러 다시 받아 주세요."
            );
          }
        }
      } finally {
        downloadStarting = false;
      }
    };
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        scheduleDownload(350);
      } else {
        clearRetryTimer();
        if (!finished) {
          setAutoDownloadStatus(
            `${statusPrefix}화면을 다시 켜면 이 기기에도 TXT가 자동 다운로드됩니다.`
          );
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    if (document.visibilityState === "visible") {
      scheduleDownload(350);
    } else {
      setAutoDownloadStatus(
        `${statusPrefix}화면을 다시 켜면 이 기기에도 TXT가 자동 다운로드됩니다.`
      );
    }
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      clearRetryTimer();
    };
  }, [activeWorkflowId, safeSaveBaseName, serverExportStatus, stage, transcript]);

  useEffect(() => {
    const wakeLockApi = "wakeLock" in navigator ? navigator.wakeLock : null;
    let cancelled = false;

    async function acquireWakeLock() {
      if (
        !keepScreenAwake ||
        document.visibilityState !== "visible" ||
        wakeLockRef.current
      ) {
        return;
      }
      if (!wakeLockApi) {
        setWakeLockStatus("unavailable");
        return;
      }
      setWakeLockStatus("requesting");
      try {
        const sentinel = await wakeLockApi.request("screen");
        if (cancelled) {
          await sentinel.release();
          return;
        }
        wakeLockRef.current = sentinel;
        setWakeLockStatus("active");
        sentinel.addEventListener("release", () => {
          if (wakeLockRef.current === sentinel) {
            wakeLockRef.current = null;
            setWakeLockStatus("released");
          }
        });
      } catch {
        setWakeLockStatus("unavailable");
      }
    }

    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        void acquireWakeLock();
      }
    }

    if (keepScreenAwake) {
      void acquireWakeLock();
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      const sentinel = wakeLockRef.current;
      wakeLockRef.current = null;
      setWakeLockStatus("idle");
      if (sentinel && !sentinel.released) {
        void sentinel.release();
      }
    };
  }, [keepScreenAwake]);

  useEffect(() => {
    if (recordingState !== "recording") return;
    const updateElapsed = () => {
      const startedAt = recordingStartedAtRef.current;
      if (startedAt !== null) {
        setRecordingElapsedSec(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
      }
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 500);
    return () => window.clearInterval(timer);
  }, [recordingState]);

  function stopProgressPolling() {
    pollingVersionRef.current += 1;
    if (workflowTimerRef.current !== null) {
      window.clearInterval(workflowTimerRef.current);
      workflowTimerRef.current = null;
    }
  }

  async function refreshWorkflow(currentWorkflowId: string, pollingVersion: number) {
    if (
      pollingVersion !== pollingVersionRef.current ||
      pollingInFlightRef.current === pollingVersion
    ) return;
    pollingInFlightRef.current = pollingVersion;
    try {
      const workflow = await getTranscriptionWorkflow(currentWorkflowId);
      if (pollingVersion !== pollingVersionRef.current) return;
      setError(null);
      setAccessNeedsReconnect(false);
      applyWorkflowStatus(workflow);
    } catch (workflowError) {
      if (pollingVersion !== pollingVersionRef.current) return;
      const message =
        workflowError instanceof Error
          ? workflowError.message
          : "서버 작업 상태를 확인하지 못했습니다.";
      if (/Workflow not found/i.test(message)) {
        setStage("failed");
        setError(message);
        stopProgressPolling();
        return;
      }
      if (isApiAuthenticationError(workflowError)) {
        requireAccessReconnect();
        return;
      }
      setError("연결을 다시 확인하고 있습니다. PC의 전사 작업은 계속 진행됩니다.");
    } finally {
      if (pollingInFlightRef.current === pollingVersion) {
        pollingInFlightRef.current = null;
      }
    }
  }

  function beginWorkflowPolling(currentWorkflowId: string) {
    stopProgressPolling();
    const pollingVersion = pollingVersionRef.current;
    void refreshWorkflow(currentWorkflowId, pollingVersion);
    workflowTimerRef.current = window.setInterval(
      () => void refreshWorkflow(currentWorkflowId, pollingVersion),
      1000
    );
  }

  function applyWorkflowStatus(workflow: TranscriptionWorkflowStatus) {
    setActiveWorkflowId(workflow.workflow_id);
    if (workflow.package) {
      setOptimizedPackage(workflow.package);
      setStagedUploadId(null);
      setSourceName((current) => current || workflow.package?.source.filename || "");
      setRecommendation(
        (current) =>
          current || {
            source: workflow.package!.source,
            original_bytes: sourceBytes,
            recommendation: workflow.package!.recommendation
          }
      );
      setSaveBaseName(
        (current) => current || baseNameFromFile(workflow.package?.source.filename || "transcript")
      );
    }
    if (workflow.transcription_progress) {
      setTranscriptionProgress(workflow.transcription_progress);
    }
    if (workflow.status === "queued" || workflow.status === "optimizing") {
      setStage("optimizing");
      return;
    }
    if (workflow.status === "transcribing") {
      setStage("transcribing");
      return;
    }
    if (workflow.status === "failed") {
      setStage("failed");
      setError(workflowErrorMessage(workflow.error));
      stopProgressPolling();
      return;
    }
    if (workflow.status === "complete" && workflow.transcript) {
      setServerExportStatus(
        workflow.auto_exported
          ? "서버 PC Downloads\\PhoneScribe 저장 완료"
          : workflow.auto_export_error || null
      );
      setTranscript(workflow.transcript);
      if (namingMode === "recommended") {
        setSaveBaseName(workflow.transcript.suggested_filename);
      }
      setStage("complete");
      stopProgressPolling();
    }
  }

  async function saveDefaultGeminiKey() {
    const apiKey = geminiApiKey.trim();
    const passcode = sharePasscode.trim();
    if (apiKey.length < 20 || passcode.length < 4 || savingShareKey) return;
    setSavingShareKey(true);
    setError(null);
    try {
      await configureGeminiShareKey(apiKey, passcode);
      setRuntime(await getRuntime());
      setGeminiApiKey("");
      setShowKeySetup(false);
      setShareAccessReady(true);
      setShareStatus("기본 키 저장 완료 · 바로 녹음하면 TXT 저장까지 자동 진행합니다.");
    } catch (saveError) {
      setShareAccessReady(false);
      setError(
        saveError instanceof Error ? saveError.message : "기본 API key를 저장하지 못했습니다."
      );
    } finally {
      setSavingShareKey(false);
    }
  }

  async function confirmShareAccess() {
    const selectionVersion = selectionVersionRef.current;
    const passcode = sharePasscode.trim();
    if (
      passcode.length < 4 ||
      verifyingShareAccess ||
      !runtime?.gemini_share_enabled ||
      !runtime.gemini_share_ready
    ) {
      return;
    }

    clearApiAccessToken();
    setVerifyingShareAccess(true);
    setShareAccessReady(false);
    setShareStatus("비밀번호 확인 중");
    setError(null);
    try {
      const result = await verifyGeminiSharePasscode(passcode);
      if (!result.valid || !result.key_ready) {
        setShareStatus("서버 PC에서 기본 키 등록이 필요합니다.");
        return;
      }

      setShareAccessReady(true);
      setAccessNeedsReconnect(false);
      if (selectionVersion !== selectionVersionRef.current) {
        setShareStatus("확인 완료 · 새로 시작한 녹음으로 진행합니다.");
        return;
      }
      if (activeWorkflowId) {
        setShareStatus("연결 복구 완료 · 작업 결과를 불러오고 있습니다.");
        setError(null);
        beginWorkflowPolling(activeWorkflowId);
        return;
      }
      if (file) {
        setShareStatus("확인 완료 · 전사부터 PC TXT 저장까지 자동 진행합니다.");
      } else {
        setShareStatus("확인 완료 · 바로 녹음을 시작하면 끝까지 자동 진행합니다.");
      }
    } catch (verificationError) {
      setShareAccessReady(false);
      setShareStatus(
        verificationError instanceof Error
          ? verificationError.message
          : "공유 비밀번호를 확인하지 못했습니다."
      );
    } finally {
      setVerifyingShareAccess(false);
    }
  }

  async function selectFile(selected: File) {
    resetFile();
    setFile(selected);
    setSourceName(selected.name);
    setSourceBytes(selected.size);
    setSaveBaseName(baseNameFromFile(selected.name));
  }

  async function startDirectRecording(replaceCurrent = false) {
    if (!recordingSupported || recordingActive || (busy && !replaceCurrent)) return;
    setRecordingState("starting");
    setRecordingElapsedSec(0);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      recordingStreamRef.current = stream;
      const mimeType = preferredRecordingMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 64000 })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recordingChunksRef.current = [];
      recordingStartedAtRef.current = Date.now();

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        setError("녹음이 중단되었습니다. 마이크 권한과 브라우저 상태를 확인하세요.");
        finishDirectRecording(null);
      };
      recorder.onstop = () => {
        const chunks = recordingChunksRef.current;
        const recordedType = recorder.mimeType || mimeType || chunks[0]?.type || "audio/webm";
        const recording = chunks.length ? new Blob(chunks, { type: recordedType }) : null;
        finishDirectRecording(recording);
      };
      recorder.start(1000);
      if (replaceCurrent) resetFile();
      setRecordingState("recording");
    } catch (recordingError) {
      stopRecordingStream();
      setRecordingState("idle");
      setError(recordingPermissionMessage(recordingError));
    }
  }

  function stopDirectRecording() {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive" || recordingState === "stopping") return;
    setRecordingState("stopping");
    try {
      recorder.requestData();
    } catch {
      // Some mobile browsers flush automatically when stop() is called.
    }
    recorder.stop();
  }

  function finishDirectRecording(recording: Blob | null) {
    mediaRecorderRef.current = null;
    recordingStartedAtRef.current = null;
    recordingChunksRef.current = [];
    stopRecordingStream();
    setRecordingState("idle");
    if (!recording || recording.size === 0) {
      setError("녹음된 음성이 없습니다. 다시 녹음해 주세요.");
      return;
    }
    const extension = recordingExtension(recording.type);
    const recordedFile = new File(
      [recording],
      `PhoneScribe_${recordingTimestampForName(new Date())}.${extension}`,
      { type: recording.type, lastModified: Date.now() }
    );
    void selectFile(recordedFile);
  }

  function stopRecordingStream() {
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
  }

  function clearRecordingRetryTimer() {
    if (recordingRetryTimerRef.current !== null) {
      window.clearTimeout(recordingRetryTimerRef.current);
      recordingRetryTimerRef.current = null;
    }
  }

  function clearRecordingRetryState() {
    clearRecordingRetryTimer();
    recordingRetryAttemptRef.current = 0;
    setRecordingRetryScheduled(false);
    setRecordingRetryNotice(null);
    setSameRecordingRetryAvailable(false);
  }

  function scheduleRecordingRetry(selected: File, selectionVersion: number) {
    if (selectionVersion !== selectionVersionRef.current) return;
    clearRecordingRetryTimer();
    setStage("failed");
    setError(null);
    setCloudUploadProgress(null);
    setSameRecordingRetryAvailable(true);

    const retryIndex = recordingRetryAttemptRef.current;
    if (retryIndex >= RECORDING_UPLOAD_RETRY_DELAYS_MS.length) {
      setRecordingRetryScheduled(false);
      setRecordingRetryNotice(
        "자동 재시도를 마쳤습니다. 녹음은 이 기기에 그대로 있습니다. 같은 녹음 다시 전송을 눌러 주세요."
      );
      return;
    }

    const delayMs = RECORDING_UPLOAD_RETRY_DELAYS_MS[retryIndex];
    recordingRetryAttemptRef.current = retryIndex + 1;
    setRecordingRetryScheduled(true);
    setRecordingRetryNotice(
      `연결이 잠시 끊겼습니다. 녹음은 이 기기에 그대로 있습니다. ${Math.ceil(
        delayMs / 1000
      )}초 후 같은 녹음을 자동으로 다시 전송합니다. (${retryIndex + 1}/${
        RECORDING_UPLOAD_RETRY_DELAYS_MS.length
      })`
    );
    recordingRetryTimerRef.current = window.setTimeout(() => {
      recordingRetryTimerRef.current = null;
      if (selectionVersion !== selectionVersionRef.current) return;
      setRecordingRetryScheduled(false);
      setRecordingRetryNotice("같은 녹음을 다시 전송하고 있습니다.");
      void analyzeSelectedFile(selected, true);
    }, delayMs);
  }

  function retrySameRecording() {
    if (!file || analysisStartingRef.current || !sameRecordingRetryAvailable) return;
    clearRecordingRetryTimer();
    recordingRetryAttemptRef.current = 0;
    setRecordingRetryScheduled(false);
    setRecordingRetryNotice("같은 녹음을 지금 다시 전송하고 있습니다.");
    void analyzeSelectedFile(file, true);
  }

  async function analyzeSelectedFile(selected: File, retrying = false) {
    if (analysisStartingRef.current) return;
    const selectionVersion = selectionVersionRef.current;
    const transferController = new AbortController();
    let remoteUploadAccepted = false;
    analysisStartingRef.current = true;
    uploadAbortRef.current = transferController;
    if (retrying) {
      clearRecordingRetryTimer();
      setRecordingRetryScheduled(false);
      setRecordingRetryNotice("같은 녹음을 다시 전송하고 있습니다.");
    } else {
      clearRecordingRetryState();
    }
    setStage("analyzing");
    setError(null);
    setCloudUploadNotice(null);
    setCloudUploadProgress(null);
    try {
      if (runtime?.cloud_upload_enabled) {
        try {
          const descriptor = await createCloudUploadDescriptor(selected);
          if (selectionVersion !== selectionVersionRef.current) return;
          remoteUploadAccepted = true;
          clearRecordingRetryState();
          await uploadCloudRecording(
            selected,
            descriptor,
            (uploadedBytes, totalBytes) => {
              if (selectionVersion !== selectionVersionRef.current) return;
              const progress = totalBytes > 0 ? uploadedBytes / totalBytes : 0;
              setCloudUploadProgress(Math.min(1, Math.max(0, progress)));
            },
            transferController.signal,
            (partNumber, attempt, delayMs) => {
              if (selectionVersion !== selectionVersionRef.current) return;
              setCloudUploadNotice(
                `연결을 다시 확인하고 있습니다. ${Math.ceil(
                  delayMs / 1000
                )}초 후 ${partNumber + 1}번 조각을 자동 재전송합니다. (${attempt}번째 전송)`
              );
            }
          );
          if (selectionVersion !== selectionVersionRef.current) return;
          const completed = await completeCloudRecordingUpload(descriptor.recording_id);
          if (selectionVersion !== selectionVersionRef.current) return;
          setCloudRecordingId(completed.recording_id);
          setCloudUploadProgress(1);
          setCloudUploadNotice(
            "업로드 완료 · 서버에 전사 작업을 접수하고 있습니다. 화면을 켜 두세요."
          );
          savePersistedWorkflow({
            version: 1,
            workflowId: null,
            stagedUploadId: null,
            cloudRecordingId: completed.recording_id,
            sourceName: selected.name,
            sourceBytes: selected.size,
            recommendation: null,
            optimizedPackage: null,
            saveBaseName: baseNameFromFile(selected.name)
          });

          workflowStartingRef.current = true;
          try {
            const workflow = await startTranscriptionWorkflow(
              geminiOptimizerOptions(selected),
              {
                cloudRecordingId: completed.recording_id,
                apiKey: shareMode ? undefined : geminiApiKey.trim() || undefined,
                sharePasscode: shareMode ? sharePasscode.trim() : undefined
              }
            );
            if (selectionVersion !== selectionVersionRef.current) return;
            setActiveWorkflowId(workflow.workflow_id);
            setWorkflowUrl(workflow.workflow_id);
            savePersistedWorkflow({
              version: 1,
              workflowId: workflow.workflow_id,
              stagedUploadId: null,
              cloudRecordingId: completed.recording_id,
              sourceName: selected.name,
              sourceBytes: selected.size,
              recommendation: null,
              optimizedPackage: null,
              saveBaseName: baseNameFromFile(selected.name)
            });
            setCloudUploadNotice(
              "서버 작업 접수 완료 · 이제 화면을 꺼도 전사와 저장이 계속됩니다."
            );
            setStage("optimizing");
            beginWorkflowPolling(workflow.workflow_id);
          } catch (workflowError) {
            if (selectionVersion !== selectionVersionRef.current) return;
            if (isApiAuthenticationError(workflowError)) {
              setStage("ready");
              requireAccessReconnect();
              return;
            }
            setStage("failed");
            setCloudUploadNotice(
              "녹음 업로드 완료 · 전사 작업을 다시 접수할 수 있습니다."
            );
            setError(
              workflowError instanceof Error
                ? workflowError.message
                : "서버에 전사 작업을 접수하지 못했습니다."
            );
          } finally {
            if (selectionVersion === selectionVersionRef.current) {
              workflowStartingRef.current = false;
            }
          }
          return;
        } catch (cloudUploadError) {
          if (
            selectionVersion !== selectionVersionRef.current ||
            isAbortError(cloudUploadError)
          ) {
            return;
          }
          if (isApiAuthenticationError(cloudUploadError)) {
            setStage("idle");
            requireAccessReconnect();
            return;
          }
          if (isApiTransientError(cloudUploadError)) {
            if (!remoteUploadAccepted) {
              scheduleRecordingRetry(selected, selectionVersion);
            } else {
              setStage("failed");
              setCloudUploadProgress(null);
              setSameRecordingRetryAvailable(false);
              setRecordingRetryScheduled(false);
              setRecordingRetryNotice(
                "같은 업로드 조각을 자동 재시도했지만 연결을 복구하지 못했습니다. 새 녹음 전까지 원본은 이 기기에 남아 있습니다."
              );
              setError(
                cloudUploadError instanceof Error
                  ? cloudUploadError.message
                  : "클라우드 업로드 연결을 복구하지 못했습니다."
              );
            }
            return;
          }
          setCloudUploadProgress(null);
          setCloudUploadNotice(
            "클라우드 업로드를 사용할 수 없어 PC 직접 업로드로 자동 전환했습니다."
          );
        }
      }

      const analysis = await analyzeOptimizer(
        geminiOptimizerOptions(selected),
        "auto",
        !shareMode,
        transferController.signal
      );
      if (selectionVersion !== selectionVersionRef.current) return;
      remoteUploadAccepted = true;
      clearRecordingRetryState();
      const analyzedRecommendation: OptimizerRecommendationResponse = {
        source: analysis.source,
        original_bytes: analysis.original_bytes,
        recommendation: analysis.recommendation
      };
      setRecommendation(analyzedRecommendation);
      setStagedUploadId(analysis.upload_id);
      setScan(analysis.quick_scan);
      savePersistedWorkflow({
        version: 1,
        workflowId: null,
        stagedUploadId: analysis.upload_id,
        cloudRecordingId: null,
        sourceName: selected.name,
        sourceBytes: selected.size,
        recommendation: analyzedRecommendation,
        optimizedPackage: null,
        saveBaseName: baseNameFromFile(selected.name)
      });

      if (shareMode && keyReady && cloudConsent && !autoStartSuppressedRef.current) {
        setCloudUploadNotice(
          "PC 직접 업로드 완료 · 서버에 전사 작업을 접수하고 있습니다. 화면을 켜 두세요."
        );
        workflowStartingRef.current = true;
        try {
          const workflow = await startTranscriptionWorkflow(
            geminiOptimizerOptions(undefined),
            {
              uploadId: analysis.upload_id,
              sharePasscode: sharePasscode.trim()
            }
          );
          if (selectionVersion !== selectionVersionRef.current) return;
          setActiveWorkflowId(workflow.workflow_id);
          setWorkflowUrl(workflow.workflow_id);
          savePersistedWorkflow({
            version: 1,
            workflowId: workflow.workflow_id,
            stagedUploadId: analysis.upload_id,
            cloudRecordingId: null,
            sourceName: selected.name,
            sourceBytes: selected.size,
            recommendation: analyzedRecommendation,
            optimizedPackage: null,
            saveBaseName: baseNameFromFile(selected.name)
          });
          setCloudUploadNotice(
            "서버 작업 접수 완료 · 이제 화면을 꺼도 전사와 저장이 계속됩니다."
          );
          setStage("optimizing");
          beginWorkflowPolling(workflow.workflow_id);
        } catch (workflowError) {
          if (selectionVersion !== selectionVersionRef.current) return;
          if (isApiAuthenticationError(workflowError)) {
            setCloudUploadNotice(
              "PC 직접 업로드 완료 · 비밀번호 확인 후 전사 작업을 다시 접수합니다."
            );
            setStage("ready");
            requireAccessReconnect();
            return;
          }
          setStage("failed");
          setCloudUploadNotice(
            "PC 직접 업로드 완료 · 전사 작업을 다시 접수할 수 있습니다."
          );
          setError(
            workflowError instanceof Error
              ? workflowError.message
              : "서버에 전사 작업을 접수하지 못했습니다."
          );
        } finally {
          if (selectionVersion === selectionVersionRef.current) {
            workflowStartingRef.current = false;
          }
        }
        return;
      }

      setStage("ready");
    } catch (analysisError) {
      if (selectionVersion !== selectionVersionRef.current) return;
      if (isApiAuthenticationError(analysisError)) {
        setStage("idle");
        requireAccessReconnect();
        return;
      }
      if (isApiTransientError(analysisError) && !remoteUploadAccepted) {
        scheduleRecordingRetry(selected, selectionVersion);
        return;
      }
      setStage("failed");
      setRecordingRetryScheduled(false);
      setSameRecordingRetryAvailable(!remoteUploadAccepted);
      setRecordingRetryNotice(
        remoteUploadAccepted
          ? null
          : "녹음은 이 기기에 그대로 있습니다. 같은 녹음 다시 전송을 눌러 즉시 다시 시도할 수 있습니다."
      );
      setError(
        analysisError instanceof Error
          ? analysisError.message
          : "녹음 파일을 분석하지 못했습니다."
      );
    } finally {
      if (uploadAbortRef.current === transferController) {
        uploadAbortRef.current = null;
      }
      if (selectionVersion === selectionVersionRef.current) analysisStartingRef.current = false;
    }
  }

  async function startTranscription() {
    if (!canStart || workflowStartingRef.current) return;
    const selectionVersion = selectionVersionRef.current;
    const enqueueingUploadedSource = Boolean(
      !optimizedPackage && (cloudRecordingId || stagedUploadId)
    );
    workflowStartingRef.current = true;
    stopProgressPolling();
    setError(null);
    setTranscript(null);
    setCopied(false);
    setTranscriptionProgress(null);
    setAutoDownloadStatus(null);
    setServerExportStatus(null);

    try {
      if (enqueueingUploadedSource) {
        setStage("analyzing");
        setCloudUploadNotice(
          `${cloudRecordingId ? "클라우드" : "PC 직접"} 업로드 완료 · 서버에 전사 작업을 접수하고 있습니다. 화면을 켜 두세요.`
        );
      } else {
        setStage(optimizedPackage ? "transcribing" : "optimizing");
      }
      const workflow = await startTranscriptionWorkflow(
        geminiOptimizerOptions(file ?? undefined),
        {
          uploadId: optimizedPackage ? undefined : stagedUploadId ?? undefined,
          packageId: optimizedPackage?.id,
          cloudRecordingId:
            optimizedPackage || stagedUploadId ? undefined : cloudRecordingId ?? undefined,
          apiKey: shareMode ? undefined : geminiApiKey.trim() || undefined,
          sharePasscode: shareMode ? sharePasscode.trim() : undefined
        }
      );
      if (selectionVersion !== selectionVersionRef.current) return;
      setActiveWorkflowId(workflow.workflow_id);
      setWorkflowUrl(workflow.workflow_id);
      savePersistedWorkflow({
        version: 1,
        workflowId: workflow.workflow_id,
        stagedUploadId,
        cloudRecordingId,
        sourceName: displaySourceName,
        sourceBytes: displaySourceBytes,
        recommendation,
        optimizedPackage,
        saveBaseName
      });
      if (enqueueingUploadedSource) {
        setCloudUploadNotice(
          "서버 작업 접수 완료 · 이제 화면을 꺼도 전사와 저장이 계속됩니다."
        );
        setStage("optimizing");
      }
      beginWorkflowPolling(workflow.workflow_id);
    } catch (transcriptionError) {
      if (selectionVersion !== selectionVersionRef.current) return;
      stopProgressPolling();
      if (isApiAuthenticationError(transcriptionError)) {
        if (enqueueingUploadedSource) {
          setCloudUploadNotice(
            "업로드 완료 · 비밀번호 확인 후 전사 작업을 다시 접수합니다."
          );
        }
        setStage("ready");
        requireAccessReconnect();
        return;
      }
      setStage("failed");
      if (enqueueingUploadedSource) {
        setCloudUploadNotice("업로드 완료 · 전사 작업을 다시 접수할 수 있습니다.");
      }
      setError(
        transcriptionError instanceof Error
          ? transcriptionError.message
          : "전사 작업을 완료하지 못했습니다."
      );
    } finally {
      if (selectionVersion === selectionVersionRef.current) workflowStartingRef.current = false;
    }
  }

  async function createPackageOnly() {
    if ((!file && !stagedUploadId) || !recommendation || busy) return;
    const selectionVersion = selectionVersionRef.current;
    autoStartSuppressedRef.current = true;
    setError(null);
    setStage("optimizing");
    try {
      const packageResult = await createOptimizerPackage(
        geminiOptimizerOptions(file ?? undefined),
        stagedUploadId ?? undefined
      );
      if (selectionVersion !== selectionVersionRef.current) return;
      setOptimizedPackage(packageResult);
      setStagedUploadId(null);
      setStage("ready");
      savePersistedWorkflow({
        version: 1,
        workflowId: null,
        stagedUploadId: null,
        cloudRecordingId: null,
        sourceName: displaySourceName,
        sourceBytes: displaySourceBytes,
        recommendation,
        optimizedPackage: packageResult,
        saveBaseName
      });
    } catch (packageError) {
      if (selectionVersion !== selectionVersionRef.current) return;
      setStage("failed");
      setError(
        packageError instanceof Error
          ? packageError.message
          : "최적화 파일을 만들지 못했습니다."
      );
    }
  }

  function resetFile() {
    selectionVersionRef.current += 1;
    uploadAbortRef.current?.abort();
    uploadAbortRef.current = null;
    clearRecordingRetryState();
    stopProgressPolling();
    clearPersistedWorkflow();
    setWorkflowUrl(null);
    autoStartSuppressedRef.current = false;
    analysisStartingRef.current = false;
    workflowStartingRef.current = false;
    setAccessNeedsReconnect(false);
    setShareStatus(null);
    setFile(null);
    setSourceName("");
    setSourceBytes(0);
    setScan(null);
    setRecommendation(null);
    setStagedUploadId(null);
    setCloudRecordingId(null);
    setCloudUploadProgress(null);
    setCloudUploadNotice(null);
    setOptimizedPackage(null);
    setTranscript(null);
    setError(null);
    setCopied(false);
    setNamingMode("original");
    setSaveBaseName("");
    setTranscriptionProgress(null);
    setActiveWorkflowId(null);
    setAutoDownloadStatus(null);
    setServerExportStatus(null);
    setStage("idle");
  }

  async function copyTranscript() {
    if (!transcript) return;
    await navigator.clipboard.writeText(transcript.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function downloadResult(url: string, downloadName: string) {
    const selectionVersion = selectionVersionRef.current;
    setError(null);
    try {
      await downloadApiFile(url, downloadName);
    } catch (downloadError) {
      if (selectionVersion !== selectionVersionRef.current) return;
      if (isApiAuthenticationError(downloadError)) {
        requireAccessReconnect();
        return;
      }
      setError(
        downloadError instanceof Error
          ? downloadError.message
          : "파일을 다운로드하지 못했습니다."
      );
    }
  }

  function requireAccessReconnect() {
    clearApiAccessToken();
    setShareAccessReady(false);
    setAccessNeedsReconnect(true);
    setShareStatus("보안을 위해 공유 비밀번호를 다시 확인해 주세요.");
    setError(
      activeWorkflowId
        ? "PC 작업은 보존되어 있습니다. 비밀번호를 다시 확인하면 결과를 바로 가져옵니다."
        : "인증 세션이 만료되었습니다. 비밀번호를 다시 확인하면 자동으로 이어집니다."
    );
    stopProgressPolling();
  }

  function chooseNamingMode(mode: NamingMode) {
    setNamingMode(mode);
    if (mode === "original" && displaySourceName) {
      setSaveBaseName(baseNameFromFile(displaySourceName));
    }
    if (mode === "recommended" && transcript) {
      setSaveBaseName(transcript.suggested_filename);
    }
  }

  return (
    <main className="app-page">
      <div className="app-shell">
        <header className="app-header">
          <div className="brand-mark" aria-hidden="true">
            <Smartphone size={21} />
          </div>
          <div>
            <p className="eyebrow">KOREAN + ENGLISH / PHONE RECORDINGS</p>
            <h1>Phone Scribe</h1>
            <p className="tagline">휴대폰에서 바로 녹음하면 전사문을 만듭니다.</p>
          </div>
        </header>

        <WorkflowProgress stage={stage} hasPackage={Boolean(optimizedPackage)} />

        {error && (
          <div className="error-banner" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {accessNeedsReconnect && shareMode && (
          <form
            className="workflow-reconnect"
            onSubmit={(event) => {
              event.preventDefault();
              void confirmShareAccess();
            }}
          >
            <div>
              <strong>{activeWorkflowId ? "전사 결과 연결 복구" : "Gemini 연결 복구"}</strong>
              <span>
                {activeWorkflowId
                  ? "PC의 작업은 그대로 있습니다. 비밀번호만 다시 확인하세요."
                  : "현재 녹음을 유지한 채 자동으로 다시 시작합니다."}
              </span>
            </div>
            <label className="secret-input single">
              <input
                type="password"
                value={sharePasscode}
                onChange={(event) => setSharePasscode(event.target.value)}
                placeholder="공유 비밀번호"
                inputMode="numeric"
                maxLength={12}
                autoComplete="off"
                spellCheck={false}
                aria-label="공유 비밀번호 다시 입력"
              />
            </label>
            <button
              className="secondary-button reconnect-button"
              type="submit"
              disabled={
                verifyingShareAccess ||
                sharePasscode.trim().length < 4 ||
                !runtime?.gemini_share_ready
              }
            >
              {verifyingShareAccess ? (
                <Loader2 className="spin" size={17} />
              ) : (
                <ShieldCheck size={17} />
              )}
              {activeWorkflowId ? "확인하고 결과 받기" : "확인하고 자동 재개"}
            </button>
          </form>
        )}

        <section className="flow-section">
          <SectionTitle index="01" title="녹음" note="바로 녹음" />
          {recordingActive ? (
            <div className="recording-panel" aria-live="polite">
              <span className="recording-indicator" aria-hidden="true" />
              <div>
                <strong>
                  {recordingState === "starting"
                    ? "마이크 연결 중"
                    : recordingState === "stopping"
                      ? "녹음 파일 만드는 중"
                      : "녹음 중"}
                </strong>
                <span>{formatClock(recordingElapsedSec)}</span>
                <small>
                  {recordingWakeLockMessage(wakeLockStatus)} 녹음을 마치면 업로드·전사가
                  자동으로 이어집니다.
                </small>
              </div>
              <button
                className="record-stop-button"
                type="button"
                disabled={recordingState !== "recording"}
                onClick={stopDirectRecording}
              >
                {recordingState === "stopping" ? (
                  <Loader2 className="spin" size={18} />
                ) : (
                  <Square size={17} fill="currentColor" />
                )}
                녹음 종료 후 자동 전사
              </button>
            </div>
          ) : !hasSource ? (
            <div className="drop-zone recording-only-zone">
              <Mic size={28} />
              <strong>지금 바로 녹음을 시작하세요</strong>
              <span>녹음 종료 후 업로드·전사·TXT 저장까지 자동 진행됩니다.</span>
              <button
                className="direct-record-button"
                type="button"
                disabled={!recordingSupported}
                onClick={() => void startDirectRecording(Boolean(activeWorkflowId))}
              >
                <Mic size={18} />
                {recordingSupported ? "바로 녹음 시작" : "이 브라우저는 직접 녹음 미지원"}
              </button>
            </div>
          ) : (
            <div className="selected-file">
              <FileAudio size={24} />
              <div>
                <strong>{displaySourceName}</strong>
                <span>
                  {stage === "analyzing" && cloudUploadProgress !== null
                    ? `클라우드 업로드 ${Math.round(cloudUploadProgress * 100)}%`
                    : stage === "analyzing"
                      ? "길이, 언어, 전사 방식을 확인하고 있습니다."
                    : recommendation
                      ? "분석 완료"
                      : cloudRecordingId
                        ? "클라우드 업로드 완료"
                      : shareMode
                        ? "비밀번호 확인 후 자동 업로드됩니다."
                        : "업로드 준비 중"}
                </span>
              </div>
              <button
                className="secondary-button change-file-button"
                type="button"
                disabled={!recordingSupported}
                onClick={() => void startDirectRecording(true)}
                title="새 녹음 시작"
                aria-label="새 녹음 시작"
              >
                <Mic size={17} />
                새 녹음 시작
              </button>
            </div>
          )}
          {cloudUploadProgress !== null && stage === "analyzing" && (
            <div className="transcription-progress" aria-live="polite">
              <div className="progress-heading">
                <span>녹음 업로드</span>
                <strong>{Math.round(cloudUploadProgress * 100)}%</strong>
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-label="녹음 클라우드 업로드 진행률"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(cloudUploadProgress * 100)}
              >
                <span style={{ width: `${Math.round(cloudUploadProgress * 100)}%` }} />
              </div>
              <div className="progress-meta">
                <span>연결이 흔들리면 현재 조각만 다시 보내고 이어집니다.</span>
              </div>
            </div>
          )}
          {cloudUploadNotice && (
            <p className="cloud-upload-notice" role="status">
              {cloudUploadNotice}
            </p>
          )}
          {recordingRetryNotice && (
            <p className="recording-retry-notice" role="status">
              {recordingRetryNotice}
            </p>
          )}
          {stage === "failed" && file && sameRecordingRetryAvailable && (
            <button
              className="secondary-button same-recording-retry-button"
              type="button"
              onClick={retrySameRecording}
            >
              <RefreshCw size={17} />
              같은 녹음 다시 전송
            </button>
          )}
          {recommendation && hasSource && (
            <div className="analysis-band">
              <InfoItem
                label="녹음 길이"
                value={formatClock(recommendation.source.duration_sec)}
              />
              <InfoItem label="원본 크기" value={formatBytes(displaySourceBytes)} />
              <InfoItem
                label="처리 단위"
                value={`${recommendation.recommendation.chunk_count}개`}
              />
              <InfoItem
                label="예상 용량"
                value={`${recommendation.recommendation.projected_size_mb.toFixed(1)} MB`}
              />
              <InfoItem
                label="언어"
                value={languageLabel(scan?.detected_language)}
              />
            </div>
          )}

          {recommendation && (
            <p className="plan-note">
              16 kHz mono MP3 · 약 30분 단위 · 발화 사이의 자연스러운 구간에서 분할
            </p>
          )}
          <p className="recording-limit-note">
            녹음 중에는 화면을 켜 둔 채 자동 잠금 방지를 사용합니다. 녹음 종료·업로드 후
            서버 작업 접수가 완료되면 화면을 꺼도 전사와 저장을 계속합니다. 전원 버튼으로 화면을 끈 상태의
            녹음을 보장하려면 Android 전용 앱이 필요합니다.
          </p>
        </section>

        <section className="flow-section">
          <SectionTitle
            index="02"
            title="Gemini 연결"
            note={
              shareMode
                ? "공유 비밀번호로 연결"
                : serverKeyReady
                  ? "서버 키 연결됨"
                  : "무료 AI Studio 키"
            }
          />

          {shareMode ? (
            <div className="share-access">
              <form
                className="passcode-confirm-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  void confirmShareAccess();
                }}
              >
                <label className="key-field">
                  <span>
                    <KeyRound size={16} />
                    공유 비밀번호
                    <small>한 번 확인하면 TXT 저장까지 자동 진행됩니다.</small>
                  </span>
                  <span className="secret-input single">
                    <input
                      type="password"
                      value={sharePasscode}
                      onChange={(event) => {
                        clearApiAccessToken();
                        setSharePasscode(event.target.value);
                        setShareAccessReady(false);
                        setShareStatus(null);
                      }}
                      placeholder="4자리 이상"
                      inputMode="numeric"
                      maxLength={12}
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </span>
                </label>
                <button
                  className="secondary-button passcode-confirm-button"
                  type="submit"
                  disabled={
                    verifyingShareAccess ||
                    sharePasscode.trim().length < 4 ||
                    !runtime?.gemini_share_ready
                  }
                >
                  {verifyingShareAccess ? (
                    <Loader2 className="spin" size={17} />
                  ) : (
                    <ShieldCheck size={17} />
                  )}
                  확인하고 자동 전사
                </button>
              </form>

              {shareStatus && (
                <p className={shareAccessReady ? "share-status connected" : "share-status"}>
                  {shareAccessReady && <CheckCircle2 size={15} />}
                  {shareStatus}
                </p>
              )}

              {runtime?.local_admin && runtime.gemini_share_ready && !showKeySetup && (
                <button
                  className="text-link"
                  type="button"
                  onClick={() => setShowKeySetup(true)}
                >
                  기본 키 변경
                </button>
              )}

              {runtime?.local_admin &&
                (!runtime.gemini_share_ready || showKeySetup) && (
                  <div className="share-key-setup">
                    <div>
                      <strong>기본 Gemini API key</strong>
                      <p>이 서버 PC에 암호화해 저장하고 공유 사용자에게는 노출하지 않습니다.</p>
                    </div>
                    <label className="key-field">
                      <span>
                        <KeyRound size={16} />
                        API key
                      </span>
                      <span className="secret-input">
                        <input
                          type={showApiKey ? "text" : "password"}
                          value={geminiApiKey}
                          onChange={(event) => setGeminiApiKey(event.target.value)}
                          placeholder="AI Studio API key"
                          autoComplete="off"
                          spellCheck={false}
                        />
                        <button
                          className="icon-button"
                          type="button"
                          onClick={() => setShowApiKey((current) => !current)}
                          title={showApiKey ? "API key 숨기기" : "API key 보기"}
                          aria-label={showApiKey ? "API key 숨기기" : "API key 보기"}
                        >
                          {showApiKey ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                      </span>
                    </label>
                    <button
                      className="secondary-button"
                      type="button"
                      disabled={
                        savingShareKey ||
                        sharePasscode.trim().length < 4 ||
                        geminiApiKey.trim().length < 20
                      }
                      onClick={() => void saveDefaultGeminiKey()}
                    >
                      {savingShareKey ? (
                        <Loader2 className="spin" size={17} />
                      ) : (
                        <ShieldCheck size={17} />
                      )}
                      이 PC의 기본 키로 저장
                    </button>
                  </div>
                )}

              {!runtime?.gemini_share_ready && !runtime?.local_admin && (
                <p className="share-status">서버 PC에서 기본 API key 등록이 필요합니다.</p>
              )}
            </div>
          ) : serverKeyReady ? (
            <div className="connected-line">
              <CheckCircle2 size={18} />
              <span>{runtime?.gemini_model} 준비됨</span>
            </div>
          ) : (
            <div className="key-layout">
              <label className="key-field">
                <span>
                  <KeyRound size={16} />
                  API key
                </span>
                <span className="secret-input">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={geminiApiKey}
                    onChange={(event) => setGeminiApiKey(event.target.value)}
                    placeholder="AI Studio API key"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => setShowApiKey((current) => !current)}
                    title={showApiKey ? "API key 숨기기" : "API key 보기"}
                    aria-label={showApiKey ? "API key 숨기기" : "API key 보기"}
                  >
                    {showApiKey ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </span>
              </label>
              <a
                className="text-link"
                href="https://aistudio.google.com/apikey"
                target="_blank"
                rel="noreferrer"
              >
                무료 키 만들기
                <ExternalLink size={14} />
              </a>
            </div>
          )}

          <label className="consent-line">
            <input
              type="checkbox"
              checked={cloudConsent}
              disabled={busy}
              onChange={(event) => setCloudConsent(event.target.checked)}
            />
            <span>
              최적화된 오디오를 Google Gemini로 전송합니다.
              <small>무료 API 콘텐츠는 Google 제품 개선에 사용될 수 있습니다.</small>
            </span>
          </label>
        </section>

        <section className="action-section">
          <button
            className="primary-action"
            type="button"
            disabled={!canStart}
            onClick={() => void startTranscription()}
          >
            {busy ? <Loader2 className="spin" size={20} /> : <Play size={20} />}
            {primaryActionLabel(stage, Boolean(optimizedPackage))}
          </button>
          <p className="action-note">
            {actionStatus(stage, keyReady, cloudConsent, hasSource, shareMode)}
          </p>
          {busy && (
            <div className={wakeLockActive ? "wake-status active" : "wake-status"}>
              <MonitorSmartphone size={16} />
              <span>
                {stage === "analyzing"
                  ? wakeLockStatus === "active"
                    ? "화면 켜짐 유지 중 · 서버 작업 접수 완료까지 기다려 주세요."
                    : wakeLockStatus === "requesting"
                      ? "자동 잠금 방지 연결 중 · 서버 작업 접수 완료까지 화면을 켜 두세요."
                      : "자동 잠금 방지가 작동하지 않습니다 · 서버 작업 접수 완료까지 화면을 직접 켜 두세요."
                  : wakeLockActive
                    ? "화면 켜짐 유지 중 · 화면이 꺼져도 PC 작업은 계속됩니다."
                    : "화면이 꺼지거나 브라우저를 닫아도 PC에서 계속 처리됩니다."}
              </span>
            </div>
          )}
          {stage === "transcribing" && transcriptionProgress && (
            <TranscriptionProgressView progress={transcriptionProgress} />
          )}
        </section>

        {optimizedPackage && !transcript && (
          <div className="resume-line">
            <CheckCircle2 size={17} />
            <span>
              오디오 최적화 완료 · {optimizedPackage.chunks.length}개 파일
            </span>
            <button
              type="button"
              onClick={() =>
                void downloadResult(
                  optimizedPackage.package_url,
                  `${safeSaveBaseName}_audio.zip`
                )
              }
            >
              ZIP
            </button>
          </div>
        )}

        {transcript && (
          <section className="result-section">
            <div className="result-header">
              <div>
                <p className="result-kicker">
                  <CheckCircle2 size={17} />
                  전사 완료
                </p>
                <h2>{safeSaveBaseName}</h2>
                {autoDownloadStatus && (
                  <p className="auto-download-status">{autoDownloadStatus}</p>
                )}
              </div>
              <div className="result-actions">
                <button
                  className="icon-button standalone"
                  type="button"
                  onClick={() => void copyTranscript()}
                  title="전사문 복사"
                  aria-label="전사문 복사"
                >
                  {copied ? <Check size={18} /> : <Clipboard size={18} />}
                </button>
                <button
                  className="download-button"
                  type="button"
                  onClick={() =>
                    void downloadResult(transcript.txt_url, `${safeSaveBaseName}.txt`)
                  }
                >
                  <Download size={16} />
                  TXT
                </button>
                <button
                  className="download-button"
                  type="button"
                  onClick={() =>
                    void downloadResult(transcript.json_url, `${safeSaveBaseName}.json`)
                  }
                >
                  <Download size={16} />
                  JSON
                </button>
              </div>
            </div>

            <div className="naming-panel">
              <div className="naming-heading">
                <FilePenLine size={17} />
                <div>
                  <strong>저장 파일명</strong>
                  <span>전사문, 음원 사본, ZIP에 같은 이름을 사용합니다.</span>
                </div>
              </div>
              <div className="naming-modes" aria-label="파일명 방식">
                <button
                  type="button"
                  className={namingMode === "original" ? "selected" : ""}
                  onClick={() => chooseNamingMode("original")}
                >
                  원본 파일명
                </button>
                <button
                  type="button"
                  className={namingMode === "recommended" ? "selected" : ""}
                  onClick={() => chooseNamingMode("recommended")}
                >
                  자동 추천
                </button>
                <button
                  type="button"
                  className={namingMode === "custom" ? "selected" : ""}
                  onClick={() => chooseNamingMode("custom")}
                >
                  직접 입력
                </button>
              </div>
              <input
                className="filename-input"
                value={saveBaseName}
                onChange={(event) => {
                  setNamingMode("custom");
                  setSaveBaseName(event.target.value);
                }}
                aria-label="저장 파일명"
              />
              <p>
                추천: <button type="button" onClick={() => chooseNamingMode("recommended")}>
                  {transcript.suggested_filename}
                </button>
              </p>
            </div>

            <div className="save-actions">
              {originalAudioUrl && file && (
                <a
                  className="download-button"
                  href={originalAudioUrl}
                  download={`${safeSaveBaseName}${fileExtension(file.name)}`}
                >
                  <FileAudio size={16} />
                  원본 음원 새 이름으로 저장
                </a>
              )}
              {optimizedPackage && (
                <button
                  className="download-button"
                  type="button"
                  onClick={() =>
                    void downloadResult(
                      optimizedPackage.package_url,
                      `${safeSaveBaseName}_audio.zip`
                    )
                  }
                >
                  <Download size={16} />
                  최적화 ZIP
                </button>
              )}
            </div>
            <p className="rename-note">
              브라우저 보안상 기존 음원 파일은 직접 변경하지 않고 새 이름의 사본을
              저장합니다.
            </p>
            <pre className="transcript-preview">{transcript.text}</pre>
          </section>
        )}

        <details className="utility-details">
          <summary>고급 / 로컬 도구</summary>
          <div className="utility-content">
            <div>
              <strong>오디오 패키지만 만들기</strong>
              <p>Google로 전송하지 않고 최적화된 MP3와 manifest를 만듭니다.</p>
            </div>
            <button
              className="secondary-button"
              type="button"
              disabled={(!file && !stagedUploadId) || !recommendation || busy}
              onClick={() => void createPackageOnly()}
            >
              <Download size={17} />
              패키지 만들기
            </button>
            {optimizedPackage && (
              <button
                className="text-link"
                type="button"
                onClick={() =>
                  void downloadResult(
                    optimizedPackage.package_url,
                    `${safeSaveBaseName}_audio.zip`
                  )
                }
              >
                최적화 ZIP 다운로드
                <Download size={14} />
              </button>
            )}
          </div>
        </details>

        <footer className="privacy-footer">
          <ShieldCheck size={16} />
          <span>
            {shareMode
              ? "기본 API key는 이 PC에 암호화해 저장하며 공유 브라우저에는 전달하지 않습니다."
              : "API key는 저장하지 않습니다. Google 전송 전까지 파일 처리는 로컬에서 진행됩니다."}
          </span>
        </footer>
      </div>
    </main>
  );
}

function SectionTitle({
  index,
  title,
  note
}: {
  index: string;
  title: string;
  note: string;
}) {
  return (
    <div className="section-title">
      <span>{index}</span>
      <h2>{title}</h2>
      <small>{note}</small>
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TranscriptionProgressView({
  progress
}: {
  progress: GeminiTranscriptionProgress;
}) {
  const percent = Math.min(100, Math.max(0, Math.round(progress.progress * 100)));
  const activeChunk =
    progress.completed_chunks < progress.total_chunks
      ? progress.current_chunk ?? progress.completed_chunks + 1
      : null;
  const chunkStatus = activeChunk
    ? `${progress.completed_chunks}/${progress.total_chunks} 구간 완료 · ${activeChunk}번째 처리 중`
    : `${progress.completed_chunks}/${progress.total_chunks} 구간 완료`;
  const etaStatus =
    progress.eta_sec === null
      ? "첫 구간 완료 후 남은 시간을 계산합니다."
      : progress.eta_sec < 10
        ? "곧 완료"
        : `약 ${formatRemainingTime(progress.eta_sec)} 남음`;

  return (
    <div className="transcription-progress" aria-live="polite">
      <div className="progress-heading">
        <span>전사 진행</span>
        <strong>{percent}%</strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Gemini 전사 진행률"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <div className="progress-meta">
        <span>{chunkStatus}</span>
        <span>{etaStatus}</span>
      </div>
    </div>
  );
}

function WorkflowProgress({
  stage,
  hasPackage
}: {
  stage: WorkflowStage;
  hasPackage: boolean;
}) {
  const activeIndex = workflowIndex(stage, hasPackage);
  return (
    <div className="workflow-progress" aria-live="polite">
      {workflowSteps.map((step, index) => {
        const className =
          stage === "failed" && index === activeIndex
            ? "failed"
            : index < activeIndex
              ? "done"
              : index === activeIndex
                ? "current"
                : "";
        return (
          <div className={className} key={step}>
            <span>{index + 1}</span>
            <small>{step}</small>
          </div>
        );
      })}
    </div>
  );
}

function workflowIndex(stage: WorkflowStage, hasPackage: boolean): number {
  if (stage === "transcribing") return 2;
  if (stage === "complete") return 3;
  if (stage === "optimizing" || stage === "ready") return 1;
  if (stage === "failed") return hasPackage ? 2 : 1;
  return 0;
}

function primaryActionLabel(stage: WorkflowStage, hasPackage: boolean): string {
  if (stage === "analyzing") return "파일 분석 중";
  if (stage === "optimizing") return "오디오 최적화 중";
  if (stage === "transcribing") return "Gemini 전사 중";
  if (stage === "failed" && hasPackage) return "전사 이어하기";
  if (stage === "complete") return "전사 다시 실행";
  return "전사 시작";
}

function workflowErrorMessage(error: string | null | undefined): string {
  const detail = error?.trim();
  if (!detail) return "PC 서버 작업을 완료하지 못했습니다.";
  if (/Internal error encountered|temporarily unavailable/i.test(detail)) {
    return "Gemini 서버가 일시적으로 응답하지 않았습니다. 최적화 음원은 저장되어 있으므로 '전사 이어하기'를 누르면 중단 지점부터 다시 시도합니다.";
  }
  return detail;
}

function actionStatus(
  stage: WorkflowStage,
  keyReady: boolean,
  consent: boolean,
  hasSource: boolean,
  shareMode: boolean
): string {
  if (stage === "analyzing") return "녹음 길이와 언어를 빠르게 확인하고 있습니다.";
  if (stage === "optimizing") return "음성을 선명하게 정리하고 안전한 크기로 나눕니다.";
  if (stage === "transcribing") return "완료된 구간은 저장되므로 중단되어도 이어집니다.";
  if (stage === "complete") return "전사문을 복사하거나 TXT / JSON으로 내려받으세요.";
  if (!hasSource) return "먼저 바로 녹음을 시작하세요.";
  if (!keyReady) {
    return shareMode ? "공유 비밀번호를 입력하세요." : "Gemini API key가 필요합니다.";
  }
  if (!consent) return "Google 전송 동의를 확인하세요.";
  if (shareMode) return "비밀번호 확인 후 최적화·전사·TXT 다운로드가 자동 진행됩니다.";
  return "준비되었습니다.";
}

function geminiOptimizerOptions(file?: File): OptimizerOptions {
  return {
    file,
    destination: "gemini",
    openaiModel: "gpt-4o-transcribe",
    wordTimestamps: false,
    codec: "",
    bitrateKbps: "",
    chunkMinutes: "",
    removeSilence: false,
    loudnorm: true,
    speechFilter: true,
    denoise: false
  };
}

function savePersistedWorkflow(workflow: PersistedWorkflow): void {
  try {
    window.localStorage.setItem(ACTIVE_WORKFLOW_STORAGE_KEY, JSON.stringify(workflow));
  } catch {
    // URL recovery still works when browser storage is unavailable.
  }
}

function loadPersistedWorkflow(): PersistedWorkflow | null {
  try {
    const value = window.localStorage.getItem(ACTIVE_WORKFLOW_STORAGE_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value) as Partial<PersistedWorkflow>;
    if (
      parsed.version !== 1 ||
      typeof parsed.sourceName !== "string" ||
      typeof parsed.sourceBytes !== "number" ||
      typeof parsed.saveBaseName !== "string" ||
      (!parsed.recommendation &&
        typeof parsed.cloudRecordingId !== "string" &&
        typeof parsed.workflowId !== "string")
    ) {
      return null;
    }
    return {
      version: 1,
      workflowId:
        typeof parsed.workflowId === "string" ? parsed.workflowId : null,
      stagedUploadId:
        typeof parsed.stagedUploadId === "string" ? parsed.stagedUploadId : null,
      cloudRecordingId:
        typeof parsed.cloudRecordingId === "string" ? parsed.cloudRecordingId : null,
      sourceName: parsed.sourceName,
      sourceBytes: parsed.sourceBytes,
      recommendation: parsed.recommendation || null,
      optimizedPackage: parsed.optimizedPackage || null,
      saveBaseName: parsed.saveBaseName
    };
  } catch {
    return null;
  }
}

function clearPersistedWorkflow(): void {
  try {
    window.localStorage.removeItem(ACTIVE_WORKFLOW_STORAGE_KEY);
  } catch {
    // Nothing else is required when browser storage is unavailable.
  }
}

function workflowIdFromUrl(): string | null {
  const value = new URLSearchParams(window.location.search).get("workflow");
  return value && /^[a-f0-9]{32}$/.test(value) ? value : null;
}

function setWorkflowUrl(workflowId: string | null): void {
  const url = new URL(window.location.href);
  if (workflowId) {
    url.searchParams.set("workflow", workflowId);
  } else {
    url.searchParams.delete("workflow");
  }
  window.history.replaceState(null, "", url);
}

function languageLabel(language: QuickScanResult["detected_language"] | undefined): string {
  if (language === "ko") return "한국어";
  if (language === "en") return "English";
  return "자동";
}

function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${minutes}:${String(secs).padStart(2, "0")}`;
}

function preferredRecordingMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/mp4",
    "audio/webm",
    "audio/ogg;codecs=opus"
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";
}

function recordingExtension(mimeType: string): "m4a" | "ogg" | "webm" {
  if (/mp4|m4a/i.test(mimeType)) return "m4a";
  if (/ogg/i.test(mimeType)) return "ogg";
  return "webm";
}

function recordingTimestampForName(date: Date): string {
  const parts = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0")
  ];
  const time = [
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0")
  ];
  return `${parts.join("")}_${time.join("")}`;
}

function recordingPermissionMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "마이크 권한이 필요합니다. 주소창의 권한 설정에서 마이크를 허용해 주세요.";
    }
    if (error.name === "NotFoundError") {
      return "사용할 수 있는 마이크를 찾지 못했습니다.";
    }
    if (error.name === "NotReadableError") {
      return "다른 앱이 마이크를 사용 중입니다. 다른 녹음 앱을 닫고 다시 시도하세요.";
    }
  }
  return "녹음을 시작하지 못했습니다. 마이크 권한과 브라우저 상태를 확인하세요.";
}

function recordingWakeLockMessage(status: WakeLockStatus): string {
  if (status === "active") return "화면 자동 잠금 방지 중입니다.";
  if (status === "requesting") {
    return "화면 자동 잠금 방지를 연결 중입니다. 연결될 때까지 화면을 켜 두세요.";
  }
  if (status === "released") {
    return "화면 자동 잠금 방지가 해제되었습니다. 녹음 중에는 화면을 직접 켜 두세요.";
  }
  if (status === "unavailable") {
    return "화면 자동 잠금 방지를 사용할 수 없습니다. 녹음 중에는 화면을 직접 켜 두세요.";
  }
  return "화면 자동 잠금 방지를 준비 중입니다. 녹음 중에는 화면을 켜 두세요.";
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatRemainingTime(seconds: number): string {
  const minutes = Math.max(1, Math.ceil(seconds / 60));
  if (minutes < 60) return `${minutes}분`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours}시간 ${remainingMinutes}분` : `${hours}시간`;
}

function baseNameFromFile(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  return lastDot > 0 ? filename.slice(0, lastDot) : filename;
}

function fileExtension(filename: string): string {
  const match = filename.match(/(\.[A-Za-z0-9]{1,8})$/);
  return match?.[1].toLowerCase() || "";
}

function sanitizeDownloadBaseName(value: string): string {
  return value
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/g, "")
    .slice(0, 96);
}

function wasWorkflowAutoDownloaded(workflowId: string): boolean {
  try {
    return window.localStorage.getItem(LAST_AUTO_DOWNLOADED_WORKFLOW_KEY) === workflowId;
  } catch {
    return false;
  }
}

function markWorkflowAutoDownloaded(workflowId: string): void {
  try {
    window.localStorage.setItem(LAST_AUTO_DOWNLOADED_WORKFLOW_KEY, workflowId);
  } catch {
    // A repeated download is preferable when browser storage is unavailable.
  }
}
