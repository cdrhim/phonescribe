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
  MonitorSmartphone,
  Play,
  RotateCcw,
  ShieldCheck,
  Smartphone,
  Upload
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  type OptimizerOptions,
  analyzeOptimizer,
  configureGeminiShareKey,
  createOptimizerPackage,
  getTranscriptionWorkflow,
  getRuntime,
  startTranscriptionWorkflow,
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

const workflowSteps = ["분석", "최적화", "전사", "완료"];
const ACTIVE_WORKFLOW_STORAGE_KEY = "local-meetscribe.active-workflow.v1";
const LAST_AUTO_DOWNLOADED_WORKFLOW_KEY =
  "local-meetscribe.last-auto-downloaded-workflow.v1";

interface PersistedWorkflow {
  version: 1;
  workflowId: string | null;
  stagedUploadId: string | null;
  sourceName: string;
  sourceBytes: number;
  recommendation: OptimizerRecommendationResponse;
  optimizedPackage: OptimizedPackageResult | null;
  saveBaseName: string;
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [sourceBytes, setSourceBytes] = useState(0);
  const [runtime, setRuntime] = useState<RuntimeProfile | null>(null);
  const [scan, setScan] = useState<QuickScanResult | null>(null);
  const [recommendation, setRecommendation] =
    useState<OptimizerRecommendationResponse | null>(null);
  const [stagedUploadId, setStagedUploadId] = useState<string | null>(null);
  const [optimizedPackage, setOptimizedPackage] =
    useState<OptimizedPackageResult | null>(null);
  const [transcript, setTranscript] = useState<GeminiTranscriptResult | null>(null);
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [sharePasscode, setSharePasscode] = useState("");
  const [shareAccessReady, setShareAccessReady] = useState(false);
  const [shareStatus, setShareStatus] = useState<string | null>(null);
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
  const [wakeLockActive, setWakeLockActive] = useState(false);
  const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(null);
  const [autoDownloadStatus, setAutoDownloadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const workflowTimerRef = useRef<number | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const restoreAttemptedRef = useRef(false);
  const workflowStartingRef = useRef(false);
  const autoStartSuppressedRef = useRef(false);

  useEffect(() => {
    void getRuntime().then(setRuntime).catch(() => setRuntime(null));
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
      if (workflowTimerRef.current !== null) {
        window.clearInterval(workflowTimerRef.current);
      }
      void wakeLockRef.current?.release();
    },
    []
  );

  useEffect(() => {
    if (restoreAttemptedRef.current) return;
    restoreAttemptedRef.current = true;
    const saved = loadPersistedWorkflow();
    const resumableWorkflowId = saved?.workflowId || workflowIdFromUrl();
    if (saved) {
      setSourceName(saved.sourceName);
      setSourceBytes(saved.sourceBytes);
      setRecommendation(saved.recommendation);
      setStagedUploadId(saved.stagedUploadId);
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

  useEffect(() => {
    if (!runtime?.gemini_share_enabled || !runtime.gemini_share_ready) return;
    const passcode = sharePasscode.trim();
    if (passcode.length < 4) {
      setShareAccessReady(false);
      setShareStatus(null);
      return;
    }
    let active = true;
    setShareStatus("비밀번호 확인 중");
    const timer = window.setTimeout(() => {
      void verifyGeminiSharePasscode(passcode)
        .then((result) => {
          if (!active) return;
          setShareAccessReady(result.valid && result.key_ready);
          setShareStatus(
            result.valid && result.key_ready
              ? "Gemini 기본 키에 연결되었습니다."
              : "서버 PC에서 기본 키 등록이 필요합니다."
          );
        })
        .catch((verificationError) => {
          if (!active) return;
          setShareAccessReady(false);
          setShareStatus(
            verificationError instanceof Error
              ? verificationError.message
              : "공유 비밀번호를 확인하지 못했습니다."
          );
        });
    }, 250);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [runtime?.gemini_share_enabled, runtime?.gemini_share_ready, sharePasscode]);

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
  const canStart = Boolean(
    (stagedUploadId || optimizedPackage) &&
      recommendation &&
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
    if (wasWorkflowAutoDownloaded(activeWorkflowId)) {
      setAutoDownloadStatus("TXT 전사문 다운로드가 준비되었습니다.");
      return;
    }

    let cancelled = false;
    let timer: number | null = null;
    const downloadWhenVisible = () => {
      if (cancelled) return;
      if (document.visibilityState !== "visible") {
        setAutoDownloadStatus("화면을 다시 켜면 TXT 전사문이 자동 다운로드됩니다.");
        return;
      }
      triggerBrowserDownload(
        withDownloadName(transcript.txt_url, `${safeSaveBaseName}.txt`),
        `${safeSaveBaseName}.txt`
      );
      markWorkflowAutoDownloaded(activeWorkflowId);
      setAutoDownloadStatus("TXT 전사문을 자동 다운로드했습니다.");
    };
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") downloadWhenVisible();
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    if (document.visibilityState === "visible") {
      timer = window.setTimeout(downloadWhenVisible, 350);
    } else {
      setAutoDownloadStatus("화면을 다시 켜면 TXT 전사문이 자동 다운로드됩니다.");
    }
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [activeWorkflowId, safeSaveBaseName, stage, transcript]);

  useEffect(() => {
    const wakeLockApi = "wakeLock" in navigator ? navigator.wakeLock : null;
    let cancelled = false;

    async function acquireWakeLock() {
      if (
        !busy ||
        !wakeLockApi ||
        document.visibilityState !== "visible" ||
        wakeLockRef.current
      ) {
        return;
      }
      try {
        const sentinel = await wakeLockApi.request("screen");
        if (cancelled) {
          await sentinel.release();
          return;
        }
        wakeLockRef.current = sentinel;
        setWakeLockActive(true);
        sentinel.addEventListener("release", () => {
          if (wakeLockRef.current === sentinel) {
            wakeLockRef.current = null;
            setWakeLockActive(false);
          }
        });
      } catch {
        setWakeLockActive(false);
      }
    }

    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        void acquireWakeLock();
      }
    }

    if (busy) {
      void acquireWakeLock();
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      const sentinel = wakeLockRef.current;
      wakeLockRef.current = null;
      setWakeLockActive(false);
      if (sentinel && !sentinel.released) {
        void sentinel.release();
      }
    };
  }, [busy]);

  function stopProgressPolling() {
    if (workflowTimerRef.current !== null) {
      window.clearInterval(workflowTimerRef.current);
      workflowTimerRef.current = null;
    }
  }

  async function refreshWorkflow(currentWorkflowId: string) {
    try {
      const workflow = await getTranscriptionWorkflow(currentWorkflowId);
      setError(null);
      applyWorkflowStatus(workflow);
    } catch (workflowError) {
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
      setError("연결을 다시 확인하고 있습니다. PC의 전사 작업은 계속 진행됩니다.");
    }
  }

  function beginWorkflowPolling(currentWorkflowId: string) {
    stopProgressPolling();
    void refreshWorkflow(currentWorkflowId);
    workflowTimerRef.current = window.setInterval(
      () => void refreshWorkflow(currentWorkflowId),
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
      setShareStatus("기본 키가 이 PC에 암호화되어 저장되었습니다.");
    } catch (saveError) {
      setShareAccessReady(false);
      setError(
        saveError instanceof Error ? saveError.message : "기본 API key를 저장하지 못했습니다."
      );
    } finally {
      setSavingShareKey(false);
    }
  }

  async function selectFile(selected: File) {
    stopProgressPolling();
    clearPersistedWorkflow();
    autoStartSuppressedRef.current = false;
    setFile(selected);
    setSourceName(selected.name);
    setSourceBytes(selected.size);
    setScan(null);
    setRecommendation(null);
    setStagedUploadId(null);
    setOptimizedPackage(null);
    setTranscript(null);
    setError(null);
    setCopied(false);
    setNamingMode("original");
    setSaveBaseName(baseNameFromFile(selected.name));
    setTranscriptionProgress(null);
    setActiveWorkflowId(null);
    setAutoDownloadStatus(null);
    setStage("analyzing");

    try {
      const analysis = await analyzeOptimizer(geminiOptimizerOptions(selected), "auto");
      setRecommendation({
        source: analysis.source,
        original_bytes: analysis.original_bytes,
        recommendation: analysis.recommendation
      });
      setStagedUploadId(analysis.upload_id);
      setScan(analysis.quick_scan);
      setStage("ready");
      savePersistedWorkflow({
        version: 1,
        workflowId: null,
        stagedUploadId: analysis.upload_id,
        sourceName: selected.name,
        sourceBytes: selected.size,
        recommendation: {
          source: analysis.source,
          original_bytes: analysis.original_bytes,
          recommendation: analysis.recommendation
        },
        optimizedPackage: null,
        saveBaseName: baseNameFromFile(selected.name)
      });
    } catch (analysisError) {
      setStage("failed");
      setError(
        analysisError instanceof Error
          ? analysisError.message
          : "녹음 파일을 분석하지 못했습니다."
      );
    }
  }

  async function startTranscription() {
    if (!canStart || !recommendation || workflowStartingRef.current) return;
    workflowStartingRef.current = true;
    stopProgressPolling();
    setError(null);
    setTranscript(null);
    setCopied(false);
    setTranscriptionProgress(null);
    setAutoDownloadStatus(null);

    try {
      setStage(optimizedPackage ? "transcribing" : "optimizing");
      const workflow = await startTranscriptionWorkflow(
        geminiOptimizerOptions(file ?? undefined),
        {
          uploadId: optimizedPackage ? undefined : stagedUploadId ?? undefined,
          packageId: optimizedPackage?.id,
          apiKey: shareMode ? undefined : geminiApiKey.trim() || undefined,
          sharePasscode: shareMode ? sharePasscode.trim() : undefined
        }
      );
      setActiveWorkflowId(workflow.workflow_id);
      setWorkflowUrl(workflow.workflow_id);
      savePersistedWorkflow({
        version: 1,
        workflowId: workflow.workflow_id,
        stagedUploadId,
        sourceName: displaySourceName,
        sourceBytes: displaySourceBytes,
        recommendation,
        optimizedPackage,
        saveBaseName
      });
      beginWorkflowPolling(workflow.workflow_id);
    } catch (transcriptionError) {
      stopProgressPolling();
      setStage("failed");
      setError(
        transcriptionError instanceof Error
          ? transcriptionError.message
          : "전사 작업을 완료하지 못했습니다."
      );
    } finally {
      workflowStartingRef.current = false;
    }
  }

  async function createPackageOnly() {
    if ((!file && !stagedUploadId) || !recommendation || busy) return;
    autoStartSuppressedRef.current = true;
    setError(null);
    setStage("optimizing");
    try {
      const packageResult = await createOptimizerPackage(
        geminiOptimizerOptions(file ?? undefined),
        stagedUploadId ?? undefined
      );
      setOptimizedPackage(packageResult);
      setStagedUploadId(null);
      setStage("ready");
      savePersistedWorkflow({
        version: 1,
        workflowId: null,
        stagedUploadId: null,
        sourceName: displaySourceName,
        sourceBytes: displaySourceBytes,
        recommendation,
        optimizedPackage: packageResult,
        saveBaseName
      });
    } catch (packageError) {
      setStage("failed");
      setError(
        packageError instanceof Error
          ? packageError.message
          : "최적화 파일을 만들지 못했습니다."
      );
    }
  }

  function resetFile() {
    stopProgressPolling();
    clearPersistedWorkflow();
    setWorkflowUrl(null);
    autoStartSuppressedRef.current = false;
    setFile(null);
    setSourceName("");
    setSourceBytes(0);
    setScan(null);
    setRecommendation(null);
    setStagedUploadId(null);
    setOptimizedPackage(null);
    setTranscript(null);
    setError(null);
    setCopied(false);
    setNamingMode("original");
    setSaveBaseName("");
    setTranscriptionProgress(null);
    setActiveWorkflowId(null);
    setAutoDownloadStatus(null);
    setStage("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function copyTranscript() {
    if (!transcript) return;
    await navigator.clipboard.writeText(transcript.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
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

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) void selectFile(selected);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const selected = event.dataTransfer.files?.[0];
    if (selected) void selectFile(selected);
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
            <p className="tagline">휴대폰 녹음을 올리면 바로 전사문을 만듭니다.</p>
          </div>
        </header>

        <WorkflowProgress stage={stage} hasPackage={Boolean(optimizedPackage)} />

        {error && (
          <div className="error-banner" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <section className="flow-section">
          <SectionTitle index="01" title="녹음 파일" note="M4A · MP3 · WAV · AAC" />
          {!hasSource ? (
            <div
              className="drop-zone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <Upload size={25} />
              <strong>휴대폰 녹음 파일을 선택하세요</strong>
              <span>파일은 먼저 이 컴퓨터에서 분석됩니다.</span>
              <button
                className="secondary-button"
                type="button"
                onClick={() => fileInputRef.current?.click()}
              >
                <FileAudio size={17} />
                녹음 파일 선택
              </button>
            </div>
          ) : (
            <div className="selected-file">
              <FileAudio size={24} />
              <div>
                <strong>{displaySourceName}</strong>
                <span>
                  {stage === "analyzing"
                    ? "길이, 언어, 전사 방식을 확인하고 있습니다."
                    : "분석 완료"}
                </span>
              </div>
              <button
                className="icon-button standalone"
                type="button"
                onClick={resetFile}
                title="다른 파일 선택"
                aria-label="다른 파일 선택"
                disabled={busy}
              >
                <RotateCcw size={18} />
              </button>
            </div>
          )}
          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept=".m4a,.mp3,.wav,.aac,.ogg,.flac,audio/*"
            onChange={handleFileChange}
            aria-hidden="true"
            tabIndex={-1}
          />

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
              <label className="key-field">
                <span>
                  <KeyRound size={16} />
                   공유 비밀번호
                   <small>확인되면 전사가 자동 시작됩니다.</small>
                </span>
                <span className="secret-input single">
                  <input
                    type="password"
                    value={sharePasscode}
                    onChange={(event) => {
                      setSharePasscode(event.target.value);
                      setShareAccessReady(false);
                    }}
                    placeholder="4자리 이상"
                    inputMode="numeric"
                    maxLength={12}
                    autoComplete="off"
                    spellCheck={false}
                  />
                </span>
              </label>

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
                  ? wakeLockActive
                    ? "화면 켜짐 유지 중 · 분석 완료까지 이 화면을 유지합니다."
                    : "파일 업로드와 분석이 끝날 때까지 휴대폰 화면을 켜 두세요."
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
            <a href={optimizedPackage.package_url}>ZIP</a>
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
                <a
                  className="download-button"
                  href={withDownloadName(transcript.txt_url, `${safeSaveBaseName}.txt`)}
                >
                  <Download size={16} />
                  TXT
                </a>
                <a
                  className="download-button"
                  href={withDownloadName(transcript.json_url, `${safeSaveBaseName}.json`)}
                >
                  <Download size={16} />
                  JSON
                </a>
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
                <a
                  className="download-button"
                  href={withDownloadName(
                    optimizedPackage.package_url,
                    `${safeSaveBaseName}_audio.zip`
                  )}
                >
                  <Download size={16} />
                  최적화 ZIP
                </a>
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
              <a
                className="text-link"
                href={withDownloadName(
                  optimizedPackage.package_url,
                  `${safeSaveBaseName}_audio.zip`
                )}
              >
                최적화 ZIP 다운로드
                <Download size={14} />
              </a>
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
  if (!hasSource) return "먼저 녹음 파일을 선택하세요.";
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
      !parsed.recommendation
    ) {
      return null;
    }
    return {
      version: 1,
      workflowId:
        typeof parsed.workflowId === "string" ? parsed.workflowId : null,
      stagedUploadId:
        typeof parsed.stagedUploadId === "string" ? parsed.stagedUploadId : null,
      sourceName: parsed.sourceName,
      sourceBytes: parsed.sourceBytes,
      recommendation: parsed.recommendation,
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

function withDownloadName(url: string, downloadName: string): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}download_name=${encodeURIComponent(downloadName)}`;
}

function triggerBrowserDownload(url: string, downloadName: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = downloadName;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
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
