import { StrictMode } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import * as api from "../src/api";
import type {
  OptimizerAnalysisResponse,
  RuntimeProfile,
  TranscriptionWorkflowStart,
  TranscriptionWorkflowStatus
} from "../src/types";

vi.mock("../src/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("../src/api")>(),
  getRuntime: vi.fn(),
  hasApiAccessToken: vi.fn(),
  clearApiAccessToken: vi.fn(),
  completeCloudRecordingUpload: vi.fn(),
  createCloudUploadDescriptor: vi.fn(),
  getTranscriptionWorkflow: vi.fn(),
  analyzeOptimizer: vi.fn(),
  startTranscriptionWorkflow: vi.fn(),
  uploadCloudRecording: vi.fn(),
  createOptimizerPackage: vi.fn(),
  downloadApiFile: vi.fn(),
  verifyGeminiSharePasscode: vi.fn()
}));

const STORAGE_KEY = "local-meetscribe.active-workflow.v1";
const OLD_ID = "a".repeat(32);
const OLD_NAME = "old-recording.m4a";
const NEW_NAME = "new-recording.wav";
const runtime: RuntimeProfile = {
  device: "cpu", cuda: false, fast_model: "mock", accurate_model: "mock",
  gemini_transcription_enabled: true, gemini_api_key_configured: true,
  gemini_model: "mock", gemini_share_enabled: true, gemini_share_ready: true,
  local_admin: false
};
let restoreRecordingBrowser: (() => void) | null = null;
let restoreWakeLockBrowser: (() => void) | null = null;

function analysis(filename: string, upload_id = "test-upload"): OptimizerAnalysisResponse {
  return {
    upload_id,
    source: { filename, duration_sec: 1, sample_rate: 16000, channels: 1 },
    original_bytes: 4,
    recommendation: {
      destination: "gemini", provider_label: "mock", model: "mock", codec: "mp3",
      sample_rate_hz: 16000, channels: 1, bitrate_kbps: 32, chunk_count: 1,
      chunk_minutes: 30, projected_size_mb: 0.1, projected_chunk_mb: 0.1,
      estimated_tokens: null, estimated_cost_usd: null, delivery: "inline",
      rationale: "test", warnings: [], prompt: "test"
    },
    quick_scan: {
      glossary: [], preview_text: "", detected_language: "unknown", scan_seconds: 0,
      warning: null
    }
  };
}

function seedOldWorkflow() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    version: 1, workflowId: OLD_ID, stagedUploadId: null, sourceName: OLD_NAME,
    sourceBytes: 4, recommendation: analysis(OLD_NAME), optimizedPackage: null,
    saveBaseName: "old-recording"
  }));
  window.history.replaceState(null, "", `/?workflow=${OLD_ID}`);
}

function completedWorkflow(): TranscriptionWorkflowStatus {
  return {
    workflow_id: OLD_ID,
    package_id: OLD_ID,
    status: "complete",
    error: null,
    transcript: {
      provider: "gemini",
      model: "mock",
      text: "mock transcript",
      suggested_filename: "old-recording",
      chunk_count: 0,
      chunks: [],
      txt_url: "/old.txt",
      json_url: "/old.json"
    }
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

function fileInput(): HTMLInputElement {
  return document.querySelector('input[type="file"]:not([multiple])')!;
}

function chooseFile(name = NEW_NAME) {
  fireEvent.change(fileInput(), { target: { files: [new File(["test"], name, { type: "audio/wav" })] } });
}

function installRecordingBrowser(getUserMedia: () => Promise<MediaStream>) {
  const recorderDescriptor = Object.getOwnPropertyDescriptor(globalThis, "MediaRecorder");
  const mediaDevicesDescriptor = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");

  class MockMediaRecorder {
    static isTypeSupported(type: string) {
      return type === "audio/webm;codecs=opus";
    }

    state: RecordingState = "inactive";
    mimeType = "audio/webm;codecs=opus";
    ondataavailable: ((event: BlobEvent) => void) | null = null;
    onstop: ((event: Event) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;

    start() {
      this.state = "recording";
    }

    requestData() {}

    stop() {
      this.state = "inactive";
      this.ondataavailable?.({
        data: new Blob(["recorded audio"], { type: this.mimeType })
      } as BlobEvent);
      this.onstop?.(new Event("stop"));
    }
  }

  Object.defineProperty(globalThis, "MediaRecorder", {
    configurable: true,
    value: MockMediaRecorder
  });
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn(getUserMedia) }
  });
  restoreRecordingBrowser = () => {
    if (recorderDescriptor) {
      Object.defineProperty(globalThis, "MediaRecorder", recorderDescriptor);
    } else {
      Reflect.deleteProperty(globalThis, "MediaRecorder");
    }
    if (mediaDevicesDescriptor) {
      Object.defineProperty(navigator, "mediaDevices", mediaDevicesDescriptor);
    } else {
      Reflect.deleteProperty(navigator, "mediaDevices");
    }
  };
}

function installWakeLockBrowser(options: { reject?: boolean } = {}) {
  const wakeLockDescriptor = Object.getOwnPropertyDescriptor(navigator, "wakeLock");
  let releaseListener: (() => void) | null = null;
  const sentinel = {
    released: false,
    release: vi.fn(async () => {
      sentinel.released = true;
    }),
    addEventListener: vi.fn((type: string, listener: () => void) => {
      if (type === "release") releaseListener = listener;
    })
  };
  const request = vi.fn(async () => {
    if (options.reject) throw new DOMException("denied", "NotAllowedError");
    return sentinel as unknown as WakeLockSentinel;
  });
  Object.defineProperty(navigator, "wakeLock", {
    configurable: true,
    value: { request }
  });
  restoreWakeLockBrowser = () => {
    if (wakeLockDescriptor) {
      Object.defineProperty(navigator, "wakeLock", wakeLockDescriptor);
    } else {
      Reflect.deleteProperty(navigator, "wakeLock");
    }
  };
  return {
    request,
    release: sentinel.release,
    releaseFromBrowser() {
      sentinel.released = true;
      releaseListener?.();
    }
  };
}

function installUnsupportedWakeLockBrowser() {
  const wakeLockDescriptor = Object.getOwnPropertyDescriptor(navigator, "wakeLock");
  Reflect.deleteProperty(navigator, "wakeLock");
  restoreWakeLockBrowser = () => {
    if (wakeLockDescriptor) {
      Object.defineProperty(navigator, "wakeLock", wakeLockDescriptor);
    }
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
  sessionStorage.clear();
  window.history.replaceState(null, "", "/");
  URL.createObjectURL = vi.fn(() => "blob:test");
  URL.revokeObjectURL = vi.fn();
  vi.mocked(api.getRuntime).mockResolvedValue(runtime);
  vi.mocked(api.hasApiAccessToken).mockReturnValue(false);
  vi.mocked(api.getTranscriptionWorkflow).mockImplementation(() => new Promise(() => {}));
  vi.mocked(api.analyzeOptimizer).mockImplementation(() => new Promise(() => {}));
  vi.mocked(api.startTranscriptionWorkflow).mockImplementation(() => new Promise(() => {}));
});

afterEach(() => {
  cleanup();
  restoreRecordingBrowser?.();
  restoreRecordingBrowser = null;
  restoreWakeLockBrowser?.();
  restoreWakeLockBrowser = null;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("direct phone recording", () => {
  it("turns the captured audio into a file and starts the existing automatic upload", async () => {
    const stopTrack = vi.fn();
    installRecordingBrowser(async () => ({
      getTracks: () => [{ stop: stopTrack }]
    } as unknown as MediaStream));
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.analyzeOptimizer).mockResolvedValue(analysis("recorded.webm"));

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "바로 녹음 시작" }));
    await screen.findByText("녹음 중");
    fireEvent.click(screen.getByRole("button", { name: "녹음 종료 후 자동 전사" }));

    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledOnce());
    const recordedFile = vi.mocked(api.analyzeOptimizer).mock.calls[0][0].file;
    expect(recordedFile?.name).toMatch(/^PhoneScribe_\d{8}_\d{6}\.webm$/);
    expect(recordedFile?.type).toBe("audio/webm;codecs=opus");
    expect(recordedFile?.size).toBeGreaterThan(0);
    expect(stopTrack).toHaveBeenCalledOnce();
  });

  it("shows a useful message when microphone permission is denied", async () => {
    installRecordingBrowser(async () => {
      throw new DOMException("denied", "NotAllowedError");
    });
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "바로 녹음 시작" }));

    expect(
      await screen.findByText("마이크 권한이 필요합니다. 주소창의 권한 설정에서 마이크를 허용해 주세요.")
    ).toBeTruthy();
  });

  it("reports the actual Wake Lock held and released states while recording", async () => {
    installRecordingBrowser(async () => ({
      getTracks: () => [{ stop: vi.fn() }]
    } as unknown as MediaStream));
    const wakeLock = installWakeLockBrowser();

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "바로 녹음 시작" }));

    expect(await screen.findByText(/화면 자동 잠금 방지 중입니다/)).toBeTruthy();
    expect(wakeLock.request).toHaveBeenCalledWith("screen");
    act(() => wakeLock.releaseFromBrowser());
    expect(
      await screen.findByText(/화면 자동 잠금 방지가 해제되었습니다/)
    ).toBeTruthy();
  });

  it("warns the user when Wake Lock cannot be acquired", async () => {
    installRecordingBrowser(async () => ({
      getTracks: () => [{ stop: vi.fn() }]
    } as unknown as MediaStream));
    installWakeLockBrowser({ reject: true });

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "바로 녹음 시작" }));

    expect(
      await screen.findByText(/화면 자동 잠금 방지를 사용할 수 없습니다/)
    ).toBeTruthy();
  });

  it("warns the user when the browser has no Wake Lock API", async () => {
    installRecordingBrowser(async () => ({
      getTracks: () => [{ stop: vi.fn() }]
    } as unknown as MediaStream));
    installUnsupportedWakeLockBrowser();

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "바로 녹음 시작" }));

    expect(
      await screen.findByText(/화면 자동 잠금 방지를 사용할 수 없습니다/)
    ).toBeTruthy();
  });
});

describe("Supabase signed upload", () => {
  it("uploads signed parts, completes the cloud recording, and starts the workflow", async () => {
    const workflowStarted = deferred<TranscriptionWorkflowStart>();
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.getRuntime).mockResolvedValue({ ...runtime, cloud_upload_enabled: true });
    vi.mocked(api.createCloudUploadDescriptor).mockResolvedValue({
      recording_id: "cloud-recording-id",
      bucket_id: "recordings",
      object_path: "owner/cloud-recording-id/source.wav",
      content_type: "audio/wav",
      parts: [{
        part_number: 0,
        byte_start: 0,
        byte_end: 4,
        size_bytes: 4,
        object_path: "owner/cloud-recording-id/parts/000000.part",
        upload: {
          protocol: "signed-put",
          url: "https://storage.test/signed/part",
          headers: { "x-upsert": "true" }
        }
      }],
      expires_in: 3600
    });
    vi.mocked(api.uploadCloudRecording).mockImplementation(
      async (_file, _descriptor, onProgress) => onProgress?.(4, 4)
    );
    vi.mocked(api.completeCloudRecordingUpload).mockResolvedValue({
      recording_id: "cloud-recording-id",
      status: "ready"
    });
    vi.mocked(api.startTranscriptionWorkflow).mockReturnValue(workflowStarted.promise);

    render(<App />);
    chooseFile();

    await waitFor(() => expect(api.startTranscriptionWorkflow).toHaveBeenCalledOnce());
    expect(api.createCloudUploadDescriptor).toHaveBeenCalledOnce();
    expect(api.uploadCloudRecording).toHaveBeenCalledOnce();
    expect(api.completeCloudRecordingUpload).toHaveBeenCalledWith("cloud-recording-id");
    expect(api.analyzeOptimizer).not.toHaveBeenCalled();
    const workflowInput = vi.mocked(api.startTranscriptionWorkflow).mock.calls[0][1];
    expect(workflowInput).toMatchObject({ cloudRecordingId: "cloud-recording-id" });
    expect(workflowInput).not.toHaveProperty("uploadId");
    expect(workflowInput).not.toHaveProperty("packageId");
    expect(screen.getByText(/서버에 전사 작업을 접수하고 있습니다/)).toBeTruthy();
    expect(screen.queryByText(/이제 화면을 꺼도 전사와 저장이 계속/)).toBeNull();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).workflowId).toBeNull();

    await act(async () => {
      workflowStarted.resolve({
        workflow_id: OLD_ID,
        package_id: OLD_ID,
        status: "queued"
      });
    });
    expect(await screen.findByText(/서버 작업 접수 완료 · 이제 화면을 꺼도/)).toBeTruthy();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).workflowId).toBe(OLD_ID);
    expect(window.location.search).toContain(OLD_ID);
    expect(api.startTranscriptionWorkflow).toHaveBeenCalledOnce();
  });

  it("falls back to the existing PC upload when cloud upload is unavailable", async () => {
    const workflowStarted = deferred<TranscriptionWorkflowStart>();
    const wakeLock = installWakeLockBrowser();
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.getRuntime).mockResolvedValue({ ...runtime, cloud_upload_enabled: true });
    vi.mocked(api.createCloudUploadDescriptor).mockRejectedValue(
      new Error("cloud unavailable")
    );
    vi.mocked(api.analyzeOptimizer).mockResolvedValue(analysis(NEW_NAME));
    vi.mocked(api.startTranscriptionWorkflow).mockReturnValue(workflowStarted.promise);

    render(<App />);
    chooseFile();

    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledOnce());
    await waitFor(() => expect(api.startTranscriptionWorkflow).toHaveBeenCalledOnce());
    expect(vi.mocked(api.startTranscriptionWorkflow).mock.calls[0][1]).toMatchObject({
      uploadId: "test-upload",
    });
    expect(screen.getByText(/PC 직접 업로드 완료 · 서버에 전사 작업을 접수/)).toBeTruthy();
    expect(screen.queryByText(/이제 화면을 꺼도 전사와 저장이 계속/)).toBeNull();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).workflowId).toBeNull();
    expect(wakeLock.release).not.toHaveBeenCalled();

    await act(async () => {
      workflowStarted.resolve({
        workflow_id: OLD_ID,
        package_id: OLD_ID,
        status: "queued"
      });
    });
    expect(await screen.findByText(/서버 작업 접수 완료 · 이제 화면을 꺼도/)).toBeTruthy();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).workflowId).toBe(OLD_ID);
    expect(api.startTranscriptionWorkflow).toHaveBeenCalledOnce();
    await waitFor(() => expect(wakeLock.release).toHaveBeenCalledOnce());
  });
});

describe("switching recordings while a restored workflow is stuck", () => {
  it("aborts the legacy PC upload before starting the replacement file", async () => {
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    let oldSignal: AbortSignal | undefined;
    vi.mocked(api.analyzeOptimizer)
      .mockImplementationOnce((_options, _language, _quickScan, signal) => {
        oldSignal = signal;
        return new Promise((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true }
          );
        });
      })
      .mockImplementationOnce(() => new Promise(() => {}));

    render(<App />);
    chooseFile(OLD_NAME);
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledOnce());
    expect(oldSignal?.aborted).toBe(false);

    chooseFile(NEW_NAME);

    expect(oldSignal?.aborted).toBe(true);
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.analyzeOptimizer).mock.calls[1][0].file?.name).toBe(NEW_NAME);
  });

  it("aborts the legacy PC upload when the page unmounts", async () => {
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    let uploadSignal: AbortSignal | undefined;
    vi.mocked(api.analyzeOptimizer).mockImplementation(
      (_options, _language, _quickScan, signal) => {
        uploadSignal = signal;
        return new Promise((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true }
          );
        });
      }
    );

    const view = render(<App />);
    chooseFile(OLD_NAME);
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledOnce());
    expect(uploadSignal?.aborted).toBe(false);

    view.unmount();

    expect(uploadSignal?.aborted).toBe(true);
  });

  it("opens the picker despite a 401; cancelling keeps the old job, choosing clears it", async () => {
    seedOldWorkflow();
    vi.mocked(api.getTranscriptionWorkflow).mockRejectedValue(new api.ApiRequestError("expired", 401));
    const view = render(<StrictMode><App /></StrictMode>);
    await screen.findByText("전사 결과 연결 복구");
    const changeButton = screen.getByRole("button", { name: "다른 파일 선택" }) as HTMLButtonElement;
    expect(changeButton.disabled).toBe(false);
    const clickPicker = vi.spyOn(fileInput(), "click").mockImplementation(() => {});
    fireEvent.click(changeButton);
    expect(clickPicker).toHaveBeenCalledOnce();
    expect(screen.getByText(OLD_NAME)).toBeTruthy();
    expect(window.location.search).toContain(OLD_ID);

    chooseFile();
    expect(screen.getByText(NEW_NAME)).toBeTruthy();
    expect(screen.queryByText(OLD_NAME)).toBeNull();
    expect(screen.queryByText("전사 결과 연결 복구")).toBeNull();
    expect(window.location.search).toBe("");
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(api.startTranscriptionWorkflow).not.toHaveBeenCalled();
    view.unmount();
    render(<App />);
    expect(screen.queryByText(OLD_NAME)).toBeNull();
  });

  it.each(["success", "unauthorized", "not-found", "network"])(
    "ignores a delayed %s response from the old poll after selection", async (outcome) => {
      seedOldWorkflow();
      const oldPoll = deferred<TranscriptionWorkflowStatus>();
      vi.mocked(api.getTranscriptionWorkflow).mockReturnValue(oldPoll.promise);
      render(<StrictMode><App /></StrictMode>);
      await waitFor(() => expect(api.getTranscriptionWorkflow).toHaveBeenCalled());
      expect((screen.getByRole("button", { name: "다른 파일 선택" }) as HTMLButtonElement).disabled).toBe(false);
      chooseFile();
      await act(async () => {
        if (outcome === "success") {
          oldPoll.resolve({
            workflow_id: OLD_ID, package_id: OLD_ID, status: "complete", error: null,
            transcript: { provider: "gemini", model: "mock", text: "mock transcript",
              suggested_filename: "old", chunk_count: 0, chunks: [], txt_url: "/old.txt", json_url: "/old.json" }
          });
        } else {
          oldPoll.reject(outcome === "network" ? new Error("offline") :
            new api.ApiRequestError(outcome === "not-found" ? "Workflow not found" : "expired",
              outcome === "not-found" ? 404 : 401));
        }
      });
      expect(screen.getByText(NEW_NAME)).toBeTruthy();
      expect(screen.queryByRole("alert")).toBeNull();
      expect(screen.queryByText("mock transcript")).toBeNull();
      expect(api.downloadApiFile).not.toHaveBeenCalled();
      expect(api.clearApiAccessToken).not.toHaveBeenCalled();
      expect(window.location.search).toBe("");
      expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    }
  );

  it("accepts the new analysis and never starts transcription from the old analysis", async () => {
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    const oldAnalysis = deferred<OptimizerAnalysisResponse>();
    const newAnalysis = deferred<OptimizerAnalysisResponse>();
    vi.mocked(api.analyzeOptimizer).mockReturnValueOnce(oldAnalysis.promise).mockReturnValueOnce(newAnalysis.promise);
    render(<App />);
    chooseFile(OLD_NAME);
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledTimes(1));
    chooseFile();
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledTimes(2));
    await act(async () => { oldAnalysis.resolve(analysis(OLD_NAME, "old-upload")); });
    expect(api.startTranscriptionWorkflow).not.toHaveBeenCalled();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    await act(async () => { newAnalysis.resolve(analysis(NEW_NAME, "new-upload")); });
    await waitFor(() => expect(api.startTranscriptionWorkflow).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.startTranscriptionWorkflow).mock.calls[0][1].uploadId).toBe("new-upload");
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).sourceName).toBe(NEW_NAME);
  });

  it("does not restore an old job when its start request returns after switching", async () => {
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.analyzeOptimizer).mockResolvedValueOnce(analysis(OLD_NAME));
    const started = deferred<TranscriptionWorkflowStart>();
    vi.mocked(api.startTranscriptionWorkflow).mockReturnValue(started.promise);
    render(<App />);
    chooseFile(OLD_NAME);
    await waitFor(() => expect(api.startTranscriptionWorkflow).toHaveBeenCalledTimes(1));
    chooseFile();
    await act(async () => { started.resolve({ workflow_id: OLD_ID, package_id: OLD_ID, status: "queued" }); });
    expect(screen.getByText(NEW_NAME)).toBeTruthy();
    expect(api.getTranscriptionWorkflow).not.toHaveBeenCalled();
    expect(window.location.search).toBe("");
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("finishing old re-authentication does not reattach the previous job", async () => {
    seedOldWorkflow();
    vi.mocked(api.getTranscriptionWorkflow).mockRejectedValue(new api.ApiRequestError("expired", 401));
    const verified = deferred<{ valid: boolean; key_ready: boolean; expires_in: number }>();
    vi.mocked(api.verifyGeminiSharePasscode).mockReturnValue(verified.promise);
    render(<App />);
    await screen.findByText("전사 결과 연결 복구");
    fireEvent.change(screen.getByLabelText("공유 비밀번호 다시 입력"), { target: { value: "test-passcode" } });
    fireEvent.click(screen.getByRole("button", { name: "확인하고 결과 받기" }));
    expect(api.verifyGeminiSharePasscode).toHaveBeenCalledOnce();
    chooseFile();
    await act(async () => { verified.resolve({ valid: true, key_ready: true, expires_in: 100 }); });
    expect(api.getTranscriptionWorkflow).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(api.analyzeOptimizer).toHaveBeenCalledOnce());
    expect(vi.mocked(api.analyzeOptimizer).mock.calls[0][0].file?.name).toBe(NEW_NAME);
    expect(window.location.search).toBe("");
  });
});

describe("automatic TXT download", () => {
  it("retries transient failures with bounded backoff until the download succeeds", async () => {
    vi.useFakeTimers();
    seedOldWorkflow();
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.getTranscriptionWorkflow).mockResolvedValue(completedWorkflow());
    vi.mocked(api.downloadApiFile)
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockRejectedValueOnce(new api.ApiRequestError("busy", 503))
      .mockResolvedValueOnce();

    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    await vi.advanceTimersByTimeAsync(350);
    expect(api.downloadApiFile).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1_000);
    expect(api.downloadApiFile).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(3_000);
    expect(api.downloadApiFile).toHaveBeenCalledTimes(3);
    expect(screen.getByText(/TXT를 자동 다운로드했습니다/)).toBeTruthy();
  });

  it("cancels a scheduled TXT retry when the page unmounts", async () => {
    vi.useFakeTimers();
    seedOldWorkflow();
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.getTranscriptionWorkflow).mockResolvedValue(completedWorkflow());
    vi.mocked(api.downloadApiFile).mockRejectedValue(new TypeError("offline"));

    const view = render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    await vi.advanceTimersByTimeAsync(350);
    expect(api.downloadApiFile).toHaveBeenCalledOnce();

    view.unmount();
    await vi.runAllTimersAsync();

    expect(api.downloadApiFile).toHaveBeenCalledOnce();
  });

  it("stops after the bounded number of automatic TXT retries", async () => {
    vi.useFakeTimers();
    seedOldWorkflow();
    vi.mocked(api.hasApiAccessToken).mockReturnValue(true);
    vi.mocked(api.getTranscriptionWorkflow).mockResolvedValue(completedWorkflow());
    vi.mocked(api.downloadApiFile).mockRejectedValue(
      new api.ApiRequestError("temporarily unavailable", 503)
    );

    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    await vi.runAllTimersAsync();

    expect(api.downloadApiFile).toHaveBeenCalledTimes(5);
    expect(screen.getByText(/TXT 버튼을 눌러 다시 받아 주세요/)).toBeTruthy();
  });
});
