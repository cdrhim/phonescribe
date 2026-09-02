import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { uploadCloudRecording } from "../src/api";
import type { CloudUploadDescriptor } from "../src/types";

function descriptor(): CloudUploadDescriptor {
  return {
    recording_id: "recording-id",
    bucket_id: "recordings",
    object_path: "owner/recording-id/source.webm",
    content_type: "audio/webm",
    parts: [
      {
        part_number: 0,
        byte_start: 0,
        byte_end: 10,
        size_bytes: 10,
        object_path: "owner/recording-id/parts/000000.part",
        upload: {
          protocol: "signed-put",
          url: "https://storage-one.test/signed/first",
          headers: { "x-upsert": "true" }
        }
      },
      {
        part_number: 1,
        byte_start: 10,
        byte_end: 15,
        size_bytes: 5,
        object_path: "owner/recording-id/parts/000001.part",
        upload: {
          protocol: "signed-put",
          url: "https://different-storage.test/signed/second",
          headers: { "x-upsert": "true" }
        }
      }
    ],
    expires_in: 3600
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("cloud signed PUT uploader", () => {
  it("uploads ordered parts to independent HTTPS origins with aggregate progress", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response(null, { status: 200 }));
    const file = new File(["123456789012345"], "meeting.webm", {
      type: "audio/webm",
      lastModified: 123
    });
    const progress: Array<[number, number]> = [];

    await uploadCloudRecording(file, descriptor(), (uploaded, total) => {
      progress.push([uploaded, total]);
    });

    expect(fetch).toHaveBeenCalledTimes(2);
    const [firstUrl, firstInit] = vi.mocked(fetch).mock.calls[0];
    const [secondUrl, secondInit] = vi.mocked(fetch).mock.calls[1];
    expect(firstUrl).toBe("https://storage-one.test/signed/first");
    expect(secondUrl).toBe("https://different-storage.test/signed/second");
    expect(firstInit?.method).toBe("PUT");
    expect(secondInit?.method).toBe("PUT");
    expect((firstInit?.body as Blob).size).toBe(10);
    expect((secondInit?.body as Blob).size).toBe(5);
    expect(new Headers(firstInit?.headers).get("Content-Type")).toBe("audio/webm");
    expect(progress).toEqual([[0, 15], [10, 15], [15, 15]]);
  });

  it("retries network and transient HTTP failures", async () => {
    vi.useFakeTimers();
    const onePart = descriptor();
    onePart.parts = [onePart.parts[0]];
    const file = new File(["1234567890"], "meeting.webm");
    vi.mocked(fetch)
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));

    const uploaded = uploadCloudRecording(file, onePart);
    await vi.runAllTimersAsync();
    await uploaded;

    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("aborts during retry backoff without sending another request", async () => {
    vi.useFakeTimers();
    const onePart = descriptor();
    onePart.parts = [onePart.parts[0]];
    const file = new File(["1234567890"], "meeting.webm");
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("offline"));
    const controller = new AbortController();

    const uploaded = uploadCloudRecording(file, onePart, undefined, controller.signal);
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await expect(uploaded).rejects.toMatchObject({ name: "AbortError" });
    await vi.runAllTimersAsync();

    expect(fetch).toHaveBeenCalledOnce();
  });

  it("times out a stalled PUT attempt and retries it", async () => {
    vi.useFakeTimers();
    const onePart = descriptor();
    onePart.parts = [onePart.parts[0]];
    const file = new File(["1234567890"], "meeting.webm");
    vi.mocked(fetch)
      .mockImplementationOnce((_url, init) => new Promise((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(new DOMException("aborted", "AbortError")),
          { once: true }
        );
      }))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));

    const uploaded = uploadCloudRecording(file, onePart);
    await vi.advanceTimersByTimeAsync(120_000);
    await vi.advanceTimersByTimeAsync(1_000);
    await uploaded;

    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("aborts a stalled PUT immediately when the user cancels", async () => {
    vi.useFakeTimers();
    const onePart = descriptor();
    onePart.parts = [onePart.parts[0]];
    const file = new File(["1234567890"], "meeting.webm");
    vi.mocked(fetch).mockImplementationOnce((_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(new DOMException("aborted", "AbortError")),
          { once: true }
        );
      })
    );
    const controller = new AbortController();

    const uploaded = uploadCloudRecording(file, onePart, undefined, controller.signal);
    controller.abort();
    await expect(uploaded).rejects.toMatchObject({ name: "AbortError" });
    await vi.runAllTimersAsync();

    expect(fetch).toHaveBeenCalledOnce();
  });

  it("rejects non-HTTPS or gapped descriptors before sending bytes", async () => {
    const broken = descriptor();
    broken.parts[0].upload.url = "http://storage.test/signed/first";
    broken.parts[1].byte_start = 11;
    const file = new File(["123456789012345"], "meeting.webm");

    await expect(uploadCloudRecording(file, broken)).rejects.toThrow(
      "클라우드 업로드 조각 정보가 올바르지 않습니다."
    );
    expect(fetch).not.toHaveBeenCalled();
  });
});
