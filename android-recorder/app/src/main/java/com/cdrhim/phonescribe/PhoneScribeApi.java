package com.cdrhim.phonescribe;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

final class PhoneScribeApi {
    private static final long MAX_PART_SIZE = 6L * 1024L * 1024L;
    private static final long[] RETRY_DELAYS_MS = {0L, 1000L, 3000L, 5000L, 10000L, 20000L};
    private static final int CONNECT_TIMEOUT_MS = 30_000;
    private static final int READ_TIMEOUT_MS = 120_000;
    private static final long WORKFLOW_TIMEOUT_MS = 4L * 60L * 60L * 1000L;

    interface ProgressListener {
        void onProgress(String stage, int percent);
    }

    private PhoneScribeApi() {}

    static Session verify(String baseUrl, String passcode) throws IOException {
        Session session = new Session(baseUrl, passcode);
        session.refreshToken();
        return session;
    }

    static WorkResult uploadAndTranscribe(
            Session session, File recording, ProgressListener progress) throws IOException {
        if (!recording.isFile() || recording.length() <= 0L) {
            throw new IOException("전송할 녹음 파일이 없습니다.");
        }

        JSONObject descriptorRequest = new JSONObject();
        try {
            descriptorRequest.put("filename", recording.getName());
            descriptorRequest.put("content_type", "audio/mp4");
            descriptorRequest.put("size_bytes", recording.length());
        } catch (JSONException impossible) {
            throw new IOException("업로드 정보를 만들 수 없습니다.", impossible);
        }

        HttpResult descriptorResponse = session.authorizedRequest(
                "POST",
                "/api/cloud-recordings/upload-descriptor",
                "application/json; charset=utf-8",
                descriptorRequest.toString().getBytes(StandardCharsets.UTF_8));
        UploadPlan plan = UploadPlan.fromJson(parseObject(descriptorResponse.body));
        plan.validate(recording.length());

        int completedParts = 0;
        for (UploadPart part : plan.parts) {
            uploadPart(recording, plan.contentType, part);
            completedParts += 1;
            int percent = (int) Math.min(95L, completedParts * 95L / plan.parts.size());
            progress.onProgress("uploading", percent);
        }

        session.authorizedRequest(
                "POST",
                "/api/cloud-recordings/" + encodePathSegment(plan.recordingId) + "/complete",
                null,
                null);
        progress.onProgress("uploading", 100);

        String workflowId = startWorkflow(session, plan.recordingId);
        progress.onProgress("transcribing", 0);
        WorkResult result = waitForTranscript(session, workflowId, progress);
        return new WorkResult(workflowId, result.suggestedFilename, result.text);
    }

    private static String startWorkflow(Session session, String recordingId) throws IOException {
        String boundary = "PhoneScribe-" + UUID.randomUUID();
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        writeFormField(body, boundary, "cloud_recording_id", recordingId);
        writeFormField(body, boundary, "destination", "gemini");
        writeFormField(body, boundary, "openai_model", "gpt-4o-transcribe");
        writeFormField(body, boundary, "word_timestamps", "false");
        writeFormField(body, boundary, "remove_silence", "false");
        writeFormField(body, boundary, "loudnorm", "true");
        writeFormField(body, boundary, "speech_filter", "true");
        writeFormField(body, boundary, "denoise", "false");
        body.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));

        HttpResult response = session.authorizedRequest(
                "POST",
                "/api/workflows",
                "multipart/form-data; boundary=" + boundary,
                body.toByteArray());
        JSONObject json = parseObject(response.body);
        String workflowId = json.optString("workflow_id", "");
        if (!workflowId.matches("[a-f0-9]{32}")) {
            throw new IOException("서버가 올바른 전사 작업 번호를 반환하지 않았습니다.");
        }
        return workflowId;
    }

    private static WorkResult waitForTranscript(
            Session session, String workflowId, ProgressListener progress) throws IOException {
        long deadline = System.currentTimeMillis() + WORKFLOW_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            HttpResult response;
            try {
                response = session.authorizedRequest(
                        "GET", "/api/workflows/" + encodePathSegment(workflowId), null, null);
            } catch (IOException failure) {
                if (failure instanceof ApiException
                        && !isTransient(((ApiException) failure).status)) {
                    throw failure;
                }
                progress.onProgress("transcribing", 0);
                sleep(3000L);
                continue;
            }
            JSONObject json = parseObject(response.body);
            String status = json.optString("status", "");
            if ("failed".equals(status)) {
                throw new IOException(safeError(json.optString("error", "전사 작업이 실패했습니다.")));
            }
            if ("complete".equals(status)) {
                JSONObject transcript = json.optJSONObject("transcript");
                if (transcript == null || !transcript.has("text")) {
                    throw new IOException("전사는 완료됐지만 TXT 결과가 없습니다.");
                }
                return new WorkResult(
                        workflowId,
                        transcript.optString("suggested_filename", "PhoneScribe_transcript"),
                        transcript.optString("text", ""));
            }
            JSONObject transcriptionProgress = json.optJSONObject("transcription_progress");
            int percent = transcriptionProgress == null
                    ? 0
                    : (int) Math.round(transcriptionProgress.optDouble("progress", 0.0) * 100.0);
            progress.onProgress("transcribing", Math.max(0, Math.min(99, percent)));
            sleep(3000L);
        }
        throw new IOException("전사가 4시간 안에 완료되지 않았습니다. 웹에서 작업 상태를 확인하세요.");
    }

    private static void uploadPart(File file, String contentType, UploadPart part) throws IOException {
        IOException lastFailure = null;
        for (long delay : RETRY_DELAYS_MS) {
            sleep(delay);
            HttpURLConnection connection = null;
            try {
                URL target = new URL(part.uploadUrl);
                if (!"https".equalsIgnoreCase(target.getProtocol())) {
                    throw new IOException("안전하지 않은 업로드 주소가 거절되었습니다.");
                }
                connection = (HttpURLConnection) target.openConnection();
                connection.setRequestMethod("PUT");
                connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
                connection.setReadTimeout(READ_TIMEOUT_MS);
                connection.setDoOutput(true);
                connection.setInstanceFollowRedirects(false);
                boolean hasContentType = false;
                for (String name : part.headers.keySet()) {
                    String value = part.headers.get(name);
                    connection.setRequestProperty(name, value);
                    if ("content-type".equalsIgnoreCase(name)) hasContentType = true;
                }
                if (!hasContentType) connection.setRequestProperty("Content-Type", contentType);
                connection.setFixedLengthStreamingMode(part.sizeBytes);

                try (RandomAccessFile input = new RandomAccessFile(file, "r");
                        OutputStream output = connection.getOutputStream()) {
                    input.seek(part.byteStart);
                    byte[] buffer = new byte[64 * 1024];
                    long remaining = part.sizeBytes;
                    while (remaining > 0L) {
                        int read = input.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                        if (read < 0) throw new IOException("녹음 파일이 업로드 중 변경되었습니다.");
                        output.write(buffer, 0, read);
                        remaining -= read;
                    }
                }

                int status = connection.getResponseCode();
                if (status >= 200 && status < 300) return;
                String detail = readResponse(connection, status);
                IOException failure = new ApiException(status, detailMessage(status, detail));
                if (!isTransient(status)) throw failure;
                lastFailure = failure;
            } catch (IOException failure) {
                if (failure instanceof ApiException
                        && !isTransient(((ApiException) failure).status)) {
                    throw failure;
                }
                lastFailure = failure;
            } finally {
                if (connection != null) connection.disconnect();
            }
        }
        throw lastFailure == null ? new IOException("녹음 업로드에 실패했습니다.") : lastFailure;
    }

    private static void writeFormField(
            ByteArrayOutputStream output, String boundary, String name, String value)
            throws IOException {
        output.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        output.write(("Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n")
                .getBytes(StandardCharsets.UTF_8));
        output.write(value.getBytes(StandardCharsets.UTF_8));
        output.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private static HttpResult request(
            String method,
            String url,
            String accessToken,
            String passcode,
            String contentType,
            byte[] body) throws IOException {
        HttpURLConnection connection = null;
        try {
            URL target = new URL(url);
            if (!"https".equalsIgnoreCase(target.getProtocol())) {
                throw new IOException("PhoneScribe는 HTTPS 서버만 연결할 수 있습니다.");
            }
            connection = (HttpURLConnection) target.openConnection();
            connection.setRequestMethod(method);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestProperty("Accept", "application/json");
            if (accessToken != null && !accessToken.isEmpty()) {
                connection.setRequestProperty("Authorization", "Bearer " + accessToken);
            }
            if (passcode != null && !passcode.isEmpty()) {
                connection.setRequestProperty("X-LocalMeetScribe-Passcode", passcode);
            }
            if (body != null) {
                connection.setDoOutput(true);
                connection.setRequestProperty(
                        "Content-Type", contentType == null ? "application/octet-stream" : contentType);
                connection.setFixedLengthStreamingMode(body.length);
                try (OutputStream output = connection.getOutputStream()) {
                    output.write(body);
                }
            }
            int status = connection.getResponseCode();
            String response = readResponse(connection, status);
            if (status < 200 || status >= 300) {
                throw new ApiException(status, detailMessage(status, response));
            }
            return new HttpResult(status, response);
        } finally {
            if (connection != null) connection.disconnect();
        }
    }

    private static String readResponse(HttpURLConnection connection, int status) throws IOException {
        InputStream stream = status >= 200 && status < 400
                ? connection.getInputStream()
                : connection.getErrorStream();
        if (stream == null) return "";
        try (InputStream input = stream; ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            int total = 0;
            while ((read = input.read(buffer)) != -1) {
                total += read;
                if (total > 64 * 1024 * 1024) {
                    throw new IOException("서버 응답이 허용 크기를 초과했습니다.");
                }
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    private static JSONObject parseObject(String body) throws IOException {
        try {
            return new JSONObject(body);
        } catch (JSONException error) {
            throw new IOException("서버가 올바른 JSON 응답을 반환하지 않았습니다.", error);
        }
    }

    private static String detailMessage(int status, String body) {
        try {
            String detail = new JSONObject(body).optString("detail", "");
            if (!detail.isEmpty()) return safeError(detail);
        } catch (JSONException ignored) {
            // Fall through to the status-only message so response bodies never leak into logs/UI.
        }
        return "PhoneScribe 서버 요청이 실패했습니다. (HTTP " + status + ")";
    }

    private static String safeError(String value) {
        String oneLine = value == null ? "" : value.replaceAll("[\\r\\n\\t]+", " ").trim();
        if (oneLine.isEmpty()) return "PhoneScribe 작업이 실패했습니다.";
        return oneLine.length() <= 240 ? oneLine : oneLine.substring(0, 240);
    }

    private static boolean isTransient(int status) {
        return status == 408 || status == 425 || status == 429 || status >= 500;
    }

    private static String encodePathSegment(String value) {
        StringBuilder encoded = new StringBuilder();
        for (byte character : value.getBytes(StandardCharsets.UTF_8)) {
            int unsigned = character & 0xff;
            if ((unsigned >= 'a' && unsigned <= 'z')
                    || (unsigned >= 'A' && unsigned <= 'Z')
                    || (unsigned >= '0' && unsigned <= '9')
                    || unsigned == '-' || unsigned == '_' || unsigned == '.' || unsigned == '~') {
                encoded.append((char) unsigned);
            } else {
                encoded.append(String.format(Locale.US, "%%%02X", unsigned));
            }
        }
        return encoded.toString();
    }

    private static void sleep(long millis) throws IOException {
        if (millis <= 0L) return;
        try {
            Thread.sleep(millis);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IOException("작업이 중단되었습니다.", interrupted);
        }
    }

    static final class Session {
        private final String baseUrl;
        private final String passcode;
        private String accessToken = "";

        Session(String baseUrl, String passcode) throws IOException {
            try {
                URI uri = URI.create(baseUrl);
                if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null) {
                    throw new IllegalArgumentException();
                }
            } catch (RuntimeException error) {
                throw new IOException("올바른 HTTPS PhoneScribe 서버 주소가 아닙니다.", error);
            }
            this.baseUrl = baseUrl.endsWith("/")
                    ? baseUrl.substring(0, baseUrl.length() - 1)
                    : baseUrl;
            this.passcode = passcode;
        }

        synchronized void refreshToken() throws IOException {
            HttpResult response = request(
                    "POST",
                    baseUrl + "/api/gemini-share/verify",
                    null,
                    passcode,
                    null,
                    null);
            JSONObject json = parseObject(response.body);
            if (!json.optBoolean("valid", false)) throw new IOException("공유 비밀번호가 올바르지 않습니다.");
            if (!json.optBoolean("key_ready", false)) {
                throw new IOException("PC에 Gemini API key가 설정되지 않았습니다.");
            }
            String token = json.optString("access_token", "");
            if (token.isEmpty()) throw new IOException("서버 인증 토큰을 받지 못했습니다.");
            accessToken = token;
        }

        HttpResult authorizedRequest(
                String method, String path, String contentType, byte[] body) throws IOException {
            if (accessToken.isEmpty()) refreshToken();
            try {
                return request(method, baseUrl + path, accessToken, null, contentType, body);
            } catch (ApiException error) {
                if (error.status != 401) throw error;
                refreshToken();
                return request(method, baseUrl + path, accessToken, null, contentType, body);
            }
        }
    }

    static final class WorkResult {
        final String workflowId;
        final String suggestedFilename;
        final String text;

        WorkResult(String workflowId, String suggestedFilename, String text) {
            this.workflowId = workflowId;
            this.suggestedFilename = suggestedFilename;
            this.text = text;
        }
    }

    static final class UploadPlan {
        final String recordingId;
        final String contentType;
        final List<UploadPart> parts;

        UploadPlan(String recordingId, String contentType, List<UploadPart> parts) {
            this.recordingId = recordingId;
            this.contentType = contentType;
            this.parts = Collections.unmodifiableList(new ArrayList<>(parts));
        }

        static UploadPlan fromJson(JSONObject json) throws IOException {
            String recordingId = json.optString("recording_id", "");
            String contentType = json.optString("content_type", "");
            JSONArray jsonParts = json.optJSONArray("parts");
            if (recordingId.isEmpty() || contentType.isEmpty() || jsonParts == null) {
                throw new IOException("클라우드 업로드 정보가 올바르지 않습니다.");
            }
            List<UploadPart> parts = new ArrayList<>();
            try {
                for (int index = 0; index < jsonParts.length(); index++) {
                    JSONObject part = jsonParts.getJSONObject(index);
                    JSONObject upload = part.getJSONObject("upload");
                    if (!"signed-put".equals(upload.optString("protocol"))) {
                        throw new IOException("지원하지 않는 클라우드 업로드 방식입니다.");
                    }
                    JSONObject jsonHeaders = upload.optJSONObject("headers");
                    java.util.Map<String, String> headers = new java.util.LinkedHashMap<>();
                    if (jsonHeaders != null) {
                        Iterator<String> names = jsonHeaders.keys();
                        while (names.hasNext()) {
                            String name = names.next();
                            headers.put(name, jsonHeaders.optString(name, ""));
                        }
                    }
                    parts.add(new UploadPart(
                            part.getInt("part_number"),
                            part.getLong("byte_start"),
                            part.getLong("byte_end"),
                            part.getLong("size_bytes"),
                            part.optString("object_path", ""),
                            upload.optString("url", ""),
                            headers));
                }
            } catch (JSONException error) {
                throw new IOException("클라우드 업로드 조각 정보가 올바르지 않습니다.", error);
            }
            parts.sort(Comparator.comparingInt(part -> part.partNumber));
            return new UploadPlan(recordingId, contentType, parts);
        }

        void validate(long fileSize) throws IOException {
            if (parts.isEmpty()) throw new IOException("클라우드 업로드 조각이 없습니다.");
            long expectedStart = 0L;
            for (int index = 0; index < parts.size(); index++) {
                UploadPart part = parts.get(index);
                if (part.partNumber != index
                        || part.byteStart != expectedStart
                        || part.byteEnd <= part.byteStart
                        || part.sizeBytes != part.byteEnd - part.byteStart
                        || part.sizeBytes > MAX_PART_SIZE
                        || part.objectPath.isEmpty()) {
                    throw new IOException("클라우드 업로드 범위가 올바르지 않습니다.");
                }
                try {
                    URL url = new URL(part.uploadUrl);
                    if (!"https".equalsIgnoreCase(url.getProtocol()) || url.getHost().isEmpty()) {
                        throw new IOException("안전하지 않은 클라우드 업로드 주소입니다.");
                    }
                } catch (RuntimeException error) {
                    throw new IOException("클라우드 업로드 주소가 올바르지 않습니다.", error);
                }
                expectedStart = part.byteEnd;
            }
            if (expectedStart != fileSize) {
                throw new IOException("클라우드 업로드 크기가 녹음 파일과 일치하지 않습니다.");
            }
        }
    }

    static final class UploadPart {
        final int partNumber;
        final long byteStart;
        final long byteEnd;
        final long sizeBytes;
        final String objectPath;
        final String uploadUrl;
        final java.util.Map<String, String> headers;

        UploadPart(
                int partNumber,
                long byteStart,
                long byteEnd,
                long sizeBytes,
                String objectPath,
                String uploadUrl,
                java.util.Map<String, String> headers) {
            this.partNumber = partNumber;
            this.byteStart = byteStart;
            this.byteEnd = byteEnd;
            this.sizeBytes = sizeBytes;
            this.objectPath = objectPath;
            this.uploadUrl = uploadUrl;
            this.headers = Collections.unmodifiableMap(new java.util.LinkedHashMap<>(headers));
        }
    }

    private static final class HttpResult {
        final int status;
        final String body;

        HttpResult(int status, String body) {
            this.status = status;
            this.body = body;
        }
    }

    private static final class ApiException extends IOException {
        final int status;

        ApiException(int status, String message) {
            super(message);
            this.status = status;
        }
    }
}
