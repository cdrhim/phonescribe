package com.cdrhim.phonescribe;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public final class RecordingService extends Service {
    static final String ACTION_START = "com.cdrhim.phonescribe.action.START";
    static final String ACTION_STOP = "com.cdrhim.phonescribe.action.STOP";
    static final String ACTION_RETRY = "com.cdrhim.phonescribe.action.RETRY";
    static final String EXTRA_SERVER_URL = "server_url";
    static final String EXTRA_PASSCODE = "passcode";

    private static final String RECORDING_CHANNEL_ID = "phonescribe-recording";
    private static final String PROCESSING_CHANNEL_ID = "phonescribe-processing";
    private static final int FOREGROUND_NOTIFICATION_ID = 2101;
    private static final int RESULT_NOTIFICATION_ID = 2102;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean operationRunning = new AtomicBoolean(false);

    private volatile MediaRecorder recorder;
    private File pendingFile;
    private PhoneScribeApi.Session session;
    private String serverUrl = "";
    private String passcode = "";
    private long recordingStartedAt;
    private PowerManager.WakeLock cpuWakeLock;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannels();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? "" : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            scheduleStopAndUpload();
            return START_NOT_STICKY;
        }
        if (ACTION_START.equals(action) || ACTION_RETRY.equals(action)) {
            if (!operationRunning.compareAndSet(false, true)) return START_NOT_STICKY;
            serverUrl = intent.getStringExtra(EXTRA_SERVER_URL);
            passcode = intent.getStringExtra(EXTRA_PASSCODE);
            if (serverUrl == null) serverUrl = "";
            if (passcode == null) passcode = "";
            boolean retry = ACTION_RETRY.equals(action);
            enterForeground(
                    retry ? ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                            : ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
                    buildForegroundNotification(
                            retry ? "최근 녹음을 다시 전송하고 있습니다." : "비밀번호를 확인하고 있습니다.",
                            false,
                            retry ? PROCESSING_CHANNEL_ID : RECORDING_CHANNEL_ID));
            acquireCpuWakeLock();
            executor.execute(() -> {
                try {
                    if (retry) authenticateAndRetry();
                    else authenticateAndRecord();
                } finally {
                    operationRunning.set(false);
                }
            });
        }
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        MediaRecorder activeRecorder = recorder;
        recorder = null;
        if (activeRecorder != null) {
            try {
                activeRecorder.stop();
            } catch (RuntimeException ignored) {
                // An interrupted recorder may not have enough samples to finalize.
            }
            activeRecorder.release();
            RecordingStateStore.update(
                    this,
                    RecordingStateStore.FAILED,
                    "Android가 녹음 서비스를 종료했습니다. 남아 있는 녹음은 다시 전송할 수 있습니다.",
                    0L,
                    pendingFile == null ? "" : pendingFile.getName());
        }
        releaseCpuWakeLock();
        executor.shutdownNow();
        passcode = "";
        super.onDestroy();
    }

    private void authenticateAndRecord() {
        try {
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                throw new IOException("마이크 권한이 없습니다.");
            }
            updateState(
                    RecordingStateStore.AUTHENTICATING,
                    "PhoneScribe 서버와 비밀번호를 확인하고 있습니다.",
                    0L,
                    null);
            session = PhoneScribeApi.verify(serverUrl, passcode);
            pendingFile = RecordingStorage.createPendingRecording(this);

            MediaRecorder prepared = Build.VERSION.SDK_INT >= 31
                    ? new MediaRecorder(this)
                    : new MediaRecorder();
            prepared.setAudioSource(MediaRecorder.AudioSource.MIC);
            prepared.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            prepared.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            prepared.setAudioChannels(1);
            prepared.setAudioSamplingRate(44_100);
            prepared.setAudioEncodingBitRate(64_000);
            prepared.setOutputFile(pendingFile.getAbsolutePath());
            prepared.prepare();
            prepared.start();
            recorder = prepared;
            recordingStartedAt = System.currentTimeMillis();
            updateState(
                    RecordingStateStore.RECORDING,
                    "녹음 중 · 이제 화면을 잠가도 녹음이 계속됩니다.",
                    recordingStartedAt,
                    pendingFile.getName());
            notifyForeground("녹음 중 · 화면을 잠가도 계속됩니다.", true, RECORDING_CHANNEL_ID,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
        } catch (Exception error) {
            failAndStop(userMessage(error, "녹음을 시작하지 못했습니다."));
        }
    }

    private void scheduleStopAndUpload() {
        if (recorder == null) return;
        if (!operationRunning.compareAndSet(false, true)) return;
        executor.execute(() -> {
            try {
                stopAndUpload();
            } finally {
                operationRunning.set(false);
            }
        });
    }

    private void stopAndUpload() {
        updateState(
                RecordingStateStore.FINALIZING,
                "녹음을 안전하게 저장하고 있습니다.",
                recordingStartedAt,
                pendingFile == null ? null : pendingFile.getName());
        MediaRecorder activeRecorder = recorder;
        recorder = null;
        try {
            if (activeRecorder == null || pendingFile == null) {
                throw new IOException("진행 중인 녹음이 없습니다.");
            }
            try {
                activeRecorder.stop();
            } catch (RuntimeException error) {
                throw new IOException("녹음 시간이 너무 짧거나 녹음이 중단되었습니다.", error);
            } finally {
                activeRecorder.release();
            }
            if (pendingFile.length() <= 0L) throw new IOException("저장된 녹음이 비어 있습니다.");
            RecordingStorage.publishRecording(this, pendingFile);
            enterForeground(
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                    buildForegroundNotification(
                            "녹음을 Supabase로 전송하고 있습니다.", false, PROCESSING_CHANNEL_ID));
            processPendingRecording(pendingFile);
        } catch (Exception error) {
            failAndStop(userMessage(error, "녹음 저장 또는 전송에 실패했습니다."));
        }
    }

    private void authenticateAndRetry() {
        try {
            File latest = RecordingStorage.latestPending(this);
            if (latest == null || latest.length() <= 0L) {
                throw new IOException("다시 전송할 녹음 파일이 없습니다.");
            }
            pendingFile = latest;
            updateState(
                    RecordingStateStore.AUTHENTICATING,
                    "비밀번호를 다시 확인하고 있습니다.",
                    0L,
                    pendingFile.getName());
            session = PhoneScribeApi.verify(serverUrl, passcode);
            RecordingStorage.publishRecording(this, pendingFile);
            processPendingRecording(pendingFile);
        } catch (Exception error) {
            failAndStop(userMessage(error, "최근 녹음을 다시 전송하지 못했습니다."));
        }
    }

    private void processPendingRecording(File source) throws IOException {
        updateState(
                RecordingStateStore.UPLOADING,
                "Supabase 업로드를 시작합니다.",
                0L,
                source.getName());
        PhoneScribeApi.WorkResult result = PhoneScribeApi.uploadAndTranscribe(
                session,
                source,
                (stage, percent) -> {
                    if ("uploading".equals(stage)) {
                        updateState(
                                RecordingStateStore.UPLOADING,
                                "Supabase 업로드 중 · " + percent + "%",
                                0L,
                                source.getName());
                        notifyForeground(
                                "녹음 업로드 중 · " + percent + "%",
                                false,
                                PROCESSING_CHANNEL_ID,
                                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
                    } else {
                        updateState(
                                RecordingStateStore.TRANSCRIBING,
                                "PC 전사 중 · " + percent + "% · 화면을 꺼도 계속됩니다.",
                                0L,
                                source.getName());
                        notifyForeground(
                                "PC 전사 중 · " + percent + "%",
                                false,
                                PROCESSING_CHANNEL_ID,
                                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
                    }
                });

        String savedName = RecordingStorage.saveTranscript(
                this, result.suggestedFilename, result.text);
        if (!source.delete() && source.exists()) {
            throw new IOException("전사는 완료됐지만 임시 녹음 정리에 실패했습니다.");
        }
        pendingFile = null;
        updateState(
                RecordingStateStore.COMPLETE,
                "완료 · Download/PhoneScribe/" + savedName + "에 TXT를 저장했습니다.",
                0L,
                null);
        finishForegroundWithResult(
                "PhoneScribe 완료",
                "녹음·전사·TXT 저장을 완료했습니다.");
        releaseCpuWakeLock();
        passcode = "";
        stopSelf();
    }

    private void failAndStop(String message) {
        updateState(
                RecordingStateStore.FAILED,
                message + " 앱에서 최근 녹음 다시 전송을 누를 수 있습니다.",
                0L,
                pendingFile == null ? null : pendingFile.getName());
        finishForegroundWithResult("PhoneScribe 확인 필요", message);
        releaseCpuWakeLock();
        passcode = "";
        stopSelf();
    }

    private void updateState(
            String state, String message, long startedAt, String pendingFileName) {
        RecordingStateStore.update(this, state, message, startedAt, pendingFileName);
    }

    private void createNotificationChannels() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        NotificationChannel recording = new NotificationChannel(
                RECORDING_CHANNEL_ID,
                getString(R.string.recording_channel),
                NotificationManager.IMPORTANCE_LOW);
        recording.setDescription("화면 잠금 중에도 진행되는 PhoneScribe 녹음");
        recording.setSound(null, null);
        manager.createNotificationChannel(recording);

        NotificationChannel processing = new NotificationChannel(
                PROCESSING_CHANNEL_ID,
                getString(R.string.processing_channel),
                NotificationManager.IMPORTANCE_DEFAULT);
        processing.setDescription("녹음 업로드와 전사 완료 상태");
        manager.createNotificationChannel(processing);
    }

    private Notification buildForegroundNotification(
            String message, boolean includeStop, String channelId) {
        Intent openIntent = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent open = PendingIntent.getActivity(
                this,
                0,
                openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = new Notification.Builder(this, channelId)
                .setSmallIcon(R.drawable.ic_mic)
                .setContentTitle("PhoneScribe")
                .setContentText(message)
                .setContentIntent(open)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setCategory(Notification.CATEGORY_SERVICE);
        if (includeStop) {
            Intent stopIntent = new Intent(this, RecordingService.class).setAction(ACTION_STOP);
            PendingIntent stop = PendingIntent.getService(
                    this,
                    1,
                    stopIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
            builder.addAction(new Notification.Action.Builder(
                    R.drawable.ic_mic, "녹음 정지 · 자동 전사", stop).build());
        }
        return builder.build();
    }

    private void enterForeground(int serviceType, Notification notification) {
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(FOREGROUND_NOTIFICATION_ID, notification, serviceType);
        } else {
            startForeground(FOREGROUND_NOTIFICATION_ID, notification);
        }
    }

    private void notifyForeground(
            String message, boolean includeStop, String channelId, int serviceType) {
        enterForeground(serviceType, buildForegroundNotification(message, includeStop, channelId));
    }

    private void finishForegroundWithResult(String title, String message) {
        if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_REMOVE);
        else stopForeground(true);
        Intent openIntent = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent open = PendingIntent.getActivity(
                this,
                2,
                openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, PROCESSING_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_mic)
                .setContentTitle(title)
                .setContentText(message)
                .setContentIntent(open)
                .setAutoCancel(true)
                .build();
        getSystemService(NotificationManager.class).notify(RESULT_NOTIFICATION_ID, notification);
    }

    private void acquireCpuWakeLock() {
        if (cpuWakeLock != null && cpuWakeLock.isHeld()) return;
        PowerManager manager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        cpuWakeLock = manager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK, "PhoneScribe::RecordAndUpload");
        cpuWakeLock.setReferenceCounted(false);
        cpuWakeLock.acquire();
    }

    private void releaseCpuWakeLock() {
        if (cpuWakeLock != null && cpuWakeLock.isHeld()) cpuWakeLock.release();
        cpuWakeLock = null;
    }

    private static String userMessage(Throwable error, String fallback) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) return fallback;
        String oneLine = message.replaceAll("[\\r\\n\\t]+", " ").trim();
        return oneLine.length() <= 240 ? oneLine : oneLine.substring(0, 240);
    }
}
