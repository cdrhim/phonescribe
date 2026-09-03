package com.cdrhim.phonescribe;

import android.Manifest;
import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.net.URI;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int REQUEST_PERMISSIONS = 41;
    private static final String PREFS = "phonescribe_user_settings";

    private final Handler timer = new Handler(Looper.getMainLooper());
    private boolean receiverRegistered;
    private boolean continueAfterPermission;

    private EditText serverUrl;
    private EditText passcode;
    private TextView status;
    private TextView elapsed;
    private Button startButton;
    private Button stopButton;
    private Button retryButton;

    private final BroadcastReceiver stateReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            refreshState();
        }
    };

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            refreshElapsed();
            timer.postDelayed(this, 1000L);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        serverUrl = findViewById(R.id.serverUrl);
        passcode = findViewById(R.id.passcode);
        status = findViewById(R.id.status);
        elapsed = findViewById(R.id.elapsed);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        retryButton = findViewById(R.id.retryButton);
        Button webButton = findViewById(R.id.webButton);

        String savedServer = getSharedPreferences(PREFS, MODE_PRIVATE)
                .getString("server_url", BuildConfig.DEFAULT_API_BASE_URL);
        serverUrl.setText(savedServer);
        passcode.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);

        startButton.setOnClickListener(view -> requestStart(false));
        stopButton.setOnClickListener(view -> sendServiceAction(RecordingService.ACTION_STOP, false));
        retryButton.setOnClickListener(view -> requestStart(true));
        webButton.setOnClickListener(view ->
                startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(BuildConfig.WEB_APP_URL))));
        refreshState();
    }

    @Override
    protected void onStart() {
        super.onStart();
        IntentFilter filter = new IntentFilter(RecordingStateStore.ACTION_STATE_CHANGED);
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(stateReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(stateReceiver, filter);
        }
        receiverRegistered = true;
        timer.post(tick);
        refreshState();
    }

    @Override
    protected void onStop() {
        timer.removeCallbacks(tick);
        if (receiverRegistered) {
            unregisterReceiver(stateReceiver);
            receiverRegistered = false;
        }
        super.onStop();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_PERMISSIONS) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            showMessage("마이크 권한이 있어야 화면 잠금 녹음을 시작할 수 있습니다.");
            continueAfterPermission = false;
            return;
        }
        if (continueAfterPermission) {
            continueAfterPermission = false;
            requestStart(false);
        }
    }

    private void requestStart(boolean retry) {
        if (!retry && checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            continueAfterPermission = true;
            if (Build.VERSION.SDK_INT >= 33) {
                requestPermissions(
                        new String[] {
                            Manifest.permission.RECORD_AUDIO,
                            Manifest.permission.POST_NOTIFICATIONS
                        },
                        REQUEST_PERMISSIONS);
            } else {
                requestPermissions(
                        new String[] {Manifest.permission.RECORD_AUDIO}, REQUEST_PERMISSIONS);
            }
            return;
        }
        sendServiceAction(retry ? RecordingService.ACTION_RETRY : RecordingService.ACTION_START, true);
    }

    private void sendServiceAction(String action, boolean includeCredentials) {
        Intent intent = new Intent(this, RecordingService.class).setAction(action);
        if (includeCredentials) {
            String normalizedServer = normalizeServerUrl(serverUrl.getText().toString());
            String secret = passcode.getText().toString();
            if (normalizedServer == null) {
                showMessage("HTTPS PhoneScribe 서버 주소를 확인하세요.");
                return;
            }
            if (secret.isEmpty()) {
                showMessage("공유 비밀번호를 입력하세요.");
                return;
            }
            getSharedPreferences(PREFS, MODE_PRIVATE)
                    .edit()
                    .putString("server_url", normalizedServer)
                    .apply();
            intent.putExtra(RecordingService.EXTRA_SERVER_URL, normalizedServer);
            intent.putExtra(RecordingService.EXTRA_PASSCODE, secret);
            passcode.setText("");
        }

        if (Build.VERSION.SDK_INT >= 26
                && !RecordingService.ACTION_STOP.equals(action)) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
    }

    private void refreshState() {
        RecordingStateStore.Snapshot snapshot = RecordingStateStore.read(this);
        status.setText(snapshot.message);
        boolean busy = snapshot.isBusy();
        startButton.setEnabled(!busy);
        stopButton.setEnabled(RecordingStateStore.RECORDING.equals(snapshot.state));
        retryButton.setVisibility(
                !busy && RecordingStorage.hasPending(this) ? View.VISIBLE : View.GONE);
        refreshElapsed();
    }

    private void refreshElapsed() {
        RecordingStateStore.Snapshot snapshot = RecordingStateStore.read(this);
        if (snapshot.startedAtMillis <= 0L) {
            elapsed.setText("00:00:00");
            return;
        }
        long total = Math.max(0L, (System.currentTimeMillis() - snapshot.startedAtMillis) / 1000L);
        long hours = total / 3600L;
        long minutes = (total % 3600L) / 60L;
        long seconds = total % 60L;
        elapsed.setText(String.format(Locale.US, "%02d:%02d:%02d", hours, minutes, seconds));
    }

    private static String normalizeServerUrl(String raw) {
        try {
            String value = raw == null ? "" : raw.trim();
            while (value.endsWith("/")) value = value.substring(0, value.length() - 1);
            URI uri = URI.create(value);
            if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null) return null;
            if (uri.getRawQuery() != null || uri.getRawFragment() != null) return null;
            return value;
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    private void showMessage(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
        status.setText(message);
    }
}
