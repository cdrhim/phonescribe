package com.cdrhim.phonescribe;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;

final class RecordingStateStore {
    static final String ACTION_STATE_CHANGED = "com.cdrhim.phonescribe.STATE_CHANGED";

    static final String IDLE = "idle";
    static final String AUTHENTICATING = "authenticating";
    static final String RECORDING = "recording";
    static final String FINALIZING = "finalizing";
    static final String UPLOADING = "uploading";
    static final String TRANSCRIBING = "transcribing";
    static final String COMPLETE = "complete";
    static final String FAILED = "failed";

    private static final String PREFS = "phonescribe_recording_state";

    private RecordingStateStore() {}

    static void update(
            Context context,
            String state,
            String message,
            long startedAtMillis,
            String pendingFileName) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putString("state", state)
                .putString("message", message)
                .putLong("started_at", startedAtMillis)
                .putString("pending_file", pendingFileName == null ? "" : pendingFileName)
                .apply();
        Intent changed = new Intent(ACTION_STATE_CHANGED).setPackage(context.getPackageName());
        context.sendBroadcast(changed);
    }

    static Snapshot read(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        return new Snapshot(
                prefs.getString("state", IDLE),
                prefs.getString("message", "준비되었습니다."),
                prefs.getLong("started_at", 0L),
                prefs.getString("pending_file", ""));
    }

    static final class Snapshot {
        final String state;
        final String message;
        final long startedAtMillis;
        final String pendingFileName;

        Snapshot(String state, String message, long startedAtMillis, String pendingFileName) {
            this.state = state;
            this.message = message;
            this.startedAtMillis = startedAtMillis;
            this.pendingFileName = pendingFileName;
        }

        boolean isBusy() {
            return AUTHENTICATING.equals(state)
                    || RECORDING.equals(state)
                    || FINALIZING.equals(state)
                    || UPLOADING.equals(state)
                    || TRANSCRIBING.equals(state);
        }
    }
}
