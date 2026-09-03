package com.cdrhim.phonescribe;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

final class RecordingStorage {
    private RecordingStorage() {}

    static File createPendingRecording(Context context) throws IOException {
        File directory = pendingDirectory(context);
        if (!directory.isDirectory() && !directory.mkdirs()) {
            throw new IOException("녹음 임시 폴더를 만들 수 없습니다.");
        }
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        return new File(directory, "PhoneScribe_" + stamp + ".m4a");
    }

    static File latestPending(Context context) {
        File[] files = pendingDirectory(context).listFiles(
                file -> file.isFile() && file.getName().toLowerCase(Locale.US).endsWith(".m4a"));
        if (files == null || files.length == 0) return null;
        File latest = files[0];
        for (File file : files) {
            if (file.lastModified() > latest.lastModified()) latest = file;
        }
        return latest;
    }

    static boolean hasPending(Context context) {
        return latestPending(context) != null;
    }

    static void publishRecording(Context context, File source) throws IOException {
        if (recordingExists(context, source.getName())) return;
        ContentValues values = new ContentValues();
        values.put(MediaStore.Audio.Media.DISPLAY_NAME, source.getName());
        values.put(MediaStore.Audio.Media.MIME_TYPE, "audio/mp4");
        values.put(
                MediaStore.Audio.Media.RELATIVE_PATH,
                "Recordings/PhoneScribe");
        values.put(MediaStore.Audio.Media.IS_PENDING, 1);

        ContentResolver resolver = context.getContentResolver();
        Uri target = resolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values);
        if (target == null) throw new IOException("휴대폰 녹음 폴더를 열 수 없습니다.");
        boolean complete = false;
        try (FileInputStream input = new FileInputStream(source);
                OutputStream output = resolver.openOutputStream(target, "w")) {
            if (output == null) throw new IOException("휴대폰 녹음 파일을 만들 수 없습니다.");
            copy(input, output);
            complete = true;
        } finally {
            if (!complete) resolver.delete(target, null, null);
        }

        ContentValues ready = new ContentValues();
        ready.put(MediaStore.Audio.Media.IS_PENDING, 0);
        resolver.update(target, ready, null, null);
    }

    private static boolean recordingExists(Context context, String displayName) {
        String relativePath = "Recordings/PhoneScribe/";
        String selection = MediaStore.Audio.Media.DISPLAY_NAME + " = ? AND "
                + MediaStore.Audio.Media.RELATIVE_PATH + " = ?";
        try (Cursor cursor = context.getContentResolver().query(
                MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                new String[] {MediaStore.Audio.Media._ID},
                selection,
                new String[] {displayName, relativePath},
                null)) {
            return cursor != null && cursor.moveToFirst();
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    static String saveTranscript(Context context, String suggestedName, String text)
            throws IOException {
        String filename = safeTxtFilename(suggestedName);
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
        values.put(MediaStore.MediaColumns.MIME_TYPE, "text/plain");
        values.put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOWNLOADS + "/PhoneScribe");
        values.put(MediaStore.MediaColumns.IS_PENDING, 1);

        ContentResolver resolver = context.getContentResolver();
        Uri target = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (target == null) throw new IOException("다운로드 폴더에 TXT를 만들 수 없습니다.");
        boolean complete = false;
        try (OutputStream output = resolver.openOutputStream(target, "w")) {
            if (output == null) throw new IOException("TXT 파일을 열 수 없습니다.");
            output.write(text.getBytes(StandardCharsets.UTF_8));
            complete = true;
        } finally {
            if (!complete) resolver.delete(target, null, null);
        }

        ContentValues ready = new ContentValues();
        ready.put(MediaStore.MediaColumns.IS_PENDING, 0);
        resolver.update(target, ready, null, null);
        return filename;
    }

    private static File pendingDirectory(Context context) {
        return new File(context.getFilesDir(), "pending-recordings");
    }

    private static String safeTxtFilename(String suggestedName) {
        String name = suggestedName == null ? "PhoneScribe_transcript" : suggestedName.trim();
        name = name.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_");
        name = name.replaceAll("^[. ]+|[. ]+$", "");
        if (name.isEmpty()) name = "PhoneScribe_transcript";
        if (!name.toLowerCase(Locale.US).endsWith(".txt")) name += ".txt";
        return name.length() <= 120 ? name : name.substring(0, 116) + ".txt";
    }

    private static void copy(FileInputStream input, OutputStream output) throws IOException {
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = input.read(buffer)) != -1) output.write(buffer, 0, read);
    }
}
