package com.cdrhim.phonescribe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.json.JSONObject;
import org.junit.Test;

import java.io.IOException;

public final class PhoneScribeApiTest {
    @Test
    public void acceptsContiguousSignedPutPlan() throws Exception {
        JSONObject json = new JSONObject("""
                {
                  "recording_id":"abc123",
                  "content_type":"audio/mp4",
                  "parts":[
                    {
                      "part_number":0,
                      "byte_start":0,
                      "byte_end":4,
                      "size_bytes":4,
                      "object_path":"abc/part-00000",
                      "upload":{
                        "protocol":"signed-put",
                        "url":"https://example.supabase.co/storage/v1/object/upload/sign/x",
                        "headers":{"x-upsert":"false"}
                      }
                    },
                    {
                      "part_number":1,
                      "byte_start":4,
                      "byte_end":7,
                      "size_bytes":3,
                      "object_path":"abc/part-00001",
                      "upload":{
                        "protocol":"signed-put",
                        "url":"https://example.supabase.co/storage/v1/object/upload/sign/y",
                        "headers":{}
                      }
                    }
                  ]
                }
                """);

        PhoneScribeApi.UploadPlan plan = PhoneScribeApi.UploadPlan.fromJson(json);
        plan.validate(7L);

        assertEquals("abc123", plan.recordingId);
        assertEquals(2, plan.parts.size());
        assertEquals("false", plan.parts.get(0).headers.get("x-upsert"));
    }

    @Test
    public void rejectsGapBetweenParts() throws Exception {
        JSONObject json = descriptorWithRange(0L, 4L, 4L, 5L, 7L, 2L);
        PhoneScribeApi.UploadPlan plan = PhoneScribeApi.UploadPlan.fromJson(json);
        assertThrows(IOException.class, () -> plan.validate(7L));
    }

    @Test
    public void rejectsOversizedPart() throws Exception {
        long oversized = 6L * 1024L * 1024L + 1L;
        JSONObject json = descriptorWithOnePart(oversized, "https://example.supabase.co/upload");
        PhoneScribeApi.UploadPlan plan = PhoneScribeApi.UploadPlan.fromJson(json);
        assertThrows(IOException.class, () -> plan.validate(oversized));
    }

    @Test
    public void rejectsNonHttpsSignedUpload() throws Exception {
        JSONObject json = descriptorWithOnePart(4L, "http://example.supabase.co/upload");
        PhoneScribeApi.UploadPlan plan = PhoneScribeApi.UploadPlan.fromJson(json);
        assertThrows(IOException.class, () -> plan.validate(4L));
    }

    @Test
    public void rejectsNonHttpsApiBase() {
        assertThrows(IOException.class, () -> new PhoneScribeApi.Session("http://10.0.0.2:8766", "secret"));
    }

    private static JSONObject descriptorWithOnePart(long size, String uploadUrl) throws Exception {
        return new JSONObject()
                .put("recording_id", "abc123")
                .put("content_type", "audio/mp4")
                .put("parts", new org.json.JSONArray().put(part(0, 0L, size, size, uploadUrl)));
    }

    private static JSONObject descriptorWithRange(
            long firstStart,
            long firstEnd,
            long firstSize,
            long secondStart,
            long secondEnd,
            long secondSize) throws Exception {
        return new JSONObject()
                .put("recording_id", "abc123")
                .put("content_type", "audio/mp4")
                .put("parts", new org.json.JSONArray()
                        .put(part(0, firstStart, firstEnd, firstSize, "https://example.com/a"))
                        .put(part(1, secondStart, secondEnd, secondSize, "https://example.com/b")));
    }

    private static JSONObject part(
            int number, long start, long end, long size, String uploadUrl) throws Exception {
        return new JSONObject()
                .put("part_number", number)
                .put("byte_start", start)
                .put("byte_end", end)
                .put("size_bytes", size)
                .put("object_path", "abc/part-" + number)
                .put("upload", new JSONObject()
                        .put("protocol", "signed-put")
                        .put("url", uploadUrl)
                        .put("headers", new JSONObject()));
    }
}
