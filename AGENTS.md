# LocalMeetScribe Engineering Notes

- The default pipeline must run with mocked engines and no model downloads.
- Real transcription must stay local. Do not add cloud transcription APIs.
- Raw transcript text is immutable. Review edits update `text_clean` only.
- Do not log transcript contents or uploaded file text. Log job IDs and stages.
- Keep heavy model integrations behind optional extras and clear errors.
- Tests must use mocked engines and tiny generated fixtures.
