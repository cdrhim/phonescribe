"""Optional cloud persistence adapters.

The transcription pipeline remains local-first. Cloud adapters are activated only
when explicitly configured and are kept separate from the local pipeline engines.
"""

from local_meetscribe.cloud.supabase import (
    CloudRecording,
    CloudTranscriptSegment,
    SignedRecordingUpload,
    SupabaseCloudClient,
    SupabaseCloudError,
)

__all__ = [
    "CloudRecording",
    "CloudTranscriptSegment",
    "SignedRecordingUpload",
    "SupabaseCloudClient",
    "SupabaseCloudError",
]
