begin;

create extension if not exists pgcrypto with schema extensions;

create schema if not exists phonescribe_private;
revoke all on schema phonescribe_private from public, anon, authenticated;

create table public.recordings (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references auth.users (id) on delete cascade,
    bucket_id text not null default 'recordings'
        constraint recordings_bucket_id_check check (bucket_id = 'recordings'),
    object_path text not null,
    original_filename text not null
        constraint recordings_original_filename_check
        check (char_length(original_filename) between 1 and 512),
    file_extension text not null
        constraint recordings_file_extension_check
        check (file_extension ~ '^[a-z0-9]{1,10}$'),
    mime_type text not null
        constraint recordings_mime_type_check
        check (char_length(mime_type) between 3 and 255),
    source_kind text not null default 'upload'
        constraint recordings_source_kind_check
        check (source_kind in ('upload', 'browser_recording')),
    size_bytes bigint not null
        constraint recordings_size_bytes_check check (size_bytes > 0),
    part_count integer not null default 1
        constraint recordings_part_count_check check (part_count between 1 and 10000),
    part_size_bytes bigint not null default 6291456
        constraint recordings_part_size_bytes_check
        check (part_size_bytes between 1 and 25165824),
    duration_ms bigint
        constraint recordings_duration_ms_check check (duration_ms is null or duration_ms >= 0),
    sample_rate_hz integer
        constraint recordings_sample_rate_check check (sample_rate_hz is null or sample_rate_hz > 0),
    channels smallint
        constraint recordings_channels_check check (channels is null or channels between 1 and 32),
    content_sha256 text
        constraint recordings_content_sha256_check
        check (content_sha256 is null or content_sha256 ~ '^[0-9a-f]{64}$'),
    storage_status text not null default 'pending_upload'
        constraint recordings_storage_status_check
        check (storage_status in ('pending_upload', 'ready', 'deleting', 'deleted', 'failed')),
    retention_until timestamptz not null default (now() + interval '30 days'),
    upload_completed_at timestamptz,
    deleted_at timestamptz,
    metadata jsonb not null default '{}'::jsonb
        constraint recordings_metadata_object_check check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint recordings_object_path_unique unique (bucket_id, object_path),
    constraint recordings_object_path_matches_row check (
        object_path = coalesce(owner_id::text, 'shared') || '/' || id::text
    ),
    constraint recordings_parts_cover_file check (
        size_bytes <= part_count::bigint * part_size_bytes
        and (
            part_count = 1
            or size_bytes > (part_count - 1)::bigint * part_size_bytes
        )
    ),
    constraint recordings_deleted_state_check check (
        deleted_at is null or storage_status = 'deleted'
    )
);

create table public.transcription_jobs (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references public.recordings (id) on delete cascade,
    workflow_id text not null unique
        constraint transcription_jobs_workflow_id_check
        check (workflow_id ~ '^[A-Za-z0-9_-]{8,128}$'),
    status text not null default 'queued'
        constraint transcription_jobs_status_check
        check (status in ('queued', 'optimizing', 'transcribing', 'complete', 'failed', 'cancelled')),
    stage text not null default 'queued'
        constraint transcription_jobs_stage_check check (char_length(stage) between 1 and 100),
    progress double precision not null default 0
        constraint transcription_jobs_progress_check check (progress between 0 and 1),
    completed_parts integer not null default 0
        constraint transcription_jobs_completed_parts_check check (completed_parts >= 0),
    total_parts integer not null default 0
        constraint transcription_jobs_total_parts_check
        check (total_parts >= 0 and completed_parts <= total_parts),
    attempt_count integer not null default 0
        constraint transcription_jobs_attempt_count_check check (attempt_count >= 0),
    error_code text
        constraint transcription_jobs_error_code_check
        check (error_code is null or char_length(error_code) between 1 and 100),
    error_message text
        constraint transcription_jobs_error_message_check
        check (error_message is null or char_length(error_message) <= 2000),
    worker_id text
        constraint transcription_jobs_worker_id_check
        check (worker_id is null or char_length(worker_id) between 1 and 255),
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint transcription_jobs_id_recording_unique unique (id, recording_id)
);

create table public.transcripts (
    id uuid primary key default gen_random_uuid(),
    recording_id uuid not null references public.recordings (id) on delete cascade,
    job_id uuid not null unique,
    provider text not null
        constraint transcripts_provider_check check (char_length(provider) between 1 and 100),
    model_name text not null
        constraint transcripts_model_name_check check (char_length(model_name) between 1 and 255),
    language text not null default 'unknown'
        constraint transcripts_language_check check (language in ('auto', 'ko', 'en', 'mixed', 'unknown')),
    text_raw text not null,
    text_clean text not null,
    metadata jsonb not null default '{}'::jsonb
        constraint transcripts_metadata_object_check check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint transcripts_job_recording_fk
        foreign key (job_id, recording_id)
        references public.transcription_jobs (id, recording_id)
        on delete cascade
);

create table public.transcript_segments (
    id uuid primary key default gen_random_uuid(),
    transcript_id uuid not null references public.transcripts (id) on delete cascade,
    segment_index integer not null
        constraint transcript_segments_index_check check (segment_index >= 0),
    start_ms bigint not null
        constraint transcript_segments_start_check check (start_ms >= 0),
    end_ms bigint not null,
    speaker_label text not null default 'SPEAKER_00'
        constraint transcript_segments_speaker_check
        check (char_length(speaker_label) between 1 and 100),
    language text not null default 'unknown'
        constraint transcript_segments_language_check
        check (language in ('ko', 'en', 'mixed', 'unknown')),
    text_raw text not null,
    text_clean text not null,
    confidence double precision
        constraint transcript_segments_confidence_check
        check (confidence is null or confidence between 0 and 1),
    needs_review boolean not null default false,
    overlap boolean not null default false,
    words_raw jsonb not null default '[]'::jsonb
        constraint transcript_segments_words_array_check check (jsonb_typeof(words_raw) = 'array'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint transcript_segments_end_check check (end_ms >= start_ms),
    constraint transcript_segments_transcript_index_unique unique (transcript_id, segment_index)
);

create index recordings_owner_created_idx
    on public.recordings (owner_id, created_at desc)
    where deleted_at is null;

create index recordings_retention_due_idx
    on public.recordings (retention_until, id)
    where deleted_at is null and storage_status not in ('deleting', 'deleted');

create index transcription_jobs_recording_created_idx
    on public.transcription_jobs (recording_id, created_at desc);

create unique index transcription_jobs_one_active_per_recording_idx
    on public.transcription_jobs (recording_id)
    where status in ('queued', 'optimizing', 'transcribing');

create index transcripts_recording_created_idx
    on public.transcripts (recording_id, created_at desc);

create index transcript_segments_timeline_idx
    on public.transcript_segments (transcript_id, segment_index);

create or replace function phonescribe_private.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
    new.updated_at := statement_timestamp();
    return new;
end;
$$;

create or replace function phonescribe_private.guard_transcript_raw()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
    if new.recording_id is distinct from old.recording_id
       or new.job_id is distinct from old.job_id
       or new.provider is distinct from old.provider
       or new.model_name is distinct from old.model_name
       or new.language is distinct from old.language
       or new.text_raw is distinct from old.text_raw
       or new.metadata is distinct from old.metadata then
        raise exception 'raw transcript fields are immutable'
            using errcode = '22000';
    end if;
    return new;
end;
$$;

create or replace function phonescribe_private.guard_segment_raw()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
    if new.transcript_id is distinct from old.transcript_id
       or new.segment_index is distinct from old.segment_index
       or new.start_ms is distinct from old.start_ms
       or new.end_ms is distinct from old.end_ms
       or new.speaker_label is distinct from old.speaker_label
       or new.language is distinct from old.language
       or new.text_raw is distinct from old.text_raw
       or new.confidence is distinct from old.confidence
       or new.overlap is distinct from old.overlap
       or new.words_raw is distinct from old.words_raw then
        raise exception 'raw transcript segment fields are immutable'
            using errcode = '22000';
    end if;
    return new;
end;
$$;

revoke all on function phonescribe_private.set_updated_at() from public, anon, authenticated;
revoke all on function phonescribe_private.guard_transcript_raw() from public, anon, authenticated;
revoke all on function phonescribe_private.guard_segment_raw() from public, anon, authenticated;

create trigger recordings_set_updated_at
before update on public.recordings
for each row execute function phonescribe_private.set_updated_at();

create trigger transcription_jobs_set_updated_at
before update on public.transcription_jobs
for each row execute function phonescribe_private.set_updated_at();

create trigger transcripts_guard_raw
before update on public.transcripts
for each row execute function phonescribe_private.guard_transcript_raw();

create trigger transcripts_set_updated_at
before update on public.transcripts
for each row execute function phonescribe_private.set_updated_at();

create trigger transcript_segments_guard_raw
before update on public.transcript_segments
for each row execute function phonescribe_private.guard_segment_raw();

create trigger transcript_segments_set_updated_at
before update on public.transcript_segments
for each row execute function phonescribe_private.set_updated_at();

alter table public.recordings enable row level security;
alter table public.transcription_jobs enable row level security;
alter table public.transcripts enable row level security;
alter table public.transcript_segments enable row level security;

revoke all on public.recordings from anon, authenticated;
revoke all on public.transcription_jobs from anon, authenticated;
revoke all on public.transcripts from anon, authenticated;
revoke all on public.transcript_segments from anon, authenticated;

grant select on public.recordings to authenticated;
grant select on public.transcription_jobs to authenticated;
grant select on public.transcripts to authenticated;
grant select on public.transcript_segments to authenticated;
grant update (text_clean) on public.transcripts to authenticated;
grant update (text_clean, needs_review) on public.transcript_segments to authenticated;

grant all on public.recordings to service_role;
grant all on public.transcription_jobs to service_role;
grant all on public.transcripts to service_role;
grant all on public.transcript_segments to service_role;

create policy recordings_owner_select
on public.recordings
for select
to authenticated
using (owner_id = (select auth.uid()));

create policy transcription_jobs_owner_select
on public.transcription_jobs
for select
to authenticated
using (
    exists (
        select 1
        from public.recordings
        where recordings.id = transcription_jobs.recording_id
          and recordings.owner_id = (select auth.uid())
    )
);

create policy transcripts_owner_select
on public.transcripts
for select
to authenticated
using (
    exists (
        select 1
        from public.recordings
        where recordings.id = transcripts.recording_id
          and recordings.owner_id = (select auth.uid())
    )
);

create policy transcripts_owner_update_clean
on public.transcripts
for update
to authenticated
using (
    exists (
        select 1
        from public.recordings
        where recordings.id = transcripts.recording_id
          and recordings.owner_id = (select auth.uid())
    )
)
with check (
    exists (
        select 1
        from public.recordings
        where recordings.id = transcripts.recording_id
          and recordings.owner_id = (select auth.uid())
    )
);

create policy transcript_segments_owner_select
on public.transcript_segments
for select
to authenticated
using (
    exists (
        select 1
        from public.transcripts
        join public.recordings on recordings.id = transcripts.recording_id
        where transcripts.id = transcript_segments.transcript_id
          and recordings.owner_id = (select auth.uid())
    )
);

create policy transcript_segments_owner_update_clean
on public.transcript_segments
for update
to authenticated
using (
    exists (
        select 1
        from public.transcripts
        join public.recordings on recordings.id = transcripts.recording_id
        where transcripts.id = transcript_segments.transcript_id
          and recordings.owner_id = (select auth.uid())
    )
)
with check (
    exists (
        select 1
        from public.transcripts
        join public.recordings on recordings.id = transcripts.recording_id
        where transcripts.id = transcript_segments.transcript_id
          and recordings.owner_id = (select auth.uid())
    )
);

insert into storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
values (
    'recordings',
    'recordings',
    false,
    25165824,
    array['audio/*']
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

create policy recordings_objects_owner_insert
on storage.objects
for insert
to authenticated
with check (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and storage.filename(name) ~ '^source\.part[0-9]{3,}\.[a-z0-9]{1,10}$'
    and exists (
        select 1
        from public.recordings
        where recordings.owner_id = (select auth.uid())
          and recordings.object_path =
              (storage.foldername(name))[1] || '/' || (storage.foldername(name))[2]
          and recordings.storage_status in ('pending_upload', 'ready', 'failed')
          and right(storage.filename(name), char_length(recordings.file_extension) + 1)
              = '.' || recordings.file_extension
    )
);

create policy recordings_objects_owner_select
on storage.objects
for select
to authenticated
using (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and exists (
        select 1
        from public.recordings
        where recordings.owner_id = (select auth.uid())
          and recordings.object_path =
              (storage.foldername(name))[1] || '/' || (storage.foldername(name))[2]
          and recordings.storage_status in ('pending_upload', 'ready', 'failed')
    )
);

create policy recordings_objects_owner_update
on storage.objects
for update
to authenticated
using (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
)
with check (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and storage.filename(name) ~ '^source\.part[0-9]{3,}\.[a-z0-9]{1,10}$'
    and exists (
        select 1
        from public.recordings
        where recordings.owner_id = (select auth.uid())
          and recordings.object_path =
              (storage.foldername(name))[1] || '/' || (storage.foldername(name))[2]
          and recordings.storage_status in ('pending_upload', 'ready', 'failed')
          and right(storage.filename(name), char_length(recordings.file_extension) + 1)
              = '.' || recordings.file_extension
    )
);

create policy recordings_objects_owner_delete
on storage.objects
for delete
to authenticated
using (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and exists (
        select 1
        from public.recordings
        where recordings.owner_id = (select auth.uid())
          and recordings.object_path =
              (storage.foldername(name))[1] || '/' || (storage.foldername(name))[2]
    )
);

create or replace function public.claim_expired_recordings(p_limit integer default 25)
returns setof public.recordings
language plpgsql
security invoker
set search_path = ''
as $$
begin
    if p_limit < 1 or p_limit > 100 then
        raise exception 'p_limit must be between 1 and 100'
            using errcode = '22023';
    end if;

    return query
    with candidates as (
        select recordings.id
        from public.recordings
        where recordings.retention_until <= statement_timestamp()
          and recordings.deleted_at is null
          and recordings.storage_status not in ('deleting', 'deleted')
        order by recordings.retention_until, recordings.id
        for update skip locked
        limit p_limit
    )
    update public.recordings
    set storage_status = 'deleting'
    from candidates
    where recordings.id = candidates.id
    returning recordings.*;
end;
$$;

revoke all on function public.claim_expired_recordings(integer) from public, anon, authenticated;
grant execute on function public.claim_expired_recordings(integer) to service_role;

comment on table public.recordings is
    'Private PhoneScribe audio metadata. object_path is a prefix; parts are source.partNNN.<ext>. Audio expiry never deletes this row or its transcripts.';
comment on column public.recordings.owner_id is
    'Supabase Auth owner. NULL denotes passcode-only shared mode and is service-role-only.';
comment on column public.recordings.storage_status is
    'Storage lifecycle only. deleted means audio objects are gone; the recording metadata and transcript rows remain.';
comment on column public.recordings.retention_until is
    'After this time, cleanup deletes only Storage audio objects, then sets storage_status to deleted and deleted_at; this row and transcript rows remain.';
comment on column public.transcripts.text_raw is
    'Immutable model output. User review edits text_clean only.';
comment on column public.transcript_segments.text_raw is
    'Immutable segment output. User review edits text_clean only.';
comment on function public.claim_expired_recordings(integer) is
    'Service-role worker claim. Delete audio through the Storage API, then set storage_status and deleted_at through the completion RPC; never delete this row or storage.objects directly.';

commit;
