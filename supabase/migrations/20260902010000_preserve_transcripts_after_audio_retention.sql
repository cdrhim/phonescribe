begin;

alter table public.recordings
    add column if not exists retention_claimed_at timestamptz,
    add column if not exists retention_attempt_count integer not null default 0,
    add column if not exists retention_error text;

alter table public.recordings
    add constraint recordings_retention_attempt_count_check
        check (retention_attempt_count >= 0),
    add constraint recordings_retention_error_check
        check (retention_error is null or char_length(retention_error) <= 1000);

create index recordings_stale_retention_claim_idx
    on public.recordings (retention_claimed_at, id)
    where deleted_at is null and storage_status = 'deleting';

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
          and recordings.storage_status <> 'deleted'
          and (
              recordings.storage_status <> 'deleting'
              or recordings.retention_claimed_at is null
              or recordings.retention_claimed_at
                  <= statement_timestamp() - interval '15 minutes'
          )
        order by recordings.retention_until, recordings.id
        for update skip locked
        limit p_limit
    )
    update public.recordings as target
    set storage_status = 'deleting',
        retention_claimed_at = statement_timestamp(),
        retention_attempt_count = target.retention_attempt_count + 1,
        retention_error = null
    from candidates
    where target.id = candidates.id
    returning target.*;
end;
$$;

create or replace function public.complete_recording_retention(p_recording_id uuid)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
declare
    affected_rows integer;
begin
    update public.recordings
    set storage_status = 'deleted',
        deleted_at = statement_timestamp(),
        retention_claimed_at = null,
        retention_error = null
    where recordings.id = p_recording_id
      and recordings.storage_status = 'deleting'
      and recordings.deleted_at is null;

    get diagnostics affected_rows = row_count;
    return affected_rows = 1;
end;
$$;

create or replace function public.fail_recording_retention(
    p_recording_id uuid,
    p_error text
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
declare
    affected_rows integer;
begin
    update public.recordings
    set storage_status = 'failed',
        retention_claimed_at = null,
        retention_error = left(coalesce(nullif(p_error, ''), 'storage_delete_failed'), 1000)
    where recordings.id = p_recording_id
      and recordings.storage_status = 'deleting'
      and recordings.deleted_at is null;

    get diagnostics affected_rows = row_count;
    return affected_rows = 1;
end;
$$;

revoke all on function public.claim_expired_recordings(integer)
    from public, anon, authenticated;
revoke all on function public.complete_recording_retention(uuid)
    from public, anon, authenticated;
revoke all on function public.fail_recording_retention(uuid, text)
    from public, anon, authenticated;

grant execute on function public.claim_expired_recordings(integer) to service_role;
grant execute on function public.complete_recording_retention(uuid) to service_role;
grant execute on function public.fail_recording_retention(uuid, text) to service_role;

comment on column public.recordings.deleted_at is
    'Time the audio objects were confirmed deleted. Recording metadata and transcripts remain.';
comment on column public.recordings.storage_status is
    'Storage lifecycle only. deleted means audio objects are gone; the recording metadata and transcript rows remain.';
comment on column public.recordings.retention_until is
    'After this time, cleanup deletes only Storage audio objects, then sets storage_status to deleted and deleted_at; this row and transcript rows remain.';
comment on column public.recordings.retention_claimed_at is
    'Lease timestamp for Storage cleanup. Claims older than 15 minutes can be retried.';
comment on function public.claim_expired_recordings(integer) is
    'Claims expired audio cleanup work. A stale deleting claim is reclaimable after 15 minutes.';
comment on function public.complete_recording_retention(uuid) is
    'Call only after every audio part was deleted through the Storage API. Preserves the recording row and transcripts.';
comment on function public.fail_recording_retention(uuid, text) is
    'Releases a failed cleanup claim for retry without removing transcript data.';

commit;
