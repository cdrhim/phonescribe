begin;

drop policy if exists recordings_objects_owner_insert on storage.objects;

create policy recordings_objects_owner_insert
on storage.objects
for insert
to authenticated
with check (
    bucket_id = 'recordings'
    and cardinality(storage.foldername(name)) = 2
    and (storage.foldername(name))[1] = (select auth.uid())::text
    and storage.filename(name) ~ '^source\.part[0-9]{3,4}\.[a-z0-9]{1,10}$'
    and exists (
        select 1
        from public.recordings
        where recordings.owner_id = (select auth.uid())
          and recordings.object_path =
              (storage.foldername(name))[1] || '/' || (storage.foldername(name))[2]
          and recordings.storage_status = 'pending_upload'
          and storage.filename(name) =
              'source.part'
              || lpad(
                  (
                      substring(
                          storage.filename(name)
                          from '^source\.part([0-9]{3,4})\.'
                      )::integer
                  )::text,
                  3,
                  '0'
              )
              || '.' || recordings.file_extension
          and substring(
                  storage.filename(name)
                  from '^source\.part([0-9]{3,4})\.'
              )::integer < recordings.part_count
          and case
              when coalesce(storage.objects.metadata ->> 'size', '') ~ '^[0-9]+$'
              then (storage.objects.metadata ->> 'size')::bigint
              else -1
          end = least(
              recordings.part_size_bytes,
              recordings.size_bytes
              - substring(
                    storage.filename(name)
                    from '^source\.part([0-9]{3,4})\.'
                )::integer * recordings.part_size_bytes
          )
    )
);

comment on policy recordings_objects_owner_insert on storage.objects is
    'Authenticated owners may upload only canonical declared parts at their exact expected size while pending. Lifecycle changes and deletion are server-only.';

commit;
