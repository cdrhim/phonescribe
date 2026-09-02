begin;

drop policy if exists recordings_objects_owner_insert on storage.objects;
drop policy if exists recordings_objects_owner_update on storage.objects;
drop policy if exists recordings_objects_owner_delete on storage.objects;

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
          and recordings.storage_status = 'pending_upload'
          and right(storage.filename(name), char_length(recordings.file_extension) + 1)
              = '.' || recordings.file_extension
    )
);

comment on policy recordings_objects_owner_insert on storage.objects is
    'Authenticated owners may add immutable parts only while a recording is pending upload. Lifecycle changes and deletion are server-only.';

commit;
