begin;

alter table public.recordings
    alter column part_size_bytes set default 6291456;

comment on column public.recordings.part_size_bytes is
    'PhoneScribe uses 6 MiB signed-upload objects so each failed mobile request can be retried independently.';

commit;
