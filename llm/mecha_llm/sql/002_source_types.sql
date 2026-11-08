-- Allow the profile + repo source types (Phase 1 of mecha-haruka-improvements).
-- 001_chunks.sql is CREATE TABLE IF NOT EXISTS, so existing tables need this.

ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_source_type_check;

ALTER TABLE chunks ADD CONSTRAINT chunks_source_type_check
CHECK (source_type IN ('journal', 'project', 'experience', 'page', 'profile', 'repo'));
