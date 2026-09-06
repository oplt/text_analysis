-- Local Postgres bootstrap for text_analysis development.
-- Run as the postgres superuser, e.g.:
--   sudo -u postgres psql -f backend/scripts/setup-local-db.sql

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'text_analysis') THEN
    CREATE ROLE text_analysis LOGIN PASSWORD 'text_analysis';
  ELSE
    ALTER ROLE text_analysis WITH LOGIN PASSWORD 'text_analysis';
  END IF;
END
$$;

SELECT format(
  'CREATE DATABASE %I OWNER %I',
  'text_analysis',
  'text_analysis'
)
WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = 'text_analysis'
)\gexec

GRANT ALL PRIVILEGES ON DATABASE text_analysis TO text_analysis;
