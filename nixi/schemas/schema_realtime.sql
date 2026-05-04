-- DDL for realtime_messages table in nixi_state.db.
-- Mirrors the schema defined in infra/ingester/schema.sql.
-- The Go ingester is the sole writer; the Python extraction pipeline reads.
-- team_id is nullable — some Socket Mode events may not include it.

CREATE TABLE IF NOT EXISTS realtime_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slack_ts TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id TEXT,
    text TEXT NOT NULL,
    thread_ts TEXT,
    parent_ts TEXT,
    is_bot INTEGER DEFAULT 0,
    channel_type TEXT,
    event_id TEXT NOT NULL,
    client_msg_id TEXT,
    team_id TEXT,
    timestamp DATETIME NOT NULL,
    UNIQUE(channel_id, slack_ts)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rm_event_id ON realtime_messages(event_id);
CREATE INDEX IF NOT EXISTS idx_rm_channel_ts ON realtime_messages(channel_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_rm_team ON realtime_messages(team_id);