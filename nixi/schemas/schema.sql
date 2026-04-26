-- DDL for nixi_state.db
-- Scraped messages table and extraction log table for the Slack extraction pipeline.

CREATE TABLE IF NOT EXISTS scraped_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slack_ts TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL,
    user_id TEXT,
    user_name TEXT NOT NULL,
    text TEXT NOT NULL,
    thread_ts TEXT,
    parent_ts TEXT,
    is_bot INTEGER DEFAULT 0,
    source_file TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    UNIQUE(channel_id, slack_ts)
);

CREATE TABLE IF NOT EXISTS nixi_extraction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    slack_ts TEXT NOT NULL,
    extraction_batch TEXT NOT NULL,
    extracted_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sm_channel_ts ON scraped_messages(channel_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sm_bot ON scraped_messages(is_bot);
CREATE INDEX IF NOT EXISTS idx_extraction_batch ON nixi_extraction_log(extraction_batch);
CREATE INDEX IF NOT EXISTS idx_el_channel_ts ON nixi_extraction_log(channel_id, slack_ts);