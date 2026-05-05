-- DDL for analyst_state.db
-- Separate database for the data analyst feature.
-- NOT co-located with nixi_state.db extraction pipeline data.

-- News articles from external APIs (NewsData.io, Guardian, etc.)
CREATE TABLE IF NOT EXISTS analyst_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT UNIQUE,
    author TEXT,
    published_at DATETIME,
    ingested_at DATETIME,
    sentiment_label TEXT,
    metadata_json TEXT
);

-- Tweets from TwitterAPI.io
CREATE TABLE IF NOT EXISTS analyst_tweets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT,
    like_count INTEGER DEFAULT 0,
    retweet_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    created_at DATETIME,
    ingested_at DATETIME,
    lang TEXT,
    metadata_json TEXT
);

-- Market data snapshots from financial APIs (yfinance, CoinGecko, etc.)
CREATE TABLE IF NOT EXISTS analyst_market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    price REAL NOT NULL,
    change_pct REAL,
    volume REAL,
    market_cap REAL,
    timestamp DATETIME,
    ingested_at DATETIME,
    metadata_json TEXT
);

-- Ingestion run log — one row per source per run
CREATE TABLE IF NOT EXISTS analyst_ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    run_at DATETIME NOT NULL,
    fetched INTEGER NOT NULL DEFAULT 0,
    inserted INTEGER NOT NULL DEFAULT 0,
    duplicates INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'success'
);

-- Standard indexes for common query patterns
-- IMPORTANT: Create indexes BEFORE FTS5 content-sync virtual tables,
-- because FTS5 content= tables intercept DDL on the content table.
CREATE INDEX IF NOT EXISTS idx_analyst_articles_source ON analyst_articles(source);
CREATE INDEX IF NOT EXISTS idx_analyst_articles_published_at ON analyst_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_analyst_tweets_username ON analyst_tweets(username);
CREATE INDEX IF NOT EXISTS idx_analyst_tweets_created_at ON analyst_tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_analyst_market_data_symbol ON analyst_market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_analyst_market_data_timestamp ON analyst_market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_analyst_ingestion_log_ingested_at ON analyst_ingestion_log(run_at);

-- FTS5 full-text search indexes (must come AFTER regular indexes)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_articles USING fts5(text, title, content=analyst_articles, content_rowid=id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_tweets USING fts5(text, content=analyst_tweets, content_rowid=id);