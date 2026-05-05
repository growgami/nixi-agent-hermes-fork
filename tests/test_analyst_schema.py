"""Tests for nixi.analyst.schema and nixi.analyst.models — analyst_state.db.

Covers: schema creation, data model construction, separate DB verification.
"""

from pathlib import Path

import pytest

from nixi.analyst.models import Article, AnalystIngestionResult, MarketData, Tweet
from nixi.analyst.schema import ensure_analyst_schema


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def analyst_db_path(tmp_path: Path) -> Path:
    """Temp path for analyst_state.db."""
    return tmp_path / "analyst_state.db"


@pytest.fixture
def nixi_db_path(tmp_path: Path) -> Path:
    """Temp path for nixi_state.db — used to verify separation."""
    return tmp_path / "nixi_state.db"


# ── Data model tests ─────────────────────────────────────────────────────────


class TestArticle:
    """Test Article dataclass construction."""

    def test_article_creation_minimal(self):
        article = Article(
            source="newsdata",
            title="Test Article",
            text="Body text",
            url="https://example.com/article",
        )
        assert article.source == "newsdata"
        assert article.title == "Test Article"
        assert article.text == "Body text"
        assert article.url == "https://example.com/article"
        assert article.author is None
        assert article.published_at is None
        assert article.sentiment_label is None
        assert article.metadata_json is None

    def test_article_creation_full(self):
        article = Article(
            source="guardian",
            title="Full Article",
            text="Full text",
            url="https://example.com/full",
            author="Jane Doe",
            published_at="2026-05-05T10:00:00Z",
            sentiment_label="positive",
            metadata_json='{"category":"tech"}',
        )
        assert article.author == "Jane Doe"
        assert article.published_at == "2026-05-05T10:00:00Z"
        assert article.sentiment_label == "positive"
        assert article.metadata_json == '{"category":"tech"}'


class TestTweet:
    """Test Tweet dataclass construction."""

    def test_tweet_creation_minimal(self):
        tweet = Tweet(
            tweet_id="1234567890",
            username="testuser",
            text="Hello world",
        )
        assert tweet.tweet_id == "1234567890"
        assert tweet.username == "testuser"
        assert tweet.text == "Hello world"
        assert tweet.url is None
        assert tweet.like_count == 0
        assert tweet.retweet_count == 0
        assert tweet.reply_count == 0
        assert tweet.view_count == 0
        assert tweet.lang is None
        assert tweet.metadata_json is None

    def test_tweet_creation_full(self):
        tweet = Tweet(
            tweet_id="999",
            username="elcapitan",
            text="Market update",
            url="https://x.com/elcapitan/status/999",
            like_count=42,
            retweet_count=10,
            reply_count=5,
            view_count=1000,
            created_at="2026-05-04T12:00:00Z",
            lang="en",
            metadata_json='{"extra":"data"}',
        )
        assert tweet.url == "https://x.com/elcapitan/status/999"
        assert tweet.like_count == 42
        assert tweet.lang == "en"


class TestMarketData:
    """Test MarketData dataclass construction."""

    def test_market_data_creation_minimal(self):
        md = MarketData(symbol="AAPL", source="yfinance", price=150.0)
        assert md.symbol == "AAPL"
        assert md.source == "yfinance"
        assert md.price == 150.0
        assert md.change_pct is None
        assert md.volume is None
        assert md.market_cap is None
        assert md.metadata_json is None

    def test_market_data_creation_full(self):
        md = MarketData(
            symbol="BTC",
            source="coingecko",
            price=95000.5,
            change_pct=2.3,
            volume=35000.0,
            market_cap=1.8e12,
            timestamp="2026-05-05T00:00:00Z",
            metadata_json='{"currency":"USD"}',
        )
        assert md.change_pct == 2.3
        assert md.market_cap == 1.8e12


class TestAnalystIngestionResult:
    """Test AnalystIngestionResult dataclass construction."""

    def test_ingestion_result_creation(self):
        result = AnalystIngestionResult(
            source="twitter",
            fetched=100,
            inserted=80,
            duplicates=15,
            errors=5,
        )
        assert result.source == "twitter"
        assert result.fetched == 100
        assert result.inserted == 80
        assert result.duplicates == 15
        assert result.errors == 5


# ── Schema creation tests ─────────────────────────────────────────────────────


class TestEnsureAnalystSchema:
    """Test ensure_analyst_schema creates all analyst tables and indexes."""

    def test_creates_analyst_tables(self, analyst_db_path: Path):
        """All four analyst tables exist after schema creation."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "analyst_articles" in tables
        assert "analyst_tweets" in tables
        assert "analyst_market_data" in tables
        assert "analyst_ingestion_log" in tables

    def test_creates_fts_indexes(self, analyst_db_path: Path):
        """FTS5 virtual tables are created for articles and tweets."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fts_%' ORDER BY name"
        )
        fts_tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "fts_articles" in fts_tables
        assert "fts_tweets" in fts_tables

    def test_creates_standard_indexes(self, analyst_db_path: Path):
        """Standard indexes on key columns are created."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_analyst_%' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_analyst_articles_source" in indexes
        assert "idx_analyst_articles_published_at" in indexes
        assert "idx_analyst_tweets_username" in indexes
        assert "idx_analyst_tweets_created_at" in indexes
        assert "idx_analyst_market_data_symbol" in indexes
        assert "idx_analyst_market_data_timestamp" in indexes
        assert "idx_analyst_ingestion_log_ingested_at" in indexes

    def test_idempotent_schema_creation(self, analyst_db_path: Path):
        """Running ensure_analyst_schema twice does not raise errors."""
        ensure_analyst_schema(analyst_db_path)
        # Second call should succeed without error
        ensure_analyst_schema(analyst_db_path)

        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM analyst_articles"
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0  # Empty, not duplicated

    def test_separate_db_from_nixi_state(self, analyst_db_path: Path, nixi_db_path: Path):
        """ensure_analyst_schema creates tables in analyst_state.db, NOT nixi_state.db."""
        from nixi.db import ensure_schema

        # Create both databases
        ensure_schema(nixi_db_path)
        ensure_analyst_schema(analyst_db_path)

        import sqlite3

        # Verify analyst tables are in analyst_state.db
        conn_analyst = sqlite3.connect(str(analyst_db_path))
        cursor = conn_analyst.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'analyst_%'"
        )
        analyst_tables = {row[0] for row in cursor.fetchall()}
        conn_analyst.close()

        # Verify analyst tables are NOT in nixi_state.db
        conn_nixi = sqlite3.connect(str(nixi_db_path))
        cursor_nixi = conn_nixi.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'analyst_%'"
        )
        nixi_analyst_tables = {row[0] for row in cursor_nixi.fetchall()}
        conn_nixi.close()

        assert len(analyst_tables) > 0, "analyst tables should exist in analyst_state.db"
        assert len(nixi_analyst_tables) == 0, "analyst tables should NOT exist in nixi_state.db"

    def test_default_db_path_creates_file(self, tmp_path: Path, monkeypatch):
        """ensure_analyst_schema with default path creates analyst_state.db in hermes_home."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        ensure_analyst_schema()

        db_file = tmp_path / "analyst_state.db"
        assert db_file.exists()

    def test_article_table_columns(self, analyst_db_path: Path):
        """analyst_articles has all expected columns."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute("PRAGMA table_info(analyst_articles)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "source" in columns
        assert "title" in columns
        assert "text" in columns
        assert "url" in columns
        assert "author" in columns
        assert "published_at" in columns
        assert "ingested_at" in columns
        assert "sentiment_label" in columns
        assert "metadata_json" in columns
        # Ensure topics_json is NOT present per postconditions
        assert "topics_json" not in columns

    def test_tweet_table_columns(self, analyst_db_path: Path):
        """analyst_tweets has all expected columns."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute("PRAGMA table_info(analyst_tweets)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "tweet_id" in columns
        assert "username" in columns
        assert "text" in columns
        assert "url" in columns
        assert "like_count" in columns
        assert "retweet_count" in columns
        assert "reply_count" in columns
        assert "view_count" in columns
        assert "created_at" in columns
        assert "ingested_at" in columns
        assert "lang" in columns
        assert "metadata_json" in columns
        assert "topics_json" not in columns

    def test_market_data_table_columns(self, analyst_db_path: Path):
        """analyst_market_data has all expected columns."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute("PRAGMA table_info(analyst_market_data)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "symbol" in columns
        assert "source" in columns
        assert "price" in columns
        assert "change_pct" in columns
        assert "volume" in columns
        assert "market_cap" in columns
        assert "timestamp" in columns
        assert "ingested_at" in columns
        assert "metadata_json" in columns
        assert "topics_json" not in columns

    def test_ingestion_log_table_columns(self, analyst_db_path: Path):
        """analyst_ingestion_log has all expected columns."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        cursor = conn.execute("PRAGMA table_info(analyst_ingestion_log)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "source" in columns
        assert "run_at" in columns
        assert "fetched" in columns
        assert "inserted" in columns
        assert "duplicates" in columns
        assert "errors" in columns
        assert "status" in columns

    def test_url_unique_constraint(self, analyst_db_path: Path):
        """analyst_articles.url is UNIQUE."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        conn.execute(
            "INSERT INTO analyst_articles (source, title, text, url) VALUES ('src', 't', 'txt', 'https://example.com/1')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO analyst_articles (source, title, text, url) VALUES ('src', 't2', 'txt2', 'https://example.com/1')"
            )
        conn.close()

    def test_tweet_id_unique_constraint(self, analyst_db_path: Path):
        """analyst_tweets.tweet_id is UNIQUE."""
        ensure_analyst_schema(analyst_db_path)
        import sqlite3

        conn = sqlite3.connect(str(analyst_db_path))
        conn.execute(
            "INSERT INTO analyst_tweets (tweet_id, username, text) VALUES ('123', 'user', 'hello')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO analyst_tweets (tweet_id, username, text) VALUES ('123', 'user2', 'hello2')"
            )
        conn.close()