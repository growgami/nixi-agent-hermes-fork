"""Data models for the analyst ingestion pipeline.

Defines dataclasses for articles, tweets, market data,
and ingestion result summaries stored in analyst_state.db.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Article:
    """A news article from an external API source.

    Attributes:
        source: API source identifier (e.g. "newsdata", "guardian").
        title: Article headline.
        text: Full article body text.
        url: Canonical URL — must be unique across articles.
        author: Author byline. Nullable.
        published_at: ISO datetime of original publication. Nullable.
        ingested_at: ISO datetime when ingested into analyst_state.db.
            Set by the ingestion pipeline if not provided.
        sentiment_label: Pre-computed sentiment (e.g. "positive",
            "negative", "neutral"). Nullable.
        metadata_json: Arbitrary JSON blob for source-specific fields.
            Nullable.
    """

    source: str
    title: str
    text: str
    url: str
    author: str | None = None
    published_at: str | None = None
    ingested_at: str | None = None
    sentiment_label: str | None = None
    metadata_json: str | None = None


@dataclass
class Tweet:
    """A tweet from TwitterAPI.io.

    Attributes:
        tweet_id: Twitter-unique tweet ID — must be unique.
        username: Twitter handle (without @).
        text: Tweet body text.
        url: Permalink to the tweet. Nullable.
        like_count: Number of likes. Defaults to 0.
        retweet_count: Number of retweets. Defaults to 0.
        reply_count: Number of replies. Defaults to 0.
        view_count: Number of impressions. Defaults to 0.
        created_at: ISO datetime of original tweet. Nullable.
        ingested_at: ISO datetime when ingested. Set by pipeline.
        lang: BCP-47 language code. Nullable.
        metadata_json: Arbitrary JSON blob. Nullable.
    """

    tweet_id: str
    username: str
    text: str
    url: str | None = None
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    view_count: int = 0
    created_at: str | None = None
    ingested_at: str | None = None
    lang: str | None = None
    metadata_json: str | None = None


@dataclass
class MarketData:
    """A market data snapshot from a financial API.

    Attributes:
        symbol: Ticker symbol (e.g. "AAPL", "BTC").
        source: API source (e.g. "yfinance", "coingecko").
        price: Current or closing price.
        change_pct: Percentage change. Nullable.
        volume: Trading volume. Nullable.
        market_cap: Market capitalization. Nullable.
        timestamp: ISO datetime of the data point. Nullable.
        ingested_at: ISO datetime when ingested. Set by pipeline.
        metadata_json: Arbitrary JSON blob. Nullable.
    """

    symbol: str
    source: str
    price: float
    change_pct: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    timestamp: str | None = None
    ingested_at: str | None = None
    metadata_json: str | None = None


@dataclass
class AnalystIngestionResult:
    """Summary of an ingestion run for a single source.

    Attributes:
        source: API source that was queried.
        fetched: Number of items fetched from the API.
        inserted: Number of new rows inserted into analyst_state.db.
        duplicates: Number of items skipped as duplicates.
        errors: Number of items that failed to insert.
    """

    source: str
    fetched: int
    inserted: int
    duplicates: int
    errors: int