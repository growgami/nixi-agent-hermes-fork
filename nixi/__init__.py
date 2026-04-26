"""Nixi — Multi-tenant AI agent package for Hermes."""

__version__ = "0.1.0"

from nixi.adapter import LogFileAdapter
from nixi.config import NixiConfig
from nixi.db import build_user_map, count_by_channel, ensure_schema, get_connection, get_unprocessed, get_unprocessed_channels, insert_messages, mark_extracted
from nixi.models import ExtractionBatch, IngestionResult, Link, ParsedLine, ScrapedMessage, UserMap
from nixi.parser import LogParser

__all__ = [
    "ExtractionBatch",
    "IngestionResult",
    "Link",
    "LogFileAdapter",
    "LogParser",
    "NixiConfig",
    "ParsedLine",
    "ScrapedMessage",
    "UserMap",
    "build_user_map",
    "count_by_channel",
    "ensure_schema",
    "get_connection",
    "get_unprocessed",
    "get_unprocessed_channels",
    "insert_messages",
    "mark_extracted",
]