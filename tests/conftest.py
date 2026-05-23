"""
Shared pytest fixtures for IntelliPipe test suite.
"""
import os
import pytest

# Set test environment variables before any imports
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "intellipipe")
os.environ.setdefault("DB_USER", "intellipipe")
os.environ.setdefault("DB_PASSWORD", "intellipipe")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("API_SECRET_KEY", "test-secret-key-32-chars-minimum!!")
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("JIRA_API_TOKEN", "test-token")
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test-token")
