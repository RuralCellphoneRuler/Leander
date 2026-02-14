"""Tests for emotion tag processing."""

import pytest
from leander_tts.data.emotion_tags import (
    INLINE_TAGS,
    SPAN_TAGS,
    build_emotion_tag_tokens,
    extract_tags,
    strip_tags,
)


class TestEmotionTagTokens:
    def test_build_tokens(self):
        tokens = build_emotion_tag_tokens(start_id=1000)
        assert len(tokens) > 0
        # Check IDs are sequential
        ids = [t.token_id for t in tokens]
        assert ids == list(range(1000, 1000 + len(tokens)))

    def test_inline_tags_counted(self):
        tokens = build_emotion_tag_tokens(start_id=0)
        inline_count = sum(1 for t in tokens if not t.is_span_open and not t.is_span_close)
        assert inline_count == len(INLINE_TAGS)

    def test_span_tags_paired(self):
        tokens = build_emotion_tag_tokens(start_id=0)
        open_count = sum(1 for t in tokens if t.is_span_open)
        close_count = sum(1 for t in tokens if t.is_span_close)
        assert open_count == len(SPAN_TAGS)
        assert close_count == len(SPAN_TAGS)


class TestExtractTags:
    def test_extract_inline(self):
        text = "Hallo <lacht> wie geht es?"
        tags = extract_tags(text)
        assert len(tags) == 1
        assert tags[0][1] == "<lacht>"

    def test_extract_multiple(self):
        text = "Hallo <lacht> und <seufzt> tja"
        tags = extract_tags(text)
        assert len(tags) == 2

    def test_extract_span(self):
        text = "<flüstert>Leise sprechen</flüstert>"
        tags = extract_tags(text)
        assert len(tags) == 2

    def test_no_tags(self):
        text = "Ganz normaler Text"
        tags = extract_tags(text)
        assert len(tags) == 0


class TestStripTags:
    def test_strip_inline(self):
        text = "Hallo <lacht> wie geht es?"
        result = strip_tags(text)
        assert "<lacht>" not in result
        assert "Hallo" in result

    def test_strip_span(self):
        text = "<flüstert>Leise</flüstert>"
        result = strip_tags(text)
        assert "flüstert" not in result
        assert "Leise" in result

    def test_strip_preserves_text(self):
        text = "Einfach nur Text"
        assert strip_tags(text) == text
