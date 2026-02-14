"""Tests for German text normalization."""

import pytest
from leander_tts.data.text_normalizer import (
    _number_to_words,
    _ordinal_to_words,
    normalize_german_text,
)


class TestNumberToWords:
    def test_zero(self):
        assert _number_to_words(0) == "null"

    def test_single_digits(self):
        assert _number_to_words(1) == "eins"
        assert _number_to_words(5) == "fünf"
        assert _number_to_words(9) == "neun"

    def test_teens(self):
        assert _number_to_words(11) == "elf"
        assert _number_to_words(12) == "zwölf"
        assert _number_to_words(16) == "sechzehn"

    def test_tens(self):
        assert _number_to_words(20) == "zwanzig"
        assert _number_to_words(30) == "dreißig"

    def test_compound_numbers(self):
        # German: ones before tens with "und"
        result = _number_to_words(42)
        assert "zwei" in result
        assert "vierzig" in result
        assert "und" in result

    def test_hundreds(self):
        result = _number_to_words(100)
        assert "hundert" in result

    def test_thousands(self):
        result = _number_to_words(1000)
        assert "tausend" in result

    def test_negative(self):
        result = _number_to_words(-5)
        assert result.startswith("minus")

    def test_million(self):
        result = _number_to_words(1000000)
        assert "Million" in result


class TestOrdinals:
    def test_first(self):
        assert _ordinal_to_words(1) == "erste"

    def test_third(self):
        assert _ordinal_to_words(3) == "dritte"

    def test_seventh(self):
        assert _ordinal_to_words(7) == "siebte"


class TestFullNormalization:
    def test_abbreviation_expansion(self):
        result = normalize_german_text("z.B. das ist gut")
        assert "zum Beispiel" in result

    def test_time_expansion(self):
        result = normalize_german_text("Um 14:30 Uhr")
        assert "Uhr" in result
        assert "14" not in result  # Should be expanded

    def test_currency(self):
        result = normalize_german_text("Das kostet 5 €")
        assert "Euro" in result
        assert "5" not in result

    def test_preserves_normal_text(self):
        text = "Hallo, wie geht es dir?"
        result = normalize_german_text(text)
        assert "Hallo" in result
        assert "wie" in result

    def test_number_expansion(self):
        result = normalize_german_text("Ich habe 3 Äpfel")
        assert "drei" in result
        assert "3" not in result
