from rag.language import detect_language, normalize_bs_date


def test_detect_language_nepali():
    assert detect_language("कम्पनी ऐन २०६३ कहिले लागू भयो?") == "nepali"


def test_detect_language_english():
    assert detect_language("What is the Nepal Company Act?") == "english"


def test_detect_language_mixed_defaults_by_ratio():
    # Mostly English with one Nepali word should still read as English.
    assert detect_language("Please translate सचिव into English") == "english"


def test_detect_language_empty_string_defaults_to_english():
    assert detect_language("") == "english"


def test_normalize_bs_date_basic():
    text = "The act came into force on 2063 साल असोज 1 गते."
    assert normalize_bs_date(text) == "The act came into force on BS 2063-06-01."


def test_normalize_bs_date_unknown_month_is_marked():
    text = "2063 साल bogus 1 गते"
    assert normalize_bs_date(text) == "BS 2063-??-01"


def test_normalize_bs_date_leaves_text_before_em_dash_untouched():
    text = "२०६३ साल असोज १ गते — 2063 साल असोज 1 गते"
    result = normalize_bs_date(text)
    assert result.startswith("२०६३ साल असोज १ गते —")
    assert "BS 2063-06-01" in result


def test_normalize_bs_date_no_match_returns_text_unchanged():
    text = "There is no date here."
    assert normalize_bs_date(text) == text
