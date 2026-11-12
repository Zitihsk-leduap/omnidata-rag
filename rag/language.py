import re

BS_MONTHS = {
    "बैशाख": "01",
    "जेठ": "02",
    "असार": "03",
    "श्रावण": "04",
    "भदौ": "05",
    "आश्विन": "06",
    "असोज": "06",
    "कार्तिक": "07",
    "मंसिर": "08",
    "पौष": "09",
    "माघ": "10",
    "फाल्गुन": "11",
    "चैत्र": "12",
}

_BS_DATE_RE = re.compile(r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते")


def normalize_bs_date(text: str) -> str:
    """Only normalize dates that appear outside of Nepali quoted text."""

    def repl(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        month_num = BS_MONTHS.get(month, "??")
        return f"BS {year}-{month_num}-{int(day):02d}"

    # Split on the em-dash that separates Nepali quote from English translation
    if "—" in text:
        nepali_part, english_part = text.split("—", 1)
        english_part = _BS_DATE_RE.sub(repl, english_part)
        return nepali_part + "—" + english_part
    return _BS_DATE_RE.sub(repl, text)


def detect_language(text: str) -> str:
    """Detect if query is Nepali or English."""
    nepali_chars = len(re.findall(r"[ऀ-ॿ]", text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    nepali_ratio = nepali_chars / total_chars
    if nepali_ratio > 0.3:
        return "nepali"
    return "english"
