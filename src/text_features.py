import easyocr

# We only need English to detect text presence.
# Script detection is done via Unicode ranges.
reader = easyocr.Reader(['en'], gpu=False)

SCRIPT_MAP = {
    "latin": 1,
    "cyrillic": 2,
    "arabic": 3,
    "hebrew": 4,
    "greek": 5,
    "hangul": 6,
    "japanese": 7,
    "chinese": 8,
    "devanagari": 9,
    "tamil": 10,
    "thai": 11
}

def detect_script(image_path):
    """
    Returns:
    has_text (0/1)
    script_id (0–11)
    """

    try:
        results = reader.readtext(str(image_path), detail=0)
    except Exception:
        return 0, 0

    if not results:
        return 0, 0

    text = "".join(results)

    for ch in text:
        code = ord(ch)

        if 0x0400 <= code <= 0x04FF:
            return 1, SCRIPT_MAP["cyrillic"]
        if 0x0600 <= code <= 0x06FF:
            return 1, SCRIPT_MAP["arabic"]
        if 0x0590 <= code <= 0x05FF:
            return 1, SCRIPT_MAP["hebrew"]
        if 0x0370 <= code <= 0x03FF:
            return 1, SCRIPT_MAP["greek"]
        if 0xAC00 <= code <= 0xD7AF:
            return 1, SCRIPT_MAP["hangul"]
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            return 1, SCRIPT_MAP["japanese"]
        if 0x4E00 <= code <= 0x9FFF:
            return 1, SCRIPT_MAP["chinese"]
        if 0x0900 <= code <= 0x097F:
            return 1, SCRIPT_MAP["devanagari"]
        if 0x0B80 <= code <= 0x0BFF:
            return 1, SCRIPT_MAP["tamil"]
        if 0x0E00 <= code <= 0x0E7F:
            return 1, SCRIPT_MAP["thai"]

    # If text exists but no special script detected
    return 1, SCRIPT_MAP["latin"]