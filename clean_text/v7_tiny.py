"""Clean text for ruSpamNS_v7_tiny."""

import re


def clean_text(text: str):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[a-zA-Z0-9_]{5,}", "", text)
    text = text.lower().strip()
    return text
