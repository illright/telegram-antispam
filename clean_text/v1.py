"""Clean text for ruSpamNS_v1."""

import re


def clean_text(text: str):
    text = re.sub(r"\w+", lambda m: recover_cyrillic_in_word(m.group()), text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^А-Яа-я0-9 ]+", " ", text)
    text = text.lower().strip()
    return text


def recover_cyrillic_in_word(word: str):
    """Convert common Latin substitutions of Cyrillic letters back to the original ones, for example, "с" -> "c"."""
    translation_table = str.maketrans(
        "cyeaopxurkbnmETYOPAHKXCBM",
        "суеаорхигкьпмЕТУОРАНКХСВМ",
    )
    cyrillic_letter_pattern = re.compile(r"[а-яё]", re.IGNORECASE)
    if cyrillic_letter_pattern.search(word):
        return word.translate(translation_table)
    else:
        return word
