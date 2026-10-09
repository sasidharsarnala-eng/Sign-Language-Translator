"""Text cleanup and clearly labeled demonstration responses."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextResult:
    """Processed manual input shown to the user."""

    original: str
    normalized: str
    response: str
    kind: str


def process_manual_text(text: str) -> TextResult:
    """Normalize text and return a useful, non-authoritative demo response."""
    original = text.strip()
    normalized = re.sub(r"\s+", " ", original)

    if not normalized:
        return TextResult(original="", normalized="", response="", kind="empty")

    folded = normalized.casefold().strip(" .!?")
    if folded in {"hi", "hello", "hey"}:
        return TextResult(
            original=original,
            normalized=normalized,
            response=(
                "Greeting recognized. The demo reference is “HELLO”; "
                "this is a placeholder, not a validated ASL translation."
            ),
            kind="greeting",
        )

    if re.fullmatch(r"[A-Za-z]", normalized):
        return TextResult(
            original=original,
            normalized=normalized,
            response=(
                f"Letter captured: {normalized.upper()}. The live camera panel can "
                "show illustrative hand cues, not validated sign recognition."
            ),
            kind="letter",
        )

    if re.fullmatch(r"[A-Za-z]+", normalized):
        return TextResult(
            original=original,
            normalized=normalized,
            response=(
                f"Word captured: “{normalized}”. This prototype only includes a "
                "sample greeting response, not a complete sign-language dictionary."
            ),
            kind="word",
        )

    if re.fullmatch(r"[A-Za-z ]+", normalized):
        letters = [character.upper() for character in normalized if character.isalpha()]
        return TextResult(
            original=original,
            normalized=normalized,
            response=(
                f"Sentence captured: {' · '.join(normalized.split())}. "
                "A sentence-level sign translation model is not included."
            ),
            kind="sentence",
        )

    return TextResult(
        original=original,
        normalized=normalized,
        response=(
            "Sentence captured. A full sentence-to-sign translation model is not "
            "included in this educational prototype."
        ),
        kind="sentence",
    )
