from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ── Engine instances ──────────────────────────────────────────────────────────
# Created once at module load — these are expensive to initialise.
# AnalyzerEngine  : detects PII entities in text using spaCy + rule-based recognisers
# AnonymizerEngine: replaces detected entities with masked placeholders
analyzer  = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ── Entity types to detect and mask ──────────────────────────────────────────
# Full list of supported entities:
# https://microsoft.github.io/presidio/supported_entities/
#
# We cover the most common PII types. Add or remove based on your use case.
ENTITIES_TO_DETECT = [
    "PERSON",           # names       → <PERSON>
    "EMAIL_ADDRESS",    # emails      → <EMAIL_ADDRESS>
    "PHONE_NUMBER",     # phone nos.  → <PHONE_NUMBER>
    "LOCATION",         # addresses   → <LOCATION>
    "CREDIT_CARD",      # card nos.   → <CREDIT_CARD>
    "IBAN_CODE",        # bank codes  → <IBAN_CODE>
    "IP_ADDRESS",       # IP addrs    → <IP_ADDRESS>
    "DATE_TIME",        # dates/times → <DATE_TIME>  (optional — remove if dates are needed)
]

# ── Anonymizer operators ──────────────────────────────────────────────────────
# "replace" swaps the detected entity with a readable placeholder like <PERSON>.
# Other options: "redact" (removes entirely), "hash", "encrypt", "mask".
# "replace" is best here — the sentence stays grammatically readable,
# which helps the LLM understand context even after masking.
OPERATORS = {
    entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
    for entity in ENTITIES_TO_DETECT
}


def sanitize(text: str, language: str = "en") -> str:
    """
    Detect and mask PII in raw text before storage or processing.

    Args:
        text    : Raw user prompt string
        language: Language code for the analyzer (default "en")
                  Presidio supports en, de, es, fr, it, pt, nl and more.

    Returns:
        Sanitized text with PII replaced by readable placeholders.
        If no PII is found, returns the original text unchanged.

    Example:
        Input : "My name is John and my email is john@example.com"
        Output: "My name is <PERSON> and my email is <EMAIL_ADDRESS>"
    """

    if not text or not text.strip():
        return text

    # ── Step 1: Analyze — find PII entities and their positions ──────────────
    analyzer_results = analyzer.analyze(
        text=text,
        entities=ENTITIES_TO_DETECT,
        language=language,
    )

    # ── No PII found — return original text immediately ───────────────────────
    if not analyzer_results:
        return text

    # ── Step 2: Anonymize — replace detected entities with placeholders ───────
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=OPERATORS,
    )

    return anonymized.text


def has_pii(text: str, language: str = "en") -> bool:
    """
    Check if text contains PII without modifying it.
    Useful for logging or flagging prompts that contained sensitive data.

    Args:
        text    : Text to check
        language: Language code

    Returns:
        True if PII was detected, False otherwise.
    """

    if not text or not text.strip():
        return False

    results = analyzer.analyze(
        text=text,
        entities=ENTITIES_TO_DETECT,
        language=language,
    )

    return len(results) > 0


def get_pii_report(text: str, language: str = "en") -> list[dict]:
    """
    Return a detailed report of all PII found in the text.
    Useful for debugging — shows what was detected and where.

    Args:
        text    : Text to analyze
        language: Language code

    Returns:
        List of dicts, each describing one detected PII entity:
        { "entity_type": "EMAIL_ADDRESS", "start": 10, "end": 28, "score": 0.85 }

    Example usage:
        report = get_pii_report("Call me at 9876543210")
        # [{"entity_type": "PHONE_NUMBER", "start": 11, "end": 21, "score": 0.75}]
    """

    if not text or not text.strip():
        return []

    results = analyzer.analyze(
        text=text,
        entities=ENTITIES_TO_DETECT,
        language=language,
    )

    return [
        {
            "entity_type": r.entity_type,
            "start"      : r.start,
            "end"        : r.end,
            "score"      : round(r.score, 2),  # confidence score 0.0 → 1.0
        }
        for r in results
    ]