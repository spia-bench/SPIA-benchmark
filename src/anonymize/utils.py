"""
Anonymization utilities
"""


def check_length_anomaly(original: str, anonymized: str, threshold: float = 0.4) -> dict:
    """
    Check length difference between original and anonymized text.

    Args:
        original: Original text
        anonymized: Anonymized text
        threshold: Ratio threshold for anomaly detection (default: 0.3 = 30%)

    Returns:
        dict with:
            - length_diff: Absolute character difference
            - length_ratio: anonymized/original ratio
            - length_anomaly: True if ratio deviates more than threshold from 1.0
    """
    orig_len = len(original)
    anon_len = len(anonymized)

    length_diff = abs(orig_len - anon_len)

    if orig_len == 0:
        length_ratio = 0.0 if anon_len == 0 else float('inf')
    else:
        length_ratio = anon_len / orig_len

    # Anomaly if ratio deviates more than threshold from 1.0
    # e.g., threshold=0.3 means anomaly if ratio < 0.7 or ratio > 1.3
    length_anomaly = abs(length_ratio - 1.0) > threshold

    return {
        "length_diff": length_diff,
        "length_ratio": round(length_ratio, 3),
        "length_anomaly": length_anomaly
    }
