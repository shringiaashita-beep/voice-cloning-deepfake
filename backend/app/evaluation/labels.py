"""Label Encoding & Decoding Module for VoxGuard.

Provides deterministic label encoding mapping human-readable dataset labels to integers:
- human = 0
- synthetic = 1

Labels MUST originate strictly from the validated manifest and are never inferred from filenames.
"""

from typing import Dict

LABEL_TO_CODE: Dict[str, int] = {
    "human": 0,
    "synthetic": 1,
}

CODE_TO_LABEL: Dict[int, str] = {
    0: "human",
    1: "synthetic",
}


def encode_label(label: str) -> int:
    """Encodes string label ('human', 'synthetic') to deterministic integer code (0 or 1).

    Args:
        label: Dataset label string.

    Returns:
        Integer code: 0 for 'human', 1 for 'synthetic'.

    Raises:
        ValueError: If label is unknown or malformed.
    """
    if not isinstance(label, str):
        raise ValueError(f"Label must be a string, got {type(label).__name__}")

    norm_label = label.strip().lower()
    if norm_label not in LABEL_TO_CODE:
        raise ValueError(f"Unknown label '{label}'. Permitted labels: 'human', 'synthetic'.")

    return LABEL_TO_CODE[norm_label]


def decode_label(code: int) -> str:
    """Decodes integer code (0 or 1) to string label ('human' or 'synthetic').

    Args:
        code: Integer label code.

    Returns:
        String label: 'human' for 0, 'synthetic' for 1.

    Raises:
        ValueError: If code is not 0 or 1.
    """
    if isinstance(code, bool) or not isinstance(code, int):
        raise ValueError(f"Label code must be an integer, got {type(code).__name__}")

    if code not in CODE_TO_LABEL:
        raise ValueError(f"Unknown label code '{code}'. Permitted codes: 0 (human), 1 (synthetic).")

    return CODE_TO_LABEL[code]
