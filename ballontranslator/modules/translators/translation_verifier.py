"""Translation Verifier & Auditor.

Audits translated blocks against source text blocks to catch:
- Omissions (dropped lines or bubbles)
- Degenerate repetition loops (hallucinations)
- Empty translations on non-empty sources
And performs targeted self-correction passes to ensure 100% complete translation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ballontranslator.utils.project.config import pcfg

LOGGER = logging.getLogger(__name__)


def is_repetitive_loop(text: str, min_repeat: int = 12) -> bool:
    """Detect runaway repetitive hallucination loops in translated text."""
    if not text or len(text) < 30:
        return False

    # Check for repeating words or short phrases
    words = text.split()
    if len(words) >= min_repeat:
        for window in range(1, 6):
            for i in range(len(words) - window * 3):
                chunk = words[i:i + window]
                next_chunk = words[i + window:i + window * 2]
                next_next = words[i + window * 2:i + window * 3]
                if chunk == next_chunk == next_next and len(chunk) > 0:
                    return True

    # Regex for repeating characters/syllables
    match = re.search(r'(.{2,15}?)\1{' + str(min_repeat) + r',}', text)
    return bool(match)


def audit_translation(
    source_texts: Sequence[str],
    translated_texts: Sequence[str],
) -> Tuple[bool, List[int], str]:
    """Audit translation completeness and formatting.
    
    Returns:
        (is_valid, bad_indices, description)
    """
    n_src = len(source_texts)
    n_trans = len(translated_texts)

    if n_src != n_trans:
        return (
            False,
            list(range(min(n_src, n_trans), max(n_src, n_trans))),
            f"Line count mismatch: expected {n_src} lines, got {n_trans}."
        )

    bad_indices: List[int] = []
    reasons: List[str] = []

    for i in range(n_src):
        src = str(source_texts[i] or '').strip()
        trans = str(translated_texts[i] or '').strip()

        # 1. Dropped line check
        if src and not trans:
            bad_indices.append(i)
            reasons.append(f"Index {i} has source text '{src[:20]}' but empty translation.")
            continue

        # 2. Runaway repetition loop check
        if is_repetitive_loop(trans):
            bad_indices.append(i)
            reasons.append(f"Index {i} detected runaway repetition loop.")
            continue

    if bad_indices:
        return False, bad_indices, "; ".join(reasons[:3])

    return True, [], "Translation passed audit."


def self_correct_translation(
    translator: Any,
    source_texts: Sequence[str],
    translated_texts: Sequence[str],
    bad_indices: List[int],
    *,
    project: Optional[Any] = None,
    page_key: Optional[str] = None,
) -> List[str]:
    """Perform a targeted self-correction pass for missing or degenerate translation blocks."""
    if not bad_indices:
        return list(translated_texts)

    repaired = list(translated_texts)
    # Ensure list length matches source length
    while len(repaired) < len(source_texts):
        repaired.append('')

    missing_indices = [idx for idx in bad_indices if idx < len(source_texts)]
    if not missing_indices:
        return repaired

    missing_queries = [str(source_texts[idx] or '') for idx in missing_indices]
    LOGGER.info(
        f"[Translation Verifier] Attempting self-correction for {len(missing_indices)} "
        f"problematic blocks: indices {missing_indices}"
    )

    try:
        if hasattr(translator, 'translate'):
            # Call single or batch translate on just the missing queries
            sub_results = translator.translate(
                missing_queries,
                project=project,
                page_key=page_key,
                commit_history_window=False,
            )
            if isinstance(sub_results, list) and len(sub_results) == len(missing_indices):
                for idx, res in zip(missing_indices, sub_results):
                    repaired[idx] = str(res or '').strip()
                LOGGER.info(f"[Translation Verifier] Successfully self-corrected {len(missing_indices)} blocks.")
            elif isinstance(sub_results, str) and len(missing_indices) == 1:
                repaired[missing_indices[0]] = sub_results.strip()
                LOGGER.info(f"[Translation Verifier] Successfully self-corrected single block.")
    except Exception as e:
        LOGGER.warning(f"[Translation Verifier] Self-correction pass failed: {e}")

    return repaired
