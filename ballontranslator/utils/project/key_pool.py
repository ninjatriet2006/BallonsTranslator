"""Smart LLM Key Pool and Rotation Manager.

Supports:
- Auto-detecting whether an API key setting is a raw key string or a Keys File path.
- Loading keys from text files (one key per line, with comment support) or JSON files.
- Thread-safe round-robin rotation across concurrent workers.
- Rate-limit cooldown and automatic failover on HTTP 429 / ResourceExhausted errors.
- Automatic hot-reloading when the keys file is modified on disk.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

_KEY_FILE_EXTENSIONS = ('.txt', '.json', '.key', '.keys')


def clean_key_file_path(value: Any) -> str:
    """Normalize and expand a potential file path string."""
    if not value or not isinstance(value, str):
        return ""
    cleaned = value.strip().strip('"').strip("'")
    return os.path.normpath(os.path.expanduser(cleaned))


def is_key_file(value: Any) -> bool:
    """Determine whether a string is a Keys File path rather than a direct API key.

    Heuristics:
    1. An existing file on the filesystem is always treated as a Keys File.
    2. A path-like string ending with a known key file extension (.txt, .json, .key, .keys)
       is treated as a Keys File intent even if it does not yet exist.
    3. Direct API keys (e.g. 'sk-1234...', 'AIzaSy...') never match existing files or extensions.
    """
    if not value or not isinstance(value, str):
        return False
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned or '\n' in cleaned:
        return False

    expanded = os.path.expanduser(cleaned)
    if os.path.isfile(expanded):
        return True

    lowered = cleaned.lower()
    has_ext = any(lowered.endswith(ext) for ext in _KEY_FILE_EXTENSIONS)
    is_path_like = (
        '/' in cleaned
        or '\\' in cleaned
        or cleaned.startswith('~')
        or cleaned.startswith('.')
        or bool(re.match(r'^[a-zA-Z]:[/\\]', cleaned))
    )
    return has_ext or (is_path_like and has_ext)


def load_keys_from_file(file_path: str) -> List[str]:
    """Parse API keys from a file (.txt or .json), ignoring blank lines and comments.

    Deduplicates keys while preserving their file ordering.
    """
    path = clean_key_file_path(file_path)
    if not os.path.isfile(path):
        return []

    try:
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            content = f.read().strip()
    except Exception:
        return []

    if not content:
        return []

    keys: List[str] = []
    if path.lower().endswith('.json'):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                keys = [str(k).strip() for k in data if str(k).strip()]
            elif isinstance(data, dict):
                for candidate_field in ('keys', 'api_keys', 'key_list'):
                    field_val = data.get(candidate_field)
                    if isinstance(field_val, list):
                        keys = [str(k).strip() for k in field_val if str(k).strip()]
                        break
        except Exception:
            # Fall back to text parsing if JSON parse fails
            pass

    if not keys:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Handle inline comments (e.g. key_one # note or key_two // note)
            for marker in (' #', ' //', '\t#', '\t//'):
                if marker in line:
                    line = line.split(marker, 1)[0].strip()
            # Strip trailing commas or quotes if accidentally copied
            cleaned_line = line.rstrip(',').strip().strip('"').strip("'")
            if cleaned_line:
                keys.append(cleaned_line)

    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


class KeyPool:
    """Thread-safe key pool supporting round-robin rotation, cooldowns, and hot-reload."""

    def __init__(self, file_path: str):
        self.file_path = clean_key_file_path(file_path)
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._index: int = 0
        self._mtime: float = 0.0
        self._cooldowns: Dict[str, float] = {}  # key -> expiry timestamp
        self.reload(force=True)

    def reload(self, force: bool = False) -> bool:
        """Reload keys if the file modification time changed or forced."""
        if not os.path.isfile(self.file_path):
            with self._lock:
                self._keys = []
                self._mtime = 0.0
            return False

        try:
            mtime = os.path.getmtime(self.file_path)
        except OSError:
            mtime = 0.0

        if not force and mtime <= self._mtime:
            return False

        loaded = load_keys_from_file(self.file_path)
        with self._lock:
            self._keys = loaded
            self._mtime = mtime
            if self._index >= len(self._keys):
                self._index = 0
        return True

    def get_all_keys(self) -> List[str]:
        """Return a snapshot of all currently loaded keys."""
        self.reload()
        with self._lock:
            return list(self._keys)

    def count(self) -> int:
        """Return the count of available keys."""
        self.reload()
        with self._lock:
            return len(self._keys)

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0) -> None:
        """Temporarily put a key on cooldown due to rate limit (429 / quota limit)."""
        if not key:
            return
        with self._lock:
            self._cooldowns[key] = time.time() + max(1.0, float(cooldown_seconds))

    def is_rate_limited(self, key: str) -> bool:
        """Check whether a key is currently in cooldown."""
        with self._lock:
            expiry = self._cooldowns.get(key, 0.0)
            return time.time() < expiry

    def get_next_key(self) -> str:
        """Select the next available key using round-robin, skipping keys in cooldown."""
        self.reload()
        now = time.time()

        with self._lock:
            if not self._keys:
                return ""

            # Purge expired cooldowns
            self._cooldowns = {k: exp for k, exp in self._cooldowns.items() if exp > now}

            total = len(self._keys)
            # Find the next key not on cooldown
            for offset in range(total):
                idx = (self._index + offset) % total
                candidate = self._keys[idx]
                if candidate not in self._cooldowns:
                    self._index = (idx + 1) % total
                    return candidate

            # If all keys are in cooldown, pick the one with earliest cooldown expiration
            sorted_by_expiry = sorted(
                self._keys,
                key=lambda k: self._cooldowns.get(k, 0.0),
            )
            fallback = sorted_by_expiry[0]
            self._index = (self._keys.index(fallback) + 1) % total
            return fallback


_KEY_POOLS: Dict[str, KeyPool] = {}
_POOLS_LOCK = threading.Lock()


def get_key_pool(file_path: str) -> KeyPool:
    """Retrieve or create a cached KeyPool for a file path."""
    normalized = clean_key_file_path(file_path)
    with _POOLS_LOCK:
        if normalized not in _KEY_POOLS:
            _KEY_POOLS[normalized] = KeyPool(normalized)
        return _KEY_POOLS[normalized]


def resolve_active_api_key(profile: Any) -> str:
    """Resolve the active API key for a profile, rotating if using a Keys File.

    If the profile's configured api_key is a file path:
        Retrieves the next round-robin key from its KeyPool.
    Otherwise:
        Returns the resolved direct API key.
    """
    from ballontranslator.utils.project.llm_profiles import resolve_api_key

    raw_val = resolve_api_key(profile).strip()
    if not is_key_file(raw_val):
        return raw_val

    pool = get_key_pool(raw_val)
    return pool.get_next_key()


def count_profile_keys(profile: Any) -> int:
    """Return the total number of keys available for a profile."""
    from ballontranslator.utils.project.llm_profiles import resolve_api_key

    raw_val = resolve_api_key(profile).strip()
    if not raw_val:
        return 0
    if not is_key_file(raw_val):
        return 1

    pool = get_key_pool(raw_val)
    return pool.count()
