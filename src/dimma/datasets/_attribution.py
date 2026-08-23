"""One-time notices, printed to stderr.

Library code should not write to stdout — a caller piping results cannot
tell the notice from the output. These go to stderr, once per process per
key, so a loop over configurations does not repeat them.
"""

from __future__ import annotations

import sys
import threading

_emitted_keys: set[str] = set()
_emit_lock = threading.Lock()


def emit_once(key: str, message: str) -> None:
    """Print ``message`` to stderr exactly once per process per ``key``."""
    with _emit_lock:
        if key in _emitted_keys:
            return
        _emitted_keys.add(key)
        print(message, file=sys.stderr)


def reset_emitted() -> None:
    """Forget every emitted key, so the next ``emit_once`` prints again.

    For tests. Production code has no reason to replay a notice.
    """
    with _emit_lock:
        _emitted_keys.clear()
