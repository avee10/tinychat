"""

Usage patterns supported:
  python your_script.py path/to/config.py --depth=4 --max_seq_len 256 --flag --lr=3e-4

Rules:
- Positional tokens that do NOT start with '-' are treated as config files and executed.
- Flags can be `--key=value` or `--key value`.
- Negative numbers (e.g. `-1`) are accepted as values.
- Booleans:
    * If the existing global default is bool and you pass bare `--flag`, it becomes True.
    * You can also pass explicit values: --flag=false, --flag 0, etc.
- Types are checked against existing globals when possible. int -> float is allowed.
"""

import os
import sys
from ast import literal_eval

# ---------------------------------------------------------------------------

def print0(s: str = "", **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def _is_config_file(token: str) -> bool:
    # Treat any non-flag token as a config file if it exists on disk
    # (keeps behavior simple & explicit)
    return (not token.startswith("-")) and os.path.exists(token)

def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    # Fallback: non-empty string => True
    return bool(s)

def _literal_or_str(token: str):
    # Try literal_eval; if it fails, return raw string
    try:
        return literal_eval(token)
    except Exception:
        return token

def _looks_like_value(token: str) -> bool:
    # Accept values that literal-eval (incl. negative numbers), or plain strings
    # but reject things that look like another flag.
    if token.startswith("--"):
        return False
    # Allow a single '-' (unlikely) or '-1', etc. by trying literal_eval
    val = _literal_or_str(token)
    # Anything is a value unless it’s clearly another flag
    return True

def _type_compatible(existing, new):
    if existing is None:
        return True
    t0 = type(existing)
    t1 = type(new)
    if t0 is float and t1 is int:
        return True
    return t0 == t1

def _maybe_cast(existing, new):
    # allow int -> float upcast
    if isinstance(existing, float) and isinstance(new, int):
        return float(new)
    return new

# ---------------------------------------------------------------------------

# Gather tokens, but ignore a literal standalone `--` separator if present
raw_tokens = [t for t in sys.argv[1:] if t != "--"]

# First pass: execute any config files (positional tokens that exist on disk)
i = 0
while i < len(raw_tokens):
    tok = raw_tokens[i]
    if _is_config_file(tok):
        print0(f"Overriding config with {tok}:")
        with open(tok, "r") as f:
            print0(f.read())
        exec(open(tok).read())
        i += 1
    else:
        i += 1

# Second pass: parse flags and apply into caller's globals()
i = 0
while i < len(raw_tokens):
    tok = raw_tokens[i]

    # skip anything that isn't a flag (we already handled config files)
    if not tok.startswith("--"):
        i += 1
        continue

    # strip leading dashes
    keyval = tok[2:]

    # case A: --key=value
    if "=" in keyval:
        key, val_str = keyval.split("=", 1)
        val = _literal_or_str(val_str)

        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")

        existing = globals()[key]
        # Special-case booleans
        if isinstance(existing, bool):
            val = _coerce_bool(val)

        # type safety (with int->float upcast)
        if not _type_compatible(existing, val):
            raise AssertionError(f"Type mismatch for --{key}: got {type(val).__name__}, expected {type(existing).__name__}")
        val = _maybe_cast(existing, val)

        print0(f"Overriding: {key} = {val}")
        globals()[key] = val
        i += 1
        continue

    # case B: --key value  (value may be negative)
    key = keyval
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")

    # If next token exists and looks like a value, consume it.
    if (i + 1) < len(raw_tokens) and _looks_like_value(raw_tokens[i + 1]):
        val_token = raw_tokens[i + 1]
        val = _literal_or_str(val_token)
        existing = globals()[key]

        if isinstance(existing, bool):
            val = _coerce_bool(val)

        if not _type_compatible(existing, val):
            raise AssertionError(f"Type mismatch for --{key}: got {type(val).__name__}, expected {type(existing).__name__}")
        val = _maybe_cast(existing, val)

        print0(f"Overriding: {key} = {val}")
        globals()[key] = val
        i += 2
        continue

    # case C: bare --flag for booleans (toggle True)
    existing = globals()[key]
    if isinstance(existing, bool):
        print0(f"Overriding: {key} = True")
        globals()[key] = True
        i += 1
    else:
        raise AssertionError(
            f"Flag '--{key}' missing value. Use --{key}=VALUE or '--{key} VALUE'. "
            f"(Note: bare '--{key}' only works for booleans.)"
        )
