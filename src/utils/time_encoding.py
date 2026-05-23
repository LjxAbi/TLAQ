"""
Seven-level binary time encoding  (TLAQ Section 3.5, Figure 7).

Encoding layout (total 52 bits, one-hot per sub-vector):

  Slot  Name  Size  Range            Description
  ────  ────  ────  ───────────────  ──────────────────────────────────
   C    cent   13   3+10 bits        3 millennium bits (AD 0–999/1000–1999/2000–2999)
                                     + 10 century-within-millennium bits
   D    dec    10   0–9              decade within century
   Y    year   10   0–9              year within decade
   Q    qtr     4   0–3              quarter (0=Q1 Jan–Mar … 3=Q4 Oct–Dec)
   M    mon     3   0–2              month within quarter (0=first, 1=second, 2=third)
   W    wk      5   0–4              week within month  (1st of month → week 0)
   DS   day     7   0–6              day of week  (0=Mon … 6=Sun, ISO weekday)

The paper gives an interval-level example (Figure 7):
  C=[0,1,0] (2nd millennium) + [0,0,1,0,…] (3rd century) →  year 1200s
  D=[0,0,1,…]  (3rd decade of that century)               →  year 122x
  Y=[0,0,0,1,…](4th year of that decade)                  →  year 1223

Both point timestamps and intervals [τs, τe] are supported:
  - Point   → encode_timestamp(year, month, day)          → Tensor[52]
  - Interval → encode_interval(year_s, ..., year_e, ...)  → Tensor[104]
"""
from __future__ import annotations

import datetime
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# bit widths of each sub-vector, in order
_WIDTHS = {
    "C_millennium": 3,   # which millennium (0,1,2)
    "C_century":   10,   # century within millennium (0–9)
    "D":           10,   # decade within century    (0–9)
    "Y":           10,   # year within decade        (0–9)
    "Q":            4,   # quarter                   (0–3)
    "M":            3,   # month within quarter      (0–2)
    "W":            5,   # week within month         (0–4)
    "DS":           7,   # day of week               (0–6)
}

TIME_DIM: int = sum(_WIDTHS.values())   # 52

# cumulative offsets for slicing
_OFFSETS: dict[str, int] = {}
_off = 0
for _name, _w in _WIDTHS.items():
    _OFFSETS[_name] = _off
    _off += _w


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _one_hot_slot(idx: int, size: int, out: torch.Tensor, offset: int) -> None:
    """Write a one-hot value at out[offset:offset+size].  Clips idx silently."""
    idx = max(0, min(idx, size - 1))
    out[offset + idx] = 1.0


def _decompose_date(year: int, month: int, day: int) -> dict[str, int]:
    """Break a calendar date into the seven-level indices."""
    # clamp month / day to valid ranges
    month = max(1, min(month, 12))
    day   = max(1, min(day,   31))

    millennium          = year // 1000
    century_in_mil      = (year % 1000) // 100
    decade_in_cent      = (year % 100) // 10
    year_in_decade      = year % 10
    quarter             = (month - 1) // 3           # 0–3
    month_in_quarter    = (month - 1) % 3            # 0–2
    week_in_month       = (day - 1) // 7             # 0–4  (paper: 1st = start of w0)

    # day-of-week via Python's datetime (ISO: Mon=0 … Sun=6)
    try:
        dow = datetime.date(year, month, min(day, 28)).weekday()   # 0–6
    except ValueError:
        dow = 0

    return {
        "C_millennium": millennium,
        "C_century":    century_in_mil,
        "D":            decade_in_cent,
        "Y":            year_in_decade,
        "Q":            quarter,
        "M":            month_in_quarter,
        "W":            week_in_month,
        "DS":           dow,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_timestamp(
    year:  int,
    month: int = 1,
    day:   int = 1,
) -> torch.Tensor:
    """
    Encode a single calendar date into a 52-bit binary vector.

    Each of the seven levels is represented by a one-hot sub-vector.
    All bits are float32 (0.0 / 1.0) for compatibility with nn.LSTM.
    """
    vec = torch.zeros(TIME_DIM, dtype=torch.float32)
    parts = _decompose_date(year, month, day)

    for name, idx in parts.items():
        size   = _WIDTHS[name]
        offset = _OFFSETS[name]
        _one_hot_slot(idx, size, vec, offset)

    return vec   # [52]


def encode_interval(
    year_s:  int,
    year_e:  int,
    month_s: int = 1,
    month_e: int = 1,
    day_s:   int = 1,
    day_e:   int = 1,
) -> torch.Tensor:
    """
    Encode a temporal interval [τs, τe] as a 104-bit vector by
    concatenating the start and end timestamp encodings.

    If year_e is None (open-ended interval), it is set to year_s + 1
    so the model still receives a valid end encoding.
    """
    if year_e is None:
        year_e = year_s + 1

    ts = encode_timestamp(year_s, month_s, day_s)   # [52]
    te = encode_timestamp(year_e, month_e, day_e)   # [52]
    return torch.cat([ts, te], dim=0)               # [104]


def batch_encode_timestamps(
    years:  list[int],
    months: Optional[list[int]] = None,
    days:   Optional[list[int]] = None,
) -> torch.Tensor:
    """Vectorised batch encoding. Returns [N, 52]."""
    N = len(years)
    months = months or [1] * N
    days   = days   or [1] * N
    return torch.stack(
        [encode_timestamp(y, m, d) for y, m, d in zip(years, months, days)]
    )   # [N, 52]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def decode_timestamp(vec: torch.Tensor) -> dict[str, int]:
    """
    Reverse-map a 52-bit vector back to human-readable level indices.
    Useful for debugging the encoding.
    """
    assert vec.shape == (TIME_DIM,), f"Expected [52], got {vec.shape}"
    result = {}
    for name, size in _WIDTHS.items():
        offset  = _OFFSETS[name]
        segment = vec[offset: offset + size]
        hot     = segment.argmax().item()
        result[name] = int(hot)
    return result


def time_encoding_info() -> str:
    lines = ["Seven-level time encoding (TLAQ Section 3.5)"]
    lines.append(f"Total dimension : {TIME_DIM}")
    lines.append("-" * 48)
    lines.append(f"{'Level':<16} {'Bits':>4}  {'Offset':>6}  {'Range'}")
    lines.append("-" * 48)
    for name, size in _WIDTHS.items():
        lines.append(f"{name:<16} {size:>4}  {_OFFSETS[name]:>6}  0 – {size-1}")
    return "\n".join(lines)
