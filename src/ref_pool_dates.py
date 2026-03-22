# -*- coding: utf-8 -*-
"""
Dates for temporal BM25 filtering (ReferencePool_Full JSON)
============================================================
Rule: keep only pool documents whose reference date is **strictly before** the
sample cutoff `meta.oa_date` (`ref_date < oa_date`). Documents on the same
calendar day as the OA are excluded.

Per-document reference date: the **first** key below whose value is present and
parsable as a date (aligned with the pool convention: prefer `filing_date`, then
`publication_date`):

  1. filing_date
  2. publication_date
  3. pub_date
  4. published_date
  5. application_filing_date
  6. grant_date
  7. issue_date
  8. patent_issue_date
  9. date
  10. priority_date

If a pool JSON has **no** parseable date, that document is **not** ranked in
temporal BM25 (it cannot be shown to precede the OA).

Cutoff on the OA side: aligned `meta.oa_date` (same as the raw office action
`oa_date` for that round). If `oa_date` is missing or unparsable, **no** pool-side
date filter is applied (full-pool BM25); callers may log a warning.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

REFERENCE_POOL_DATE_KEYS = (
    "filing_date",
    "publication_date",
    "pub_date",
    "published_date",
    "application_filing_date",
    "grant_date",
    "issue_date",
    "patent_issue_date",
    "date",
    "priority_date",
)


def parse_loose_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    head = s[:19].replace("T", " ").split()[0].replace("/", "-")
    if len(head) >= 10:
        try:
            return date.fromisoformat(head[:10])
        except ValueError:
            pass
    compact = re.sub(r"\D", "", s)
    if len(compact) >= 8:
        y, mo, da = int(compact[:4]), int(compact[4:6]), int(compact[6:8])
        try:
            return date(y, mo, da)
        except ValueError:
            pass
    return None


def ref_doc_effective_date(data: Dict[str, Any]) -> Tuple[Optional[date], Optional[str]]:
    """Return (parsed_date, key_used) for one ReferencePool JSON object."""
    if not isinstance(data, dict):
        return None, None
    for key in REFERENCE_POOL_DATE_KEYS:
        v = data.get(key)
        if v is None or v == "":
            continue
        d = parse_loose_date(v)
        if d is not None:
            return d, key
    return None, None


def oa_cutoff_date(oa_date_value: Any) -> Optional[date]:
    """Parse `meta.oa_date` / raw `oa_date` for BM25 temporal cutoff."""
    return parse_loose_date(oa_date_value)


def reference_pool_date_policy_log_line() -> str:
    keys = ", ".join(REFERENCE_POOL_DATE_KEYS)
    return (
        "BM25 temporal filter: ReferencePool doc date = first parseable among "
        f"({keys}); keep docs with ref_date < meta.oa_date; skip docs with no parseable date."
    )
