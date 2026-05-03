#Dates for temporal BM25 filtering (ReferencePool_Full)

#Rule: keep only pool documents whose reference date is strictly before the sample

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
    return parse_loose_date(oa_date_value)


def reference_pool_date_policy_log_line() -> str:
    keys = ", ".join(REFERENCE_POOL_DATE_KEYS)
    return (
        #BM25 temporal filter
        f"({keys}); keep docs with ref_date < meta.oa_date"
    )
