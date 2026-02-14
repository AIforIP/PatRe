
# 用法示例

# 前 15 个唯一专利号（不限 IPC）
# python tools/make_subset.py --input dataset/processed/test_set.jsonl --output dataset/processed/test_set_15p.jsonl --num-ids 15 --mode first
# 前 20 个唯一专利号，限定 IPC=A
# python tools/make_subset.py --input dataset/processed/test_set.jsonl --output dataset/processed/test_set_20p_A.jsonl --num-ids 20 --ipc A --mode first
# 随机 15 个专利号（可复现）
# 

# python tools/make_subset.py --input dataset/processed/test_set.jsonl --output dataset/processed/test_set_15p_rand.jsonl --num-ids 5 --mode random --seed 42



import argparse
import json
import random
from collections import OrderedDict
from typing import Iterable, List, Dict


def _iter_items(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _ipc_match(item: dict, ipc_filter: str | None) -> bool:
    if not ipc_filter:
        return True
    ipc_value = str(item.get("meta", {}).get("ipc", "")).upper()
    return ipc_value == ipc_filter


def _pick_ids_first(items: List[dict], max_ids: int) -> List[str]:
    seen = OrderedDict()
    for item in items:
        pid = item.get("id")
        if pid is None:
            continue
        if pid not in seen:
            seen[pid] = True
            if len(seen) >= max_ids:
                break
    return list(seen.keys())


def _pick_ids_random(items: List[dict], max_ids: int, seed: int | None) -> List[str]:
    ids = []
    seen = set()
    for item in items:
        pid = item.get("id")
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:max_ids]


def _filter_rows_by_ids(path: str, ids_set: set, ipc_filter: str | None) -> List[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if not _ipc_match(item, ipc_filter):
                continue
            pid = item.get("id")
            if pid in ids_set:
                rows.append(line)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a subset of test_set.jsonl by unique patent ids")
    parser.add_argument("--input", required=True, help="Source jsonl path")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--num-ids", type=int, required=True, help="Number of unique patent ids")
    parser.add_argument("--ipc", type=str, default=None, help="IPC filter (A-H)")
    parser.add_argument("--mode", choices=["first", "random"], default="first", help="Pick ids by order or random")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (for mode=random)")
    args = parser.parse_args()

    ipc_filter = args.ipc.upper() if args.ipc else None

    items = [item for item in _iter_items(args.input) if _ipc_match(item, ipc_filter)]
    if not items:
        raise SystemExit("No items matched the IPC filter.")

    if args.mode == "first":
        picked_ids = _pick_ids_first(items, args.num_ids)
    else:
        picked_ids = _pick_ids_random(items, args.num_ids, args.seed)

    if not picked_ids:
        raise SystemExit("No ids selected.")

    rows = _filter_rows_by_ids(args.input, set(picked_ids), ipc_filter)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line)

    print(f"Wrote {len(rows)} lines for {len(set(picked_ids))} unique ids to {args.output}")


if __name__ == "__main__":
    main()
