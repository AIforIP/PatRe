# src/utils.py
import json
import logging
import os


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each record so `tail -f experiment.log` updates live."""

    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def load_dataset_json(path):
    """
    Load a test set from either:
    - A JSON file containing an array of sample objects: [ {...}, {...} ]
    - JSONL: one JSON object per line (legacy)

    Returns a list of dicts (same order as in file).
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw_stripped = raw.strip()
    if not raw_stripped:
        return []
    lead = raw_stripped.lstrip()
    if lead.startswith("["):
        try:
            parsed = json.loads(raw_stripped)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
    out = []
    for line in raw.splitlines():
        if not line or not line.strip():
            continue
        s = line.lstrip()
        if s.startswith("//"):
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
        except json.JSONDecodeError:
            continue
    return out

def setup_logger(exp_dir):
    logger = logging.getLogger("PatRe_Eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Replace handlers so each experiment directory gets its own experiment.log
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = FlushingFileHandler(
        os.path.join(exp_dir, "experiment.log"),
        encoding="utf-8",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger