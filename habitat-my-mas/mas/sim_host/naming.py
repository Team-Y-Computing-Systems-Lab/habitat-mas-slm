"""Turn scene handles (hashes, dataset prefixes) into names an SLM can read.

  061_foam_brick_:0000                        -> foam_brick
  frl_apartment_wall_cabinet_01_:0000         -> wall_cabinet_01
  e8dd67ee..._part_3_:0003 (HSSD hash)        -> shelves   (from HSSD metadata)

Names are not unique; `NameTable` appends _0, _1, ... per episode.
"""

import csv
import os
import re
from collections import defaultdict
from typing import Dict, Optional

HSSD_METADATA = "data/scene_datasets/hssd-hab/metadata/fpmodels-with-decomposed.csv"

_HASH_RE = re.compile(r"^([0-9a-f]{40})")
_hssd_names: Optional[Dict[str, str]] = None


def _load_hssd_names() -> Dict[str, str]:
    global _hssd_names
    if _hssd_names is None:
        _hssd_names = {}
        if os.path.exists(HSSD_METADATA):
            with open(HSSD_METADATA, newline="") as f:
                for row in csv.DictReader(f):
                    name = row.get("main_category") or row.get("wnsynsetkey", "")
                    name = name.split(".")[0]  # "shelf.n.01" -> "shelf"
                    if name:
                        _hssd_names[row["id"]] = name
    return _hssd_names


def base_name(handle: str) -> str:
    """Readable, non-unique name for a scene handle."""
    h = handle.split(":")[0].rstrip("_")
    m = _HASH_RE.match(h)
    if m:
        name = _load_hssd_names().get(m.group(1), "object")
    else:
        h = re.sub(r"^\d+(-[a-z])?[-_]", "", h)  # YCB index: "061_", "065-f_", "070-a_"
        name = h.replace("frl_apartment_", "")
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "object"


class NameTable:
    """Unique readable names within one episode, stable per handle."""

    def __init__(self):
        self._by_handle: Dict[str, str] = {}
        self._counts: Dict[str, int] = defaultdict(int)

    def name(self, handle: str) -> str:
        if handle not in self._by_handle:
            base = base_name(handle)
            self._by_handle[handle] = f"{base}_{self._counts[base]}"
            self._counts[base] += 1
        return self._by_handle[handle]

    def handle(self, name: str) -> Optional[str]:
        for h, n in self._by_handle.items():
            if n == name:
                return h
        return None
