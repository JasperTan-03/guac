"""Merge sharded judge outputs back into canonical per-split JSONL files.

After running ``scripts/judge_difficulty.py`` with ``judge.world_size>1``
from N separate processes, this script stitches
``{scored_dir}/{stem}.part{rank}-of-{world_size}.jsonl`` for
``rank in 0..world_size-1`` into ``{scored_dir}/{stem}.jsonl``.

Shards are interleaved so the merged record order matches the
original ``processed/`` input ordering — index ``i`` in the source file
was sent to rank ``i % world_size``, so reading one element from each
rank in round-robin reproduces ``i = 0, 1, 2, ...`` exactly.

After a successful merge, the per-shard ``*.jsonl`` and ``*.jsonl.ckpt``
files are deleted (use ``--keep-shards`` to retain them for debugging).

Usage::

    python scripts/merge_judge_shards.py --world-size 8
    python scripts/merge_judge_shards.py --world-size 4 --splits train val
    python scripts/merge_judge_shards.py --world-size 8 --keep-shards
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from guac.data.prep import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def merge_split(
    scored_dir: Path,
    split: str,
    world_size: int,
    keep_shards: bool,
) -> int:
    """Merge shards for a single split. Returns 0 on success, non-zero on error."""
    shards: List[List[dict]] = []
    shard_paths: List[Path] = []
    for r in range(world_size):
        shard_path = scored_dir / f"{split}.part{r}-of-{world_size}.jsonl"
        if not shard_path.exists():
            log.error("Missing shard for split=%s rank=%d: %s", split, r, shard_path)
            return 1
        shards.append(load_jsonl(str(shard_path)))
        shard_paths.append(shard_path)

    # Interleave to reconstruct original order: original index i was sent
    # to rank (i % world_size), so rank r's k-th element was original
    # index r + k*world_size. Round-robin pops reproduce 0, 1, 2, ...
    max_len = max(len(s) for s in shards)
    merged: List[dict] = []
    for k in range(max_len):
        for r in range(world_size):
            if k < len(shards[r]):
                merged.append(shards[r][k])

    out_path = scored_dir / f"{split}.jsonl"
    save_jsonl(merged, str(out_path))
    log.info(
        "merged split=%s | shards=%d | records=%d | -> %s",
        split,
        world_size,
        len(merged),
        out_path,
    )

    if not keep_shards:
        for shard_path in shard_paths:
            ckpt_path = shard_path.with_suffix(shard_path.suffix + ".ckpt")
            shard_path.unlink(missing_ok=True)
            ckpt_path.unlink(missing_ok=True)
        log.info("cleaned up %d shard+ckpt files for split=%s", 2 * world_size, split)

    return 0


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--scored-dir",
        type=Path,
        default=Path("data/scored"),
        help="Directory containing the per-shard JSONL files (default: data/scored).",
    )
    p.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Number of shards to merge (matches judge.world_size from the judging run).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split stems to merge (default: train val test).",
    )
    p.add_argument(
        "--keep-shards",
        action="store_true",
        help="Retain per-rank shard and ckpt files after merging.",
    )
    args = p.parse_args()

    if args.world_size < 1:
        log.error("--world-size must be >= 1, got %d", args.world_size)
        return 2

    rc = 0
    for split in args.splits:
        rc = max(rc, merge_split(args.scored_dir, split, args.world_size, args.keep_shards))
    return rc


if __name__ == "__main__":
    sys.exit(main())
