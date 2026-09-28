#!/usr/bin/env python3
"""TEST ONLY: assign posting lists to partitions in contiguous ranges of list IDs.

How lists should be grouped into partitions is open design question U2 (graph locality,
k-means over centroids, hashing, ...). This placeholder exists so that the pipeline can be
exercised end to end; do not use it for experiments.

    python3 scripts/testing/make_range_assignment.py INDEX_DIR NUM_PARTITIONS OUT.bin
"""

import struct
import sys


def main(index_dir: str, parts: int, out: str) -> None:
    with open(f"{index_dir}/heads.bin", "rb") as f:  # one head per posting list
        (num_lists,) = struct.unpack("<Q", f.read(8))
    assign = [c * parts // num_lists for c in range(num_lists)]
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", num_lists))
        f.write(struct.pack(f"<{num_lists}I", *assign))
    print(f"{num_lists} lists -> {parts} partitions (TEST-ONLY range assignment) -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: make_range_assignment.py INDEX_DIR NUM_PARTITIONS OUT.bin")
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
