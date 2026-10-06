#!/usr/bin/env python3
"""Drop the given files (directories: every file under them) from the page cache.

Without root a run cannot drop the whole page cache (/proc/sys/vm/drop_caches), but it can
drop its own files: an index, partitions and query files cached by the previous run would make
the next one's bootstrap and first reads faster than a clean start. Dirty pages are written
back first; pages another process maps stay cached.

    python3 scripts/evict_cache.py /data/rtier/index /data/rtier/work data/queries.u8bin
"""

import os
import sys


def evict(path: str) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError:
        return 0
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return 1
    except OSError:
        return 0
    finally:
        os.close(fd)


def files_under(path: str):
    if os.path.isdir(path):
        for d, _, names in os.walk(path):
            for n in names:
                yield os.path.join(d, n)
    elif os.path.exists(path):
        yield path


def main(paths) -> None:
    files = done = 0
    for p in paths:
        for f in files_under(p):
            if os.path.isfile(f):  # a symlink counts as its target
                files += 1
                done += evict(f)
    print(f"evicted {done} of {files} files from the page cache")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: evict_cache.py PATH...")
    main(sys.argv[1:])
