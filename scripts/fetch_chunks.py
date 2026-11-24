"""Read parts of the full SubPipe archive (28 GB) straight from Zenodo, without downloading all of it.

Zenodo answers HTTP range requests, so the zip's table of contents and just the wanted files can be read remotely.
Usage:
  .venv/bin/python -I scripts/fetch_chunks.py list                          # chunks, folders, sizes
  .venv/bin/python -I scripts/fetch_chunks.py get Chunk3 SSS_HF_images EstimatedState.csv
  .venv/bin/python -I scripts/fetch_chunks.py get Chunk3 --every 5 SSS_HF_images   # every 5th image, all labels
Files land in data/SubPipe/DATA/<chunk>/..., outside git.
"""

import http.client
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parent.parent
RECORD = "https://zenodo.org/api/records/12666132"


def fetch(url, start, end):
    for attempt in range(5):  # big transfers meet the odd network hiccup
        try:
            with urlopen(Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=120) as r:
                return r.headers, r.read()
        except (OSError, http.client.HTTPException):  # includes a connection cut mid-read (IncompleteRead)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


class Remote(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(fetch(url, 0, 0)[0]["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        data = fetch(self.url, self.pos, min(self.pos + len(buf), self.size) - 1)[1]
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


url = next(f["links"]["self"] for f in json.loads(urlopen(RECORD, timeout=60).read())["files"] if f["key"] == "SubPipe.zip")
archive = zipfile.ZipFile(io.BufferedReader(Remote(url), buffer_size=1 << 20))  # small: every jump refills the buffer
entries = [i for i in archive.infolist() if not i.is_dir()]

if sys.argv[1:2] == ["list"]:
    size, count = defaultdict(int), defaultdict(int)
    for i in entries:
        parts = PurePosixPath(i.filename).parts
        key = "/".join(parts[: parts.index("DATA") + 3] if "DATA" in parts else parts[:3])
        size[key] += i.compress_size
        count[key] += 1
    for key in sorted(size):
        print(f"{size[key] / 1e9:7.2f} GB  {count[key]:6d} files  {key}")
elif sys.argv[1:2] == ["get"] and len(sys.argv) > 3:
    chunk, args = sys.argv[2], sys.argv[3:]
    every = int(args[args.index("--every") + 1]) if "--every" in args else 1
    wanted = [a for k, a in enumerate(args) if a != "--every" and (k == 0 or args[k - 1] != "--every")]
    picks = [i for i in entries if any(f"/{chunk}/{w}" in f"/{i.filename}" for w in wanted)]
    images = defaultdict(list)  # thin out images only; labels and logs are small and always kept
    for i in picks:
        if i.filename.endswith((".jpg", ".pbm")):
            images[str(PurePosixPath(i.filename).parent)].append(i)
    keep = {i.filename for group in images.values() for i in sorted(group, key=lambda i: i.filename)[::every]}
    picks = [i for i in picks if not i.filename.endswith((".jpg", ".pbm")) or i.filename in keep]
    picks.sort(key=lambda i: i.header_offset)  # archive order, so neighbouring small files share one request
    target = root / "data"
    total, start = sum(i.compress_size for i in picks), time.time()
    print(f"{len(picks)} files, {total / 1e9:.2f} GB compressed")
    for n, i in enumerate(picks, 1):
        out = (target / i.filename).resolve()
        assert out.is_relative_to(target.resolve()), f"refusing to write outside data/: {i.filename}"
        if out.exists() and out.stat().st_size == i.file_size:
            continue  # already fetched; lets an interrupted run resume
        out.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(i) as src, open(out, "wb") as dst:
            while block := src.read(4 << 20):
                dst.write(block)
        if n % 200 == 0 or n == len(picks):
            print(f"{n}/{len(picks)} files, {time.time() - start:.0f} s", flush=True)
else:
    sys.exit(__doc__)
