"""Image loading and timing shared by the benchmarks."""

import os
import time
import urllib.request
from urllib.parse import urlparse

from PIL import Image


def timed(fn, warmup: int, repeats: int):
    """Run ``fn`` warmup + repeats times; return (best seconds, last result)."""
    for _ in range(warmup):
        fn()
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def load_image(url: str, cache_dir: str = "/tmp/fastencdec/coco") -> Image.Image:
    path = os.path.join(cache_dir, os.path.basename(urlparse(url).path))
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with Image.open(path) as image:
        return image.convert("RGB")
