"""Image loading and timing shared by the benchmarks."""

import os
import statistics
import time
import urllib.request
from urllib.parse import urlparse

from PIL import Image


def timed(fn, warmup: int, repeats: int):
    """Run ``fn`` warmup + repeats times; return (median seconds, last result)."""
    for _ in range(warmup):
        fn()
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def load_image(url: str, cache_dir: str = "/tmp/vflorence/coco") -> Image.Image:
    path = os.path.join(cache_dir, os.path.basename(urlparse(url).path))
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with Image.open(path) as image:
        return image.convert("RGB")
