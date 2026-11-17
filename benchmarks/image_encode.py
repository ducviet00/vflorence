"""Time Florence-2 image encoding: pixel values -> visual tokens for the BART encoder.

Covers the DaViT vision tower and the multimodal projector, one image per call as
the engine runs it (``ModelRunner._image_embeds``).  Preprocessing (resize /
normalize) and the BART encoder are outside the timed region.

Prints the compiled-engine latency over ``--images`` images, then an eager
per-stage profile (the compiled graph is one opaque call, so the breakdown is
measured in eager mode and shows where the time goes, not the compiled split).

  PYTHONPATH=. python benchmarks/image_encode.py
  PYTHONPATH=. python benchmarks/image_encode.py --model florence-community/Florence-2-large --dtype float32
"""

import argparse
import statistics
import time
from collections import defaultdict

import torch

from benchmarks.continuous_batching import IMAGE_URLs
from benchmarks.utils import load_image


def time_each(fn, pixels) -> list[float]:
    """Seconds per image for ``fn`` on each [1, 3, H, W] image."""
    times = []
    for px in pixels:
        start = time.perf_counter()
        fn(px)
        times.append(time.perf_counter() - start)
    return times


def profile_stages(model, pixels) -> dict[str, float]:
    """Eager seconds per tower stage / projector, summed over ``pixels``."""
    tower, totals, started = model.model.vision_tower, defaultdict(float), {}
    parts = {"projector": [model.model.multi_modal_projector]}
    for i, (conv, blocks) in enumerate(zip(tower.convs, tower.blocks)):
        parts[f"stage {i} conv embed"] = [conv]
        parts[f"stage {i} blocks x{len(blocks)}"] = list(blocks)
    hooks = []
    for name, modules in parts.items():
        for module in modules:
            hooks += [
                module.register_forward_pre_hook(
                    lambda *_, name=name: started.__setitem__(name, time.perf_counter())
                ),
                module.register_forward_hook(
                    lambda *_, name=name: totals.__setitem__(
                        name, totals[name] + time.perf_counter() - started[name]
                    )
                ),
            ]
    try:
        for px in pixels:
            model.get_image_features(px)
    finally:
        for hook in hooks:
            hook.remove()
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--profile-images", type=int, default=10, help="0 skips the profile"
    )
    args = ap.parse_args()

    import transformers
    from transformers import AutoProcessor

    from vflorence import LLM

    transformers.logging.set_verbosity_error()
    image_processor = AutoProcessor.from_pretrained(args.model).image_processor
    # The tower's cost does not depend on image content, so the COCO set just repeats.
    images = [load_image(url) for url, _ in IMAGE_URLs[: args.images]]
    pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
    pixels = [pixel_values[i % len(images)][None] for i in range(args.images)]

    llm = LLM(args.model, max_num_seqs=2, dtype=getattr(torch, args.dtype))
    runner = llm.runner

    with torch.inference_mode(), runner._inductor_ctx():  # the engine's mode and graph
        for px in pixels[: args.warmup]:
            embeds = runner.image_fn(px)
        assert embeds.shape[:2] == (1, runner.config.image_seq_length), embeds.shape
        times = time_each(runner.image_fn, pixels)
        eager = profile_stages(runner.model, pixels[: args.profile_images])

    ms = sorted(t * 1e3 for t in times)
    print(
        f"\nmodel={args.model}  dtype={args.dtype}  threads={torch.get_num_threads()}  "
        f"image={tuple(pixel_values.shape[1:])}  tokens/image={embeds.shape[1]}\n"
    )
    print(
        f"{args.images} images: {sum(times):.2f} s  ({args.images / sum(times):.1f} images/s)"
    )
    print(
        f"per image: mean {statistics.mean(ms):.1f} ms  median {statistics.median(ms):.1f}  "
        f"p90 {ms[int(0.9 * (len(ms) - 1))]:.1f}  min {ms[0]:.1f}  max {ms[-1]:.1f}"
    )
    if eager:
        total = sum(eager.values())
        print(
            f"\neager profile ({args.profile_images} images, {total / args.profile_images * 1e3:.1f} ms/image)"
        )
        print(f"{'part':<24}{'ms/image':>10}{'share':>8}")
        for name, seconds in eager.items():
            print(
                f"{name:<24}{seconds / args.profile_images * 1e3:>10.2f}{seconds / total:>8.1%}"
            )


if __name__ == "__main__":
    main()
