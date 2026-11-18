"""Why is a batched vision-tower call slower per image? Compare batch sizes op by op.

For each batch size: the compiled tower's torch.profiler op table (self CPU time per
image) and the eager per-stage split. Scratch profiling script.

  PYTHONPATH=. python benchmarks/profile_tower_batch.py --batch-sizes 1,2
"""

import argparse
import re
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.continuous_batching import IMAGE_URLs
from benchmarks.image_encode import profile_stages
from benchmarks.utils import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="florence-community/Florence-2-base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--batch-sizes", default="1,2")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    import transformers
    from transformers import AutoProcessor

    from vflorence import LLM

    transformers.logging.set_verbosity_error()
    torch._inductor.config.cpp.enable_kernel_profile = (
        True  # name each cpp_fused kernel
    )
    proc = AutoProcessor.from_pretrained(args.model).image_processor
    images = [load_image(u) for u, _ in IMAGE_URLs[: args.images]]
    px = proc(images, return_tensors="pt")["pixel_values"]
    runner = LLM(args.model, max_num_seqs=2, dtype=getattr(torch, args.dtype)).runner
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    ops, stages = {}, {}
    with torch.inference_mode(), runner._inductor_ctx():
        for bs in batch_sizes:
            batches = [px[i : i + bs] for i in range(0, args.images, bs)]
            runner.image_fn(batches[0])
            runner.image_fn(batches[0])
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for b in batches:
                    runner.image_fn(b)
            ops[bs] = defaultdict(float)
            for e in prof.key_averages(group_by_input_shape=True):
                key = re.sub(
                    r"^graph_\d+_", "", e.key
                )  # same kernel across batch graphs
                if e.key.startswith("mkldnn::"):  # per-image rows -> which stage it is
                    shape = e.input_shapes[0] if e.input_shapes else []
                    key += f" {[shape[0] // bs] + shape[1:] if shape else ''}"
                ops[bs][key] += e.self_cpu_time_total / 1e3 / args.images  # ms/image
            stages[bs] = {
                k: v * 1e3 / args.images
                for k, v in profile_stages(runner.model, batches).items()
            }

    print(f"\nthreads={torch.get_num_threads()}  images={args.images}")
    print("\ncompiled tower, self CPU ms/image")
    keys = sorted(
        ops[batch_sizes[0]], key=lambda k: -max(o.get(k, 0) for o in ops.values())
    )
    print(f"{'op':<60}" + "".join(f"{'bs' + str(b):>10}" for b in batch_sizes))
    for k in keys[: args.top]:
        print(
            f"{k[:59]:<60}" + "".join(f"{ops[b].get(k, 0):>10.2f}" for b in batch_sizes)
        )
    print(
        f"{'TOTAL':<60}"
        + "".join(f"{sum(ops[b].values()):>10.2f}" for b in batch_sizes)
    )

    print("\neager tower, ms/image per stage")
    print(f"{'part':<28}" + "".join(f"{'bs' + str(b):>10}" for b in batch_sizes))
    for k in stages[batch_sizes[0]]:
        print(f"{k:<28}" + "".join(f"{stages[b][k]:>10.2f}" for b in batch_sizes))
    print(
        f"{'TOTAL':<28}"
        + "".join(f"{sum(stages[b].values()):>10.2f}" for b in batch_sizes)
    )


if __name__ == "__main__":
    main()
