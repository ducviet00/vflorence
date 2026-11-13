"""Florence-2 object detection on a real image (DaViT + paged BART decoder)."""

import logging
import time

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from vflorence import LLM, Request, SamplingParams

MODEL = "florence-community/Florence-2-base"
URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "transformers/tasks/car.jpg?download=true"
)
TASK = "<OD>"


def main():
    image = Image.open(requests.get(URL, stream=True).raw).convert("RGB")
    processor = AutoProcessor.from_pretrained(MODEL)
    llm = LLM(MODEL, dtype=torch.bfloat16, max_decoder_len=1024)

    inputs = processor(text=TASK, images=image, return_tensors="pt")
    request = Request(inputs["input_ids"][0].tolist(), inputs["pixel_values"][0])

    start = time.perf_counter()
    outputs = llm.generate(request, SamplingParams(max_tokens=1024, num_beams=3))
    text = processor.tokenizer.decode(outputs[0].completion_token_ids)
    print(f"{time.perf_counter() - start:.2f}s  {text}")

    parsed = processor.post_process_generation(text, task=TASK, image_size=image.size)
    print(parsed)


if __name__ == "__main__":
    logging.basicConfig(format="%(name)s: %(message)s")
    logging.getLogger("vflorence").setLevel(logging.INFO)  # memory usage
    main()
