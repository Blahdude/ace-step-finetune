#!/usr/bin/env python3

import argparse
import json
from math import sqrt

import safetensors.torch
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--lora_config_path", type=str)
    args = parser.parse_args()

    with open(args.lora_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    rank = config["r"]
    alpha = config["lora_alpha"]
    if config["use_rslora"]:
        alpha /= sqrt(rank)
    else:
        alpha /= rank

    # See https://github.com/comfyanonymous/ComfyUI/blob/856448060ce42674eea66c835bd754644c322723/comfy/weight_adapter/lora.py#L106
    comfy_alpha = alpha * rank
    print("comfy_alpha", comfy_alpha)

    tensors = safetensors.torch.load_file(args.input_name)
    # Copy a list of keys to avoid modifying the iterator
    ks = list(tensors.keys())
    for k in ks:
        if not k.endswith(".lora_A.weight"):
            continue
        k_alpha = k.replace(".lora_A.weight", ".alpha")
        if k_alpha in ks:
            tensors[k_alpha] *= comfy_alpha
        else:
            dtype = tensors[k].dtype
            tensors[k_alpha] = torch.tensor(comfy_alpha, dtype=dtype)
    safetensors.torch.save_file(tensors, args.output_name)


if __name__ == "__main__":
    main()
