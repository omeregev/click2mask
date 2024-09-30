"""Similarity tests as done in the CLick2Mask paper https://arxiv.org/abs/2409.08272.

Usage:
    python clip_similarity_tests.py --path PATH [--edited_alpha_clip_outs OUTPUT_PATH]

    Arguments:
        path: A path to a directory which should contain:

            1. A sub-directory for each item we wish to compare the methods on.
            Each sub-directory should be named as the item's index, and include the files:
            img.png  - The *input* file from Emu Edit benchmark
            emu.png  - Emu Edit output
            mb.png   - MagicBrush output
            ours.png - Our output

            2. A json named "captions_and_prompts.json", with the following structure:
            {
                "<item_index>": {
                        "input_caption": "<Input_caption from Emu Edit benchmark>.",
                        "output_caption": "<Output_caption from Emu Edit benchmark>"
                        "prompt": "<Un-localized instruction from Emu Edit benchmark:
                                    instruction from Emu Edit benchmark
                                    without the word indicating addition ('add', 'insert', etc.),
                                    and without the location to be edited>"
                },
                "<another item_index>": {
                        ...
                },
                ...
                }
            }

        edited_alpha_clip_outs:
            Optional output path for visualization of Edited Alpha-CLIP mask extractions.

See further explanations in methods below.
"""

import argparse
import torch
import os
import json
from similarities import SimTests
from edited_alpha_clip import EditedAlphaCLip


join = os.path.join


# Edited Alpha-CLIP similarity
def calc_edited_alpha_clip_sim(edited_alpha_clip, path, methods, texts, save_outs=None):
    print("\nCalculating Edited Alpha-CLIP similarities...")
    res_edited_alpha_clip = {m: {} for m in methods}

    dirs = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
    save_to = None
    for method in methods:
        for d in dirs:
            image_in_p = join(path, d, "img.png")
            image_out_p = join(path, d, f"{method}.png")
            prompt = texts[d]["prompt"]

            if save_outs:
                os.makedirs(join(save_outs, d), exist_ok=True)
                save_to = join(save_outs, d, method)
            changed_sim_out = edited_alpha_clip.edited_alpha_clip_sim(
                image_in_p, image_out_p, prompt, save_outs=save_to
            )
            res_edited_alpha_clip[method][d] = changed_sim_out

    print(f'{"*" * 4}\nEdited Alpha-CLIP similarities: (higher is better)')
    for method in methods:
        print(
            f"{method}: {torch.cat(list(res_edited_alpha_clip[method].values())).mean()}"
        )
    print(f'{"*" * 4}')
    if save_outs:
        print(f"Extracted masks saved to {save_outs}")
    print("\n")


# CLIP similarity
def calc_clip_sim(sim_tests, path, methods, texts):
    print("Calculating CLIP similarities...")
    res_clip_out = {m: {} for m in methods}
    res_clip_direction = {m: {} for m in methods}

    dirs = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
    for method in methods:
        for d in dirs:
            image_in = sim_tests.read_image(join(path, d, "img.png"))
            image_out = sim_tests.read_image(join(path, d, f"{method}.png"))
            text_in = texts[d]["input_caption"]
            text_out = texts[d]["output_caption"]

            sim_out, sim_direction = sim_tests.clip_sim(
                image_in=image_in,
                image_out=image_out,
                text_in=text_in,
                text_out=text_out,
            )
            res_clip_out[method][d] = sim_out
            res_clip_direction[method][d] = sim_direction

    print(f'{"*" * 4}\nCLIP output similarities: (higher is better)')
    for method in methods:
        print(f"{method}: {torch.cat(list(res_clip_out[method].values())).mean()}")
    print(f'{"*" * 4}\n')

    print(f'{"*" * 4}\nDirectional CLIP similarities: (higher is better)')
    for method in methods:
        print(
            f"{method}: {torch.cat(list(res_clip_direction[method].values())).mean()}"
        )
    print(f'{"*" * 4}\n\n')


# L1 distance
def calc_l1(sim_tests, path, methods):
    print("Calculating L1 distances...")
    res_L1 = {m: {} for m in methods}

    dirs = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
    for method in methods:
        for d in dirs:
            image_in = sim_tests.read_image(
                join(path, d, "img.png"), dest_size=(512, 512)
            )
            image_out = sim_tests.read_image(
                join(path, d, f"{method}.png"), dest_size=(512, 512)
            )
            res_L1[method][d] = sim_tests.L1_dist(
                image_in=image_in, image_out=image_out
            )

    print(f'{"*" * 4}\nL1 distances: (lower is better)')
    for method in methods:
        print(f"{method}: {torch.cat(list(res_L1[method].values())).mean()}")
    print(f'{"*" * 4}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Your path as explained above")
    parser.add_argument(
        "--edited_alpha_clip_outs",
        help="Optional output path for visualization of Edited Alpha-CLIP mask extractions.",
    )
    args = parser.parse_args()
    path = args.path
    edited_alpha_clip_outs = args.edited_alpha_clip_outs

    methods = ("emu", "mb", "ours")

    # A json with the dictionary explained above
    with open(join(path, "captions_and_prompts.json"), "r") as f:
        texts = json.load(f)

    edited_alpha_clip = EditedAlphaCLip()
    calc_edited_alpha_clip_sim(
        edited_alpha_clip, path, methods, texts, save_outs=edited_alpha_clip_outs
    )

    sim_tests = SimTests()
    calc_clip_sim(sim_tests, path, methods, texts)
    calc_l1(sim_tests, path, methods)
