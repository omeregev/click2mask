import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, help="The target text prompt")
    parser.add_argument(
        "--image_path",
        type=str,
        help="The path to the input image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help=(
            "The output dir. Output file will be: "
            "<output_dir>/<image_path without extension>_out.jpg"
        ),
    )
    parser.add_argument(
        "--refresh_click",
        action="store_true",
        help=(
            "If True, will recreate the click on the image. "
            "If False, will create the click if "
            "<image_path without extension>_click.<jpg/JPG/png/PNG> doesn't exist."
        ),
    )
    parser.add_argument(
        "--n_masks",
        type=int,
        default=1,
        help="Number of dynamic masks to generate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="omeregev/sd-2-1-base-mirror",
        help="The path to the HuggingFace model",
    )
    parser.add_argument(
        "--alpha_clip_scale",
        type=int,
        default=336,
        choices=[336, 224],
        help="Set to 224 in case of GPU memory incapacity",
    )
    parser.add_argument("--seed", type=int, help="Optional random seed")
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    return args
