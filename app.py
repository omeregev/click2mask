from gradio_image_prompter import ImagePrompter
from scripts.text_editing_click2mask import click2mask_app
import warnings
import contextlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
import gradio as gr
import numpy as np
import random
import os
import requests
from tqdm import tqdm
from PIL import Image
from scripts.clicker import ClickDraw
from scripts.constants import Const

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

example_prompts = [
    "A sea monster",
    "A big ship",
    "An iceberg",
]
example_point = [[320, 285]]


class Cache:
    orig_image = None
    point512 = None
    generation_performed = False


def handle_checkpoint(chunk_size=1024 * 1024):
    url = "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth"
    os.makedirs("checkpoints", exist_ok=True)
    destination_path = "checkpoints/clip_l14_336_grit1m_fultune_8xe.pth"

    if os.path.exists(destination_path):
        return

    print(f"Downloading {url} to {destination_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure download is successful

    total_size = int(response.headers.get("content-length", 0))
    progress_step = total_size // 20

    with open(destination_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=progress_step,  # Update only every 5%
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)
                progress_bar.update(len(chunk))

    print("Download completed.")


def extract_last_point512(image_prompter):
    point = np.array(image_prompter["points"][-1][:2]).astype(int)
    point[0] = (point[0] * (Const.W / image_prompter["image"].size[0])).astype(int)  # pil, width is first
    point[1] = (point[1] * (Const.H / image_prompter["image"].size[1])).astype(int)
    point = point[::-1]

    return point


def assert_inputs(image_prompter, text, example):
    ACTION = "<br><br>Refresh or click 'Clear' to continue"

    if Cache.generation_performed:
        raise gr.Error("Please upload a new image before generating again." + ACTION)
    if image_prompter is None:
        raise gr.Error("Please upload an image." + ACTION)
    if not text or text.strip() == "":
        raise gr.Error("Please provide a text prompt." + ACTION)
    if not example and (
        not image_prompter["points"] or len(image_prompter["points"]) == 0
    ):
        raise gr.Error("Please click on the image to indicate the edit area." + ACTION)


def arrange_inputs(image_prompter, text, example=False):
    assert_inputs(image_prompter, text, example)
    click_draw = ClickDraw()
    if not example:
        Cache.point512 = extract_last_point512(image_prompter)
    else:
        Cache.point512 = np.array(example_point[-1]).astype(int)[::-1]

    Cache.orig_image = image_prompter["image"]
    img_size = (Const.W, Const.H)
    if Cache.orig_image.mode != "RGB":
        Cache.orig_image = Cache.orig_image.convert("RGB")
    if Cache.orig_image.size != img_size:
        Cache.orig_image = Cache.orig_image.resize(img_size, Image.LANCZOS)
    _, image_prompter["image"] = click_draw(Cache.orig_image, point512=Cache.point512)

    return image_prompter


def arrange_example_inputs(image_prompter, text):
    return arrange_inputs(image_prompter, text, example=True)


def generate(text):
    Cache.generation_performed = True
    return click2mask_app(text, Cache.orig_image, Cache.point512)


def load_example_inputs():
    Cache.generation_performed = False
    example_image = Image.open("examples/gradio/img3.jpg")
    example_prompt = random.choice(example_prompts)
    image_dict = {
        "image": example_image,
    }
    return image_dict, example_prompt


def clear_fields():
    Cache.generation_performed = False
    return None, "", None


with gr.Blocks(
    theme=gr.themes.Default().set(
        button_primary_background_fill="rgb(164, 190, 237)",
        button_primary_background_fill_hover="rgb(144, 170, 217)",
        button_primary_text_color="black",
        button_secondary_background_fill="rgb(183, 225, 183)",
        button_secondary_background_fill_hover="rgb(163, 205, 163)",
        button_secondary_text_color="black",
    )
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            image_input = ImagePrompter(
                type="pil",
                label="Click on the image to indicate the edit area (last click counts)",
                height=512,
                width=512,
            )
        with gr.Column(scale=1):
            gr.Markdown("### Edited Image")
            image_output = gr.Image(
                show_download_button=True,
                interactive=False,
                height=512,
                width=512,
            )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Describe the object to add...",
                interactive=True,
            )
        with gr.Column(scale=1):
            pass

    with gr.Row():
        with gr.Column(scale=1):
            generate_button = gr.Button(
                "Generate",
                variant="primary",
            )
        with gr.Column(scale=1):
            pass

    with gr.Row():
        gr.Markdown(
            "💡 *Click 'Load Example' for preloaded examples*",
            elem_classes="center-text",
        )

    with gr.Row():
        with gr.Column(scale=1):
            example_button = gr.Button(
                "Load Example",
                variant="secondary",
            )
        with gr.Column(scale=1):
            clear_button = gr.Button(
                "Clear",
                variant="stop",
            )

    generate_button.click(
        fn=arrange_inputs,
        inputs=[image_input, text_input],
        outputs=image_input
    ).success(
        fn=generate,
        inputs=text_input,
        outputs=image_output,
    )

    example_button.click(
        fn=load_example_inputs,
        inputs=[],
        outputs=[image_input, text_input],
    ).success(
        fn=arrange_example_inputs,
        inputs=[image_input, text_input],
        outputs=image_input
    ).success(
        fn=generate,
        inputs=text_input,
        outputs=image_output
    )

    clear_button.click(
        fn=clear_fields,
        inputs=[],
        outputs=[image_input, text_input, image_output],
    )


if __name__ == "__main__":
    handle_checkpoint()
    with contextlib.suppress(DeprecationWarning):
        demo.queue(concurrency_count=1)
        demo.launch(share=False)
