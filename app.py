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
example_point = [[308, 285]]


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


def mark_last_click(image_prompter, example=True):
    click_draw = ClickDraw()
    if not example:
        point512 = extract_last_point512(image_prompter)
    else:
        point512 = np.array(example_point[-1]).astype(int)[::-1]

    _, image_prompter["image"] = click_draw(image_prompter["image"],
                                            point512=point512)
    return image_prompter


def assert_inputs_and_mark_last_click(image_prompter, text):
    if image_prompter is None:
        raise gr.Error("Please upload an image")
    if not text or text.strip() == "":
        raise gr.Error("Please provide a text prompt")
    if not image_prompter["points"] or len(image_prompter["points"]) == 0:
        raise gr.Error("Please click on the image to indicate the edit area")

    return mark_last_click(image_prompter, example=False)


def generate(image_prompter, text):
    point512 = extract_last_point512(image_prompter)
    return click2mask_app(text, image_prompter["image"], point512)


def load_example_inputs():
    example_prompt = random.choice(example_prompts)
    example_image_clicked = Image.open("examples/gradio/img3_clicked.png")
    clicked_image_dict = {
        "image": example_image_clicked,
    }
    return clicked_image_dict, example_prompt


def generate_example_output(text):
    example_image = Image.open("examples/gradio/img3.jpg")
    input_image_dict = {"image": example_image, "points": example_point}
    assert_inputs_and_mark_last_click(input_image_dict, text)
    return generate(input_image_dict, text)


def clear_fields():
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
        fn=assert_inputs_and_mark_last_click,
        inputs=[image_input, text_input],
        outputs=image_input
    ).then(
        fn=generate,
        inputs=[image_input, text_input],
        outputs=image_output
    )

    example_button.click(
        fn=load_example_inputs,
        inputs=[],
        outputs=[image_input, text_input],
    ).then(
        fn=mark_last_click,
        inputs=image_input,
        outputs=image_input
    ).then(
        fn=generate_example_output,
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
        demo.launch(share=True)
