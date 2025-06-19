import os
import random
import requests
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from scripts.constants import Const
from scripts.text_editing_click2mask import click2mask_app
from scripts.clicker import ClickDraw

# -----------------------------------------------------------------------------#
# Download model checkpoint once                                               #
# -----------------------------------------------------------------------------#
CKPT_URL = (
    "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/"
    "clip_l14_336_grit1m_fultune_8xe.pth"
)
CKPT_PATH = Path("checkpoints") / "clip_l14_336_grit1m_fultune_8xe.pth"


def download_checkpoint() -> None:
    """Download the model weights if they are not already on disk."""
    if CKPT_PATH.exists():
        print("Checkpoint already exists, skipping download.")
        return

    print("Downloading model checkpoint …")
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(CKPT_URL, stream=True) as r:
        r.raise_for_status()
        with CKPT_PATH.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download completed.")


# -----------------------------------------------------------------------------#
# Helper functions                                                             #
# -----------------------------------------------------------------------------#
example_prompts = ["A sea monster", "A big ship", "An iceberg"]
example_image_path = "examples/gradio/img3.jpg"
example_point = [320, 285]  # x, y


def resize_to_512(image: Image.Image) -> Image.Image:
    """Convert to RGB and resize to 512×512 if needed."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (Const.W, Const.H):
        image = image.resize((Const.W, Const.H), Image.LANCZOS)
    return image


def extract_point512(xy, img_size):
    """Convert click (x, y) in UI coords → (row, col) in 512×512 space."""
    x, y = xy
    scale_x = Const.W / img_size[0]
    scale_y = Const.H / img_size[1]
    return int(y * scale_y), int(x * scale_x)  # (row, col)


# -----------------------------------------------------------------------------#
# Gradio UI                                                                    #
# -----------------------------------------------------------------------------#
CSS = """
.btn-generate { background-color: #b2f2bb !important; color: black !important; }
.btn-clear    { background-color: #ffc9c9 !important; color: black !important; }
.btn-example  { background-color: #dee2e6 !important; color: black !important; }
.centered     { text-align: center; }
"""

with gr.Blocks(css=CSS) as demo:
    # ---------- per-session state ----------
    orig_image_state = gr.State()   # PIL.Image or None
    point_state = gr.State()        # tuple(row, col) or None

    # ---------- layout ----------
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload and Click on Image", elem_classes="centered")
            image = gr.Image(type="pil", height=512, width=512, label="Input")
        with gr.Column(scale=1):
            gr.Markdown("### Generated Image", elem_classes="centered")
            output = gr.Image(
                height=512, width=512, label="Output", show_download_button=True
            )

    prompt = gr.Textbox(label="What object to add?", placeholder="e.g. a sea monster")

    with gr.Row():
        gen_btn = gr.Button("Generate", elem_classes="btn-generate")
        ex_btn = gr.Button("Load Example", elem_classes="btn-example")
        clr_btn = gr.Button("Reset", elem_classes="btn-clear")

    # ---------- callbacks ----------
    def on_upload(image):
        """Handle file drop / selection."""
        image = resize_to_512(image)
        return image, image, None  # show img, store orig_image_state, clear point

    def on_click(image, evt: gr.SelectData, orig_image):
        """Handle a click on the input image."""
        if orig_image is None:
            raise gr.Error("Upload an image first.")
        point512 = extract_point512(evt.index, image.size)
        _, img_marked = ClickDraw()(orig_image, point512=point512)
        return img_marked, orig_image, point512

    def generate_image(prompt_txt, orig_image, point512):
        """Generate new image given prompt and point."""
        if not prompt_txt or prompt_txt.strip() == "":
            raise gr.Error("Please enter a prompt.")
        if orig_image is None or point512 is None:
            raise gr.Error("Upload and click on an image first.")
        return click2mask_app(prompt_txt, orig_image, point512)

    def clear_all():
        """Reset UI and session state."""
        return None, "", None, None  # clear image, prompt, orig_state, point_state

    def load_example():
        """Load predefined example."""
        img = resize_to_512(Image.open(example_image_path))
        point512 = np.array(example_point[::-1])  # (y, x)
        _, img_marked = ClickDraw()(img, point512=point512)
        example_prompt = random.choice(example_prompts)
        return img_marked, example_prompt, img, point512

    # ---------- wiring ----------
    image.upload(
        fn=on_upload,
        inputs=image,
        outputs=[image, orig_image_state, point_state],
    )
    image.select(
        fn=on_click,
        inputs=[image, orig_image_state],
        outputs=[image, orig_image_state, point_state],
    )
    gen_btn.click(
        fn=generate_image,
        inputs=[prompt, orig_image_state, point_state],
        outputs=output,
    )
    ex_btn.click(
        fn=load_example,
        inputs=[],
        outputs=[image, prompt, orig_image_state, point_state],
    )
    clr_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image, prompt, orig_image_state, point_state],
    )

# -----------------------------------------------------------------------------#
# Run                                                                          #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    download_checkpoint()
    demo.queue()
    demo.launch(share=True)
