import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="PyTorch version 1.7.1 or higher is recommended")
import clip


join = os.path.join


class SimTests:
    def __init__(self, clip_model_name="ViT-B/16", ac_scale=336, device="cuda:0"):
        self.device = device
        self.clip_size = 224
        self.clip_model_name = clip_model_name
        self.clip_model = (
            clip.load(self.clip_model_name, device=self.device, jit=False)[0]
            .eval()
            .requires_grad_(False)
        )

    def read_image(self, img_path, dest_size=None):
        image = Image.open(img_path).convert("RGB")
        if (dest_size is not None) and (image.size != dest_size):
            image = image.resize(dest_size, Image.LANCZOS)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(self.device)
        return image

    def assert_properties(self, image_in, image_out):
        for image in (image_in, image_out):
            assert torch.is_tensor(image)
            assert image.min() >= 0 and image.max() <= 1

    # CLIP similarity and Directional CLIP similarity (higher is better)
    @torch.no_grad()
    def clip_sim(self, image_in, image_out, text_in, text_out):
        """Calculates:
        1. CLIP similarity between output image and output caption
        2. Directional CLIP similarity of the change between input and output images,
           with the change between input and output captions
        """
        self.assert_properties(image_in, image_out)
        text_in = [text_in]
        text_out = [text_out]
        image_features_in = self.encode_image(image_in)
        image_features_out = self.encode_image(image_out)
        text_features_in = self.encode_text(text_in)
        text_features_out = self.encode_text(text_out)
        sim_out = F.cosine_similarity(image_features_out, text_features_out)
        sim_direction = F.cosine_similarity(
            image_features_out - image_features_in, text_features_out - text_features_in
        )
        return sim_out, sim_direction

    def encode_text(self, prompt):
        assert type(prompt) in (list, tuple)
        text = clip.tokenize(prompt).to(self.device)
        text_features = self.clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image):
        image_transform = transforms.Compose(
            [
                transforms.Resize(self.clip_size, interpolation=Image.BICUBIC),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        image = image_transform(image).half()
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    # L1 distance (lower is better)
    def L1_dist(self, image_in, image_out):
        """
        Mean L1 pixel distance between input and output images, to measure the
        amount of change in the entire image
        """
        self.assert_properties(image_in, image_out)
        return F.l1_loss(image_in, image_out).unsqueeze(0)
