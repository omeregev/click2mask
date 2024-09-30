"""
"Edited Alpha-CLIP", as proposed in the CLick2Mask paper https://arxiv.org/abs/2409.08272.
Evaluates the similarity between the masked edited region, and the un-localized prompt
(prompt without the word indicating addition ('add', 'insert', etc.),
and without the location to be edited.
A mask indicating the edit made is extracted automatically,
and a similarity is calculated between the masked output and the un-localized prompt, using Alpha-CLIP.

Can optionally output the image with the extracted masks overlayed.
"""
import os

import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
from einops import rearrange
import warnings
warnings.filterwarnings("ignore", message="PyTorch version 1.7.1 or higher is recommended")
import alpha_clip

DEST_SIZE = (512, 512)


def make_overlay(im, mask, alpha=1.0, beta=0.5):
    mask[:, :, 0] = 0
    mask[:, :, 2] = 0
    ret = cv2.addWeighted(im, alpha, mask, beta, 0)
    ret = np.clip(ret, a_min=0, a_max=1)
    return ret


class EditedAlphaCLip:
    def __init__(self, ac_scale=336, device="cuda:0"):
        assert ac_scale in (224, 336)
        self.device = device
        self.ac_size = (ac_scale, ac_scale)
        if self.ac_size == (336, 336):
            self.ac_model, self.ac_preprocess = alpha_clip.load(
                "ViT-L/14@336px",
                alpha_vision_ckpt_pth="./checkpoints/clip_l14_336_grit1m_fultune_8xe.pth",
                device=self.device,
            )
        else:
            self.ac_model, self.ac_preprocess = alpha_clip.load(
                "ViT-L/14",
                alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit20m_fultune_2xe.pth",
                device=self.device,
            )

        self.im_to_sqz32 = lambda x: rearrange(
            x.cpu().numpy().squeeze().astype(np.float32), "c h w -> h w c"
        )
        self.im_to_cat_32 = lambda x: rearrange(
            torch.stack([x.squeeze(0).squeeze(0)] * 3, dim=0)
            .cpu()
            .numpy()
            .astype(np.float32),
            "c h w -> h w c",
        )

    def save_im(self, im, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray((im * 255).round().astype("uint8")).save(path, quality=95)

    def read_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if image.size != DEST_SIZE:
            image = image.resize(DEST_SIZE, Image.LANCZOS)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(self.device)
        return image

    # Edited Alpha-CLIP (higher is better)
    @torch.no_grad()
    def edited_alpha_clip_sim(self, image_in_p, image_out_p, prompt, save_outs=None):
        """
        Args:
            image_in_p: The input image path
            image_out_p: The output image path
            prompt: The un-localized prompt (as explained above)
            save_outs: If given, will save:
                * The output image with extracted mask overlayed to <save_outs>_out_masked.jpg,
                * The output image to <save_outs>_out.jpg.
                * The input image to <save_outs>_in.jpg.
            All in size (512, 512).
        """
        assert type(prompt) is str
        prompt = [prompt]
        image_in = self.read_image(image_in_p)
        image_out = self.read_image(image_out_p)

        mask_transform = transforms.Compose(
            [nn.AdaptiveAvgPool2d(self.ac_size), transforms.Normalize(0.5, 0.26)]
        )
        image_transform = transforms.Compose(
            [
                transforms.Resize(self.ac_size, interpolation=Image.BICUBIC),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        mask = self.extract_mask(image_in=image_in, image_out=image_out)
        alpha = mask_transform(mask).half()
        image = image_transform(image_out).half()
        image_features = self.ac_model.visual(image, alpha)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text = alpha_clip.tokenize(prompt).to(self.device)
        text_features = self.ac_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        alpha_loss = image_features @ text_features.T

        alpha_loss = alpha_loss.mean(dim=0)

        if save_outs:
            self.save_im(
                make_overlay(self.im_to_sqz32(image_out), self.im_to_cat_32(mask)),
                f"{save_outs}_out_masked.jpg",
            )
            self.save_im(self.im_to_sqz32(image_out), f"{save_outs}_out.jpg")
            self.save_im(self.im_to_sqz32(image_in), f"{save_outs}_in.jpg")

        return alpha_loss

    def create_multiple_convex_hulls(self, binary_mask, min_hull_area=100):
        if binary_mask.is_cuda:
            binary_mask = binary_mask.cpu()
        np_mask = binary_mask.squeeze().numpy().astype(np.uint8)

        contours, _ = cv2.findContours(
            np_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return binary_mask

        all_hulls_mask = np.zeros_like(np_mask)
        for contour in contours:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area >= min_hull_area:
                cv2.drawContours(all_hulls_mask, [hull], 0, 1, -1)

        hull_tensor = torch.from_numpy(all_hulls_mask).unsqueeze(0).unsqueeze(0).half()
        hull_tensor = hull_tensor.to(self.device)

        return hull_tensor

    def extract_mask(self, image_in, image_out):
        mask = (torch.mean(torch.abs(image_in - image_out), dim=1) > 0.1).half()
        pool_for_min = nn.MaxPool2d(3, stride=1, padding=1)
        mask = -pool_for_min(-mask)
        pool_for_max = nn.MaxPool2d(5, stride=1, padding=2)
        mask = pool_for_max(mask)
        mask = self.create_multiple_convex_hulls(mask)

        return mask
