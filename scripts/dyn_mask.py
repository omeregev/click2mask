import torch
from torchvision import transforms
import numpy as np
import skfmm
from PIL import Image
import torch.nn as nn
import cv2
import scipy
from scipy.ndimage.filters import gaussian_filter
import kornia
import warnings
warnings.filterwarnings("ignore", message="PyTorch version 1.7.1 or higher is recommended")
import alpha_clip
from augmentations import ImageAugmentations
from constants import Const, N


@torch.no_grad()
def get_dist_field(dist_from, device, as_squeezed_np=False):
    if not isinstance(dist_from, np.ndarray):
        dist_from = dist_from.cpu().numpy()
    assert np.max(dist_from) <= 1
    dist_from = -(np.where(dist_from, 0, -1) + 0.5)
    dist_field = skfmm.distance(dist_from, dx=1)
    if as_squeezed_np:
        return dist_field
    return torch.tensor(dist_field).to(device)


def get_surround(surround_from, surround_width, device, as_squeezed_np=False):
    dists = get_dist_field(surround_from, device)
    surround = (dists <= surround_width).to(surround_from.dtype)
    if as_squeezed_np:
        return surround.cpu().numpy()
    return surround


class DynMask:

    def __init__(self, click_pil, args, init_image_tensor, device, total_steps):
        self.args = args
        self.device = device
        self.init_image = init_image_tensor
        self.total_steps = total_steps

        self.ac_size = (self.args.alpha_clip_scale, self.args.alpha_clip_scale)
        if self.args.alpha_clip_scale == 336:
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

        self.image_augmentations = ImageAugmentations(
            self.args.alpha_clip_scale, Const.AUG_NUM
        )
        self.text_features = self.get_text_features([self.args.prompt])

        self.latent_size = Const.LATENT_SIZE
        self.decoded_size = (Const.H, Const.W)
        self.thresh_val = Const.THRESH_VAL
        self.base_potential = None
        self.potential = None
        self.latent_mask = None
        self.set_init_masks(click_pil)

        self.cached_masks_clones = {}
        self.closs_hist = {}
        self.latents_hist = {}
        self.latent_masks_hist = {}

    @torch.no_grad()
    def normalize_point_size(self, click, radius_for64=1.367):
        threshed = (click > 0.5).astype(float)
        x, y = np.where(threshed)
        center = int(x.mean().round()), int(y.mean().round())
        norm_threshed = np.zeros_like(threshed)
        norm_threshed[center[0], center[1]] = 1
        norm_threshed = get_surround(
            torch.tensor(norm_threshed).to(self.device),
            click.shape[0] / 64 * radius_for64 - 0.3,
            self.device,
            as_squeezed_np=True,
        )

        return norm_threshed

    @torch.no_grad()
    def calc_potential(self, click_pil, sigma_for_shape64):
        dest_size = self.latent_size
        click = click_pil.convert("L").resize(dest_size, Image.NEAREST)
        click = (np.array(click) > 125).astype(float)
        click = self.normalize_point_size(
            click, radius_for64=Const.POINT_ON_LATENT_RADIUS
        )
        potential = gaussian_filter(
            click, sigma=sigma_for_shape64 * (click.shape[0]) / 64
        )
        potential = (potential - np.min(potential)) / max(
            np.max(potential) - np.min(potential), 1e-8
        )
        potential = potential[np.newaxis, np.newaxis, ...]
        potential = torch.from_numpy(potential).half().to(self.device)

        return potential

    @torch.no_grad()
    def set_init_masks(self, click_pil, stretch_factor=1.0):
        potential = self.calc_potential(
            click_pil, sigma_for_shape64=Const.SIGMA_FOR_SHAPE64
        )
        self.base_potential = potential.detach().to(torch.float64)
        if self.base_potential.ndim == 2:
            self.base_potential = self.base_potential.unsqueeze(0).unsqueeze(0)
        self.base_potential = self.base_potential * (Const.POTENTIAL_PEAK - (-1)) - 1
        self.base_potential = stretch_factor * self.base_potential

        self.set_cur_masks(step_i=0)

    @torch.no_grad()
    def set_cur_masks(
        self, step_i, grads_to_update=None, surround_ring=None, return_only=None
    ):
        potential = self.base_potential + self.get_bias(step_i)

        if grads_to_update is not None:
            potential = potential + (surround_ring * Const.MASK_LR * grads_to_update)
            potential = transforms.GaussianBlur(
                Const.GAUSS_K_MASK, sigma=Const.GAUSS_SIGMA_MASK
            )(potential)

        if torch.all(potential <= 0):
            potential += Const.ADDITION_IN_COLLAPSE
            print(
                f"{'*' * 10} Mask shrunk entirely, added {Const.ADDITION_IN_COLLAPSE}"
            )
        elif torch.all(potential >= 0):
            potential -= Const.ADDITION_IN_COLLAPSE
            print(
                f"{'*' * 10} Mask expanded entirely, reduced {Const.ADDITION_IN_COLLAPSE}"
            )

        self.potential = potential.half()
        self.latent_mask = self.get_threshed_mask(self.potential)

        return self.get_curr_masks(return_only=return_only)

    @torch.no_grad()
    def get_curr_masks(self, return_only=None):
        if return_only is not None:
            if return_only == N.POTENTIAL:
                return self.potential
            elif return_only == N.LATENT_MASK:
                return self.latent_mask
            else:
                raise ValueError(f"return_only should be in ('{N.POTENTIAL}', '{N.LATENT_MASK}')")

        return self.potential, self.latent_mask

    @torch.no_grad()
    def make_cached_masks_clones(self, name):
        self.cached_masks_clones[name] = {
            N.POTENTIAL: self.potential.detach().clone(),
            N.LATENT_MASK: self.latent_mask.detach().clone(),
        }

    @torch.no_grad()
    def set_masks_from_cached_masks_clones(self, name):
        self.potential = self.cached_masks_clones[name][N.POTENTIAL]
        self.latent_mask = self.cached_masks_clones[name][N.LATENT_MASK]

    @torch.no_grad()
    def evolve_mask(
        self, step_i, decoder, latent_pred_z0, source_latents, return_only=None
    ):

        potential, latent_mask = self.get_curr_masks()
        surround_ring = self.get_ring(latent_mask)
        grads_latent = self.calc_grads(
            latent_pred_z0=latent_pred_z0,
            source_latents=source_latents,
            potential=potential,
            step_i=step_i,
            decoder=decoder,
        )
        grads_latent = torch.abs(grads_latent)
        grads_latent = transforms.GaussianBlur(
            Const.GAUSS_K_GRADS, sigma=Const.GAUSS_SIGMA_GRADS
        )(grads_latent)

        grads_latent = (grads_latent - grads_latent.mean()) / max(
            grads_latent.std(), 1e-6
        )
        grads_latent = torch.maximum(grads_latent, torch.tensor(0.0).to(self.device))

        self.set_cur_masks(
            step_i=step_i, grads_to_update=grads_latent, surround_ring=surround_ring
        )

        return self.get_curr_masks(return_only=return_only)

    def calc_grads(self, latent_pred_z0, source_latents, potential, step_i, decoder):
        with torch.enable_grad():
            latent_mask = self.get_threshed_mask(potential)
            latent_mask = latent_mask.detach().requires_grad_()

            blend_predz0_origz0 = latent_pred_z0 * latent_mask + (
                source_latents * (1 - latent_mask)
            )

            scaled_blend_pred_z0_origz0 = 1 / 0.18215 * blend_predz0_origz0
            decoded_blend_predz0_origz0 = decoder(
                scaled_blend_pred_z0_origz0
            ).sample.to(torch.float32)

            alpha_mask = transforms.Resize(self.decoded_size, interpolation=0)(
                latent_mask
            )
            alpha_mask = (alpha_mask > 0.5).half().clone().detach()
            alpha_mask = get_surround(
                alpha_mask,
                Const.ALPHA_MASK_DILATION_ON_512 * (Const.HW / 512.0),
                self.device,
            )

            alpha_loss = self.alpha_clip_loss(
                decoded_blend_predz0_origz0,
                alpha_mask,
                self.text_features,
                self.image_augmentations,
                augs_with_orig=True,
            )

            self.closs_hist[
                step_i - 1
            ] = alpha_loss.detach()  # The mask used for the loss is prev step mask

            grads_latent = torch.autograd.grad(alpha_loss, latent_mask)[0].to(
                torch.float64
            )

            return grads_latent.detach()

    def alpha_clip_loss(
        self,
        image,
        mask,
        text_features,
        image_augmentations,
        augs_with_orig=True,
        return_as_similarity=False,
    ):
        """
        image and mask in range 0.0 to 1.0
        """
        assert mask.min() >= 0 and mask.max() <= 1

        mask_transform = transforms.Compose(
            [nn.AdaptiveAvgPool2d(self.ac_size), transforms.Normalize(0.5, 0.26)]
        )
        mask_normalize = transforms.Normalize(0.5, 0.26)

        image_transform = transforms.Compose(
            [
                transforms.Resize(self.ac_size, interpolation=Image.BICUBIC),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        image_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        image = image.add(1).div(2)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        alpha = mask
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(dim=0)

        if image_augmentations is not None:
            image, alpha = image_augmentations(image, alpha, with_orig=augs_with_orig)
            image = image_normalize(image).half()
            alpha = mask_normalize(alpha).half()
        else:
            image = image_transform(image).half()
            alpha = mask_transform(alpha).half()

        image_features = self.ac_model.visual(image, alpha)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if return_as_similarity:
            alpha_loss = image_features @ text_features.T
        else:
            alpha_loss = 1 - image_features @ text_features.T
        alpha_loss = alpha_loss.mean(dim=0)

        return alpha_loss

    def get_text_features(self, prompt):
        assert type(prompt) in (list, tuple)
        text = alpha_clip.tokenize(prompt).to(self.device)
        text_features = self.ac_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_bias(self, step_i):
        bias = Const.BIAS_DILATION_VAL * (Const.BIAS_DILATION_DEC_FACTOR**step_i)
        while torch.all(self.base_potential + bias > 0) and bias > 1e-8:
            bias *= 0.9

        return bias

    def get_threshed_mask(self, potential):
        thresh_val = self.thresh_val
        t_m = (potential > thresh_val).half()

        t_m = t_m.cpu().numpy().squeeze().astype(np.uint8)
        t_m = scipy.ndimage.binary_fill_holes(t_m)
        t_m = torch.tensor(t_m).to(self.device).unsqueeze(0).unsqueeze(0).half()
        t_m = self.close_gaps_with_connection(
            t_m, thickness=Const.CLOSE_GAPS_WITH_CONNECTION_THICKNESS
        )

        t_m = kornia.morphology.closing(
            t_m, torch.ones(Const.CLOSING_K, Const.CLOSING_K).to(self.device)
        )
        t_m = t_m.cpu().numpy().squeeze().astype(np.uint8)
        t_m = scipy.ndimage.binary_fill_holes(t_m)
        t_m = torch.tensor(t_m).to(self.device).unsqueeze(0).unsqueeze(0).half()

        t_m = transforms.GaussianBlur(
            Const.GAUSS_K_THRESHED, sigma=Const.GAUSS_SIGMA_THRESHED
        )(t_m)
        t_m = (t_m > Const.THRESH_POST_GAUSS).half()

        return t_m

    @torch.no_grad()
    def close_gaps_with_connection(self, threshed_mask, thickness):
        # also cleans small contours
        given_threshed_mask = threshed_mask
        threshed_mask = threshed_mask.cpu().numpy().squeeze().astype(np.uint8)

        connected_mask = threshed_mask * 0
        contours, hierarchy = cv2.findContours(
            threshed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 1:
            return given_threshed_mask

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        contours = [
            cnt
            for cnt in contours
            if cv2.contourArea(cnt)
            > threshed_mask.shape[-1] * threshed_mask.shape[-2] * 0.001
        ]

        cv2.drawContours(connected_mask, contours, 0, 255, -1)
        for i in range(1, len(contours)):
            cv2.drawContours(connected_mask, contours, i, 255, -1)
            hull = cv2.convexHull(contours[i])  # Convex hull of contour
            hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            connect = hull.copy()
            for hp in hull:
                dists = np.linalg.norm(contours[0] - hp, axis=2).squeeze()
                min_points = np.where(dists == dists.min())[0]
                for mp in min_points:
                    connect = np.append(
                        connect, np.expand_dims(contours[0][mp], axis=0), axis=0
                    )
            connected_mask = cv2.drawContours(
                connected_mask, [connect], -1, color=255, thickness=thickness
            )
            connected_mask = cv2.drawContours(
                connected_mask, [connect], -1, color=255, thickness=-1
            )

        connected_mask = (
            ((torch.tensor(connected_mask).to(self.device)) > 125)
            .unsqueeze(0)
            .unsqueeze(0)
            .half()
        )
        return connected_mask

    @torch.no_grad()
    def get_plain_dilated_latent_mask(
        self,
        last_step_latent_mask,
        step_i,
        total_steps,
        max_area_ratio_for_dilation=None,
        rerun_dyn_start_step_i=None,
    ):
        max_area_ratio_for_dilation = (
            Const.MAX_AREA_RATIO_FOR_DILATION
            if max_area_ratio_for_dilation is None
            else max_area_ratio_for_dilation
        )
        if (
            last_step_latent_mask.sum()
            > max_area_ratio_for_dilation * last_step_latent_mask.nelement()
        ):
            return last_step_latent_mask

        first_k = self.latent_size[-1] // 2
        while (
            get_surround(last_step_latent_mask, first_k, self.device).sum()
            > 0.75 * self.latent_size[-1] ** 2
        ):
            first_k -= 1
        if rerun_dyn_start_step_i:
            plain_dilation_ws = np.linspace(
                first_k, 0, rerun_dyn_start_step_i + 2 - Const.RERUN_STOP_DILATION
            ).round()
            plain_dilation_ws = np.pad(
                plain_dilation_ws, (0, total_steps - len(plain_dilation_ws))
            )
        else:
            plain_dilation_ws = np.array(
                [first_k / max(1, (i / 3)) for i in range(0, total_steps)]
            ).round()
            plain_dilation_ws[-10:] = 0

        return get_surround(
            last_step_latent_mask, plain_dilation_ws[step_i], self.device
        ).half()

    @torch.no_grad()
    def get_ring(self, latent_mask):
        assert (latent_mask.min() >= 0) and (latent_mask.max() <= 1)
        out_ring_width = Const.OUT_RING_WIDTH
        in_on_ring_width = Const.IN_ON_RING_WIDTH
        latent_mask = (latent_mask.cpu().numpy() >= 0.5).astype(np.float16)
        dists = get_dist_field(latent_mask, self.device, as_squeezed_np=True)

        in_ring_width = in_on_ring_width - 1
        in_ring = dists.copy()
        in_ring[in_ring > -1] = 0
        in_ring[in_ring <= -in_ring_width - 1] = 0
        in_ring[in_ring != 0] = 1

        on_ring = latent_mask.copy()
        on_ring[dists < -1] = 0

        in_on_ring = in_ring.astype(bool) | on_ring.astype(bool)

        out_ring = dists.copy()
        out_ring[out_ring <= 0] = 0
        out_ring[out_ring > out_ring_width] = 0
        out_ring[out_ring != 0] = 1

        surround_ring = in_on_ring.astype(np.uint8) | out_ring.astype(np.uint8)
        surround_ring = torch.tensor(surround_ring).to(self.device)

        return surround_ring
