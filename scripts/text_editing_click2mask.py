import random
import os
import gc
import numpy as np
from PIL import Image
import cv2
from diffusers import DDIMScheduler, StableDiffusionPipeline
from pytorch_lightning import seed_everything
import torch
from scipy.ndimage import gaussian_filter
import sys
sys.path.append("./scripts")
from dyn_mask import DynMask, get_surround
from arguments import parse_args
from clicker import ClickCreate, ClickDraw
from augmentations import ImageAugmentations
from constants import Const, N


def read_image(image: Image.Image, device, dest_size):
    image = image.convert("RGB")
    image = image.resize(dest_size, Image.LANCZOS) if dest_size != image.size else image
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)
    image = image * 2.0 - 1.0
    return image


class Click2Mask:
    def __init__(self):
        self.args = parse_args()
        self.device = torch.device(f"cuda:{self.args.gpu_id}")
        self.load_models()

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    @torch.enable_grad()
    def blended_latent_diffusion(
        self,
        dyn_mask,
        create_dyn_mask,
        seed,
        original_rand_latents,
        scheduler,
        blending_percentage,
        total_steps,
        source_latents,
        text_embeddings,
        guidance_scale,
        dyn_start_step_i=None,
        dyn_cond_stop_step_i=None,
        dyn_final_stop_step_i=None,
        max_area_ratio_for_dilation=None,
        last_step_threshed_latent_mask=None,
        rerun_return_during_step_i=None,
    ):

        seed_everything(seed)
        use_plain_dilation_from_latent_mask = not create_dyn_mask
        blending_steps_t = scheduler.timesteps[
            int(len(scheduler.timesteps) * blending_percentage) :
        ]
        latents = original_rand_latents

        if create_dyn_mask:
            update_steps = list(range(dyn_start_step_i, dyn_cond_stop_step_i + 1))
            update_steps = [u for u in update_steps if 0 != u < len(blending_steps_t)]
            first_update_step, orig_last_update_step = update_steps[0], update_steps[-1]
            best_step_i = orig_last_update_step

        if last_step_threshed_latent_mask is not None:
            latent_mask = last_step_threshed_latent_mask

        for step_i, t in enumerate(blending_steps_t):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latent_pred_z0 = scheduler.step(noise_pred, t, latents).pred_original_sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if rerun_return_during_step_i == step_i:
                return latents, latent_mask

            # dilation for rerun + final runs
            elif use_plain_dilation_from_latent_mask:
                latent_mask = dyn_mask.get_plain_dilated_latent_mask(
                    last_step_latent_mask=last_step_threshed_latent_mask,
                    step_i=step_i,
                    total_steps=total_steps,
                    max_area_ratio_for_dilation=max_area_ratio_for_dilation,
                    rerun_dyn_start_step_i=None
                    if not rerun_return_during_step_i
                    else dyn_start_step_i,
                )

            # mask evolution
            elif create_dyn_mask:
                if step_i in update_steps:
                    latent_mask = dyn_mask.evolve_mask(
                        step_i=step_i,
                        decoder=self.vae.decode,
                        latent_pred_z0=latent_pred_z0,
                        source_latents=source_latents,
                        return_only=N.LATENT_MASK,
                    )
                    # Rerun
                    latents, _ = self.blended_latent_diffusion(
                        dyn_mask,
                        create_dyn_mask=False,
                        seed=seed,
                        original_rand_latents=original_rand_latents,
                        scheduler=scheduler,
                        blending_percentage=blending_percentage,
                        total_steps=total_steps,
                        source_latents=source_latents,
                        text_embeddings=text_embeddings,
                        guidance_scale=guidance_scale,
                        dyn_start_step_i=dyn_start_step_i,
                        max_area_ratio_for_dilation=Const.RERUN_MAX_AREA_RATIO_FOR_DILATION,
                        last_step_threshed_latent_mask=latent_mask,
                        rerun_return_during_step_i=step_i,
                    )
                elif step_i < first_update_step:  # initial dilation
                    latent_mask = dyn_mask.set_cur_masks(
                        step_i=step_i, return_only=N.LATENT_MASK
                    )

            # Blending
            noise_source_latents = scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

            if create_dyn_mask:
                if step_i >= orig_last_update_step:
                    dyn_mask.make_cached_masks_clones(name=step_i)
                    dyn_mask.latents_hist[step_i] = latents
                    dyn_mask.latent_masks_hist[step_i] = latent_mask

                    if step_i >= orig_last_update_step + 2:
                        step_prev1_better = (
                            dyn_mask.closs_hist[step_i - 1]
                            < dyn_mask.closs_hist[step_i - 2]
                        )
                        if step_prev1_better:
                            best_step_i = step_i - 1
                        if (not step_prev1_better) or (step_i > dyn_final_stop_step_i):
                            # we need an extra step to calculate clip loss for last evolved mask
                            latents = dyn_mask.latents_hist[best_step_i]
                            latent_mask = dyn_mask.latent_masks_hist[best_step_i]
                            dyn_mask.set_masks_from_cached_masks_clones(
                                name=best_step_i
                            )
                            break
                    update_steps.append(step_i + 1)

        return latents, latent_mask

    @torch.no_grad()
    def edit_image(
        self,
        image_pil,
        click_pil,
        prompts,
        height,
        width,
        num_inference_steps,
        num_static_inference_steps,
        guidance_scale,
        seed,
        blending_percentage,
    ):
        generator = torch.manual_seed(seed)
        batch_size = len(prompts)

        self.scheduler.set_timesteps(num_inference_steps)

        image_pil = image_pil.resize((height, width), Image.LANCZOS)
        image_np = np.array(image_pil)[:, :, :3]
        source_latents = self._image2latent(image_np)

        init_image_tensor = read_image(
            image=image_pil, device=self.device, dest_size=(height, width)
        )

        total_steps = num_inference_steps - int(
            len(self.scheduler.timesteps) * blending_percentage
        )
        dyn_mask = DynMask(
            click_pil, self.args, init_image_tensor, self.device, total_steps
        )

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device).half()

        original_rand_latents = latents

        dyn_start_step_i = (
            Const.DYN_START
            if Const.DYN_START > 1
            else round(Const.DYN_START * total_steps)
        )
        dyn_cond_stop_step_i = (
            Const.DYN_COND_STOP
            if Const.DYN_COND_STOP > 1
            else round(Const.DYN_COND_STOP * total_steps)
        )
        dyn_final_stop_step_i = (
            Const.DYN_FINAL_STOP
            if Const.DYN_FINAL_STOP > 1
            else round(Const.DYN_FINAL_STOP * total_steps)
        )

        # Evolve mask
        self.blended_latent_diffusion(
            dyn_mask=dyn_mask,
            create_dyn_mask=True,
            seed=seed,
            original_rand_latents=original_rand_latents,
            scheduler=self.scheduler,
            blending_percentage=blending_percentage,
            total_steps=total_steps,
            source_latents=source_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            dyn_start_step_i=dyn_start_step_i,
            dyn_cond_stop_step_i=dyn_cond_stop_step_i,
            dyn_final_stop_step_i=dyn_final_stop_step_i,
        )

        # Final run
        self.static_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.static_scheduler.set_timesteps(num_static_inference_steps)
        total_static_steps = num_static_inference_steps - int(
            len(self.static_scheduler.timesteps) * blending_percentage
        )
        latents_list = []
        latent_masks_list = []
        seeds_list = []

        seeds_to_run = random.sample(range(1, Const.MAX_SEED), Const.N_OUTS_FOR_DYN_MASK - 1)
        print(f"running output (from {Const.N_OUTS_FOR_DYN_MASK}): ", end="")
        for out_i in range(Const.N_OUTS_FOR_DYN_MASK):
            print(f"{out_i + 1}", end="... ")
            orig_l = original_rand_latents
            seed_i = seed
            if out_i > 0:
                seed_i = seeds_to_run[out_i - 1]
                orig_l = torch.randn(
                    (batch_size, self.unet.config.in_channels, height // 8, width // 8),
                    generator=torch.manual_seed(seed_i),
                )
                orig_l = orig_l.to(self.device).half()
            latents, latent_mask = self.blended_latent_diffusion(
                dyn_mask=dyn_mask,
                create_dyn_mask=False,
                seed=seed_i,
                original_rand_latents=orig_l,
                scheduler=self.static_scheduler if out_i > 0 else self.scheduler,
                blending_percentage=blending_percentage,
                total_steps=total_static_steps if out_i > 0 else total_steps,
                source_latents=source_latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                max_area_ratio_for_dilation=Const.MAX_AREA_RATIO_FOR_DILATION,
                last_step_threshed_latent_mask=dyn_mask.get_curr_masks(
                    return_only=N.LATENT_MASK
                ),
            )
            latents_list.append(latents)
            latent_masks_list.append(latent_mask)
            seeds_list.append(seed_i)

        print("scoring...")
        results = self.score_and_arrange_results(
            dyn_mask=dyn_mask,
            latents_list=latents_list,
            latent_masks_list=latent_masks_list,
            n_runs=Const.N_RUNS_ON_SCORES,
            aug_num=Const.N_AUGS_ON_SCORES,
            alpha_mask_dilation_on_512=Const.ALPHA_MASK_DILATION_ON_512,
        )

        return results

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def back_preserve_with_gauss(self, decoded_img, latent_mask, dyn_mask):
        upsampled_mask = latent_mask.cpu().numpy().squeeze()
        upsampled_mask = cv2.resize(
            upsampled_mask.squeeze().astype(np.float32),
            dyn_mask.decoded_size,
            Image.LANCZOS,
        )
        upsampled_mask = upsampled_mask > 0.5
        g_mask = gaussian_filter(
            upsampled_mask.astype(float), sigma=Const.BACK_PRES_SIGMA
        )
        g_mask = torch.from_numpy(g_mask).half().to(self.device)
        g_mask = (g_mask * Const.BACK_PRES_SCALE).clip(0, 1)
        g_mask[upsampled_mask > 0.5] = 1
        blended = decoded_img * g_mask + dyn_mask.init_image * (1 - g_mask)

        return blended

    def score_and_arrange_results(
        self,
        dyn_mask,
        latents_list,
        latent_masks_list,
        n_runs,
        aug_num,
        alpha_mask_dilation_on_512,
    ):
        results = []
        raw_d_prompt = np.zeros((n_runs, len(latents_list)))

        for i, (latents, latent_mask) in enumerate(
            zip(latents_list, latent_masks_list)
        ):
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                img = self.vae.decode(latents).sample
            img = self.back_preserve_with_gauss(img, latent_mask, dyn_mask)
            results.append({"im": img, "latent_mask": latent_mask})

            alpha_mask = get_surround(
                latent_mask,
                alpha_mask_dilation_on_512 * (latent_mask.shape[-1] / 512.0),
                self.device,
            )
            if aug_num is not None:
                image_augmentations = ImageAugmentations(
                    self.args.alpha_clip_scale, aug_num
                )
            else:
                image_augmentations = None
            for run_i in range(n_runs):
                raw_d_prompt[run_i][i] = dyn_mask.alpha_clip_loss(
                    img,
                    alpha_mask,
                    dyn_mask.text_features,
                    image_augmentations=image_augmentations,
                    augs_with_orig=(run_i == 0),
                    return_as_similarity=True,
                )

        raw_d_prompt = raw_d_prompt.mean(axis=0)
        for i, res in enumerate(results):
            res["dist"] = float(raw_d_prompt[i])

        return results


def click2mask_app(prompt: str, image_pil: Image.Image, point512: np.ndarray):
    c2m = Click2Mask()
    c2m.args.prompt = prompt
    results = []

    for mask_i in range(c2m.args.n_masks):
        print(f"\nEvolving mask {mask_i + 1}...")
        seed = (
            c2m.args.seed
            if (c2m.args.seed and mask_i == 0)
            else random.sample(range(1, Const.MAX_SEED), 1)[0]
        )
        seed_everything(seed)

        click_draw = ClickDraw()
        click_pil, _ = click_draw(image_pil, point512=point512)

        mask_i_results = c2m.edit_image(
            image_pil=image_pil,
            click_pil=click_pil,
            prompts=[c2m.args.prompt] * Const.BATCH_SIZE,
            height=Const.H,
            width=Const.W,
            num_inference_steps=Const.NUM_INFERENCE_STEPS,
            num_static_inference_steps=Const.NUM_STATIC_INFERENCE_STEPS,
            guidance_scale=Const.GUIDANCE_SCALE,
            seed=seed,
            blending_percentage=Const.BLENDING_START_PERCENTAGE,
        )

        results += mask_i_results

    sorted_results = sorted(results, key=lambda k: k["dist"], reverse=True)
    out_img = sorted_results[0]["im"]
    out_img = (out_img / 2 + 0.5).clamp(0, 1)
    out_img = out_img.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
    out_img = (out_img * 255).round().astype(np.uint8)

    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nCompleted.")

    return out_img


if __name__ == "__main__":
    c2m = Click2Mask()

    img_dir = os.path.dirname(c2m.args.image_path)
    img_name = os.path.basename(os.path.normpath(c2m.args.image_path))
    img_base_name = os.path.splitext(img_name)[0]
    results = []

    for mask_i in range(c2m.args.n_masks):
        print(f"\nEvolving mask {mask_i + 1}...")
        seed = (
            c2m.args.seed
            if (c2m.args.seed and mask_i == 0)
            else random.sample(range(1, Const.MAX_SEED), 1)[0]
        )
        seed_everything(seed)

        click_ext = [
            ext
            for ext in ("jpg", "JPG", "JPEG", "jpeg", "png", "PNG")
            if os.path.exists(os.path.join(img_dir, f"{img_base_name}_click.{ext}"))
        ]
        if (not click_ext) or (mask_i == 0 and c2m.args.refresh_click):
            click_create = ClickCreate()
            c2m.args.click_path = click_create(
                c2m.args.image_path, os.path.join(img_dir, f"{img_base_name}_click.jpg")
            )
        else:
            c2m.args.click_path = os.path.join(
                img_dir, f"{img_base_name}_click.{click_ext[0]}"
            )

        mask_i_results = c2m.edit_image(
            image_pil=Image.open(c2m.args.image_path),
            click_pil=Image.open(c2m.args.click_path),
            prompts=[c2m.args.prompt] * Const.BATCH_SIZE,
            height=Const.H,
            width=Const.W,
            num_inference_steps=Const.NUM_INFERENCE_STEPS,
            num_static_inference_steps=Const.NUM_STATIC_INFERENCE_STEPS,
            guidance_scale=Const.GUIDANCE_SCALE,
            seed=seed,
            blending_percentage=Const.BLENDING_START_PERCENTAGE,
        )

        results += mask_i_results

    os.makedirs(c2m.args.output_dir, exist_ok=True)
    sorted_results = sorted(results, key=lambda k: k["dist"], reverse=True)
    out_img = sorted_results[0]["im"]
    out_img = (out_img / 2 + 0.5).clamp(0, 1)
    out_img = out_img.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
    out_img = (out_img * 255).round().astype(np.uint8)
    out_path = os.path.join(c2m.args.output_dir, f"{img_base_name}_out.jpg")
    Image.fromarray(out_img).save(out_path, quality=95)

    print(f"\nCompleted.\nOutput image path:\n{os.path.abspath(out_path)}")
