import torch
from torch import nn
import kornia.augmentation as K


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7, resize=True):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number

        self.augmentations = [
            K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),
            K.RandomPerspective(0.7, p=p),
        ]

        self.resize = (
            nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
            if resize
            else (lambda x: x)
        )

    def forward(self, image, mask, with_orig=True):
        """Extends the image and mask with identical augmentations

        If the input consists of image I, and mask M, the extended augmented output will be:
         [I_aug1, I_aug2, I_aug3, ...], [M_aug1, M_aug2, M_aug3, ...]
        If with_orig=True, the extended augmented output will be:
         [I, I_aug1, I_aug2, ...], [M, M_aug1, M_aug2, ...]

        Args:
            image: tensor of shape [1, C, H, W]
            mask: tensor of shape [1, 1, H, W]
            with_orig: if True, first returned image and mask will be un-augmented inputs

        Returns:
            tuple of (extended images of shape [augmentations_number, C, H, W],
                      extended masks of shape [augmentations_number, 1, H, W])
        """
        # Duplicate the inputs, in contrast to regular augmentations that do not change the number of samples
        resized_images = self.resize(image)
        resized_images = resized_images.repeat(self.augmentations_number, 1, 1, 1)

        resized_masks = self.resize(mask)
        resized_masks = resized_masks.repeat(self.augmentations_number, 1, 1, 1)

        batch_size = image.shape[0]
        if with_orig:
            # At least one non-augmented image
            non_aug_inputs = resized_images[:batch_size]
            aug_inputs = resized_images[batch_size:]

            non_aug_masks = resized_masks[:batch_size]
            aug_masks = resized_masks[batch_size:]

            for trans in self.augmentations:
                trans_params = trans.forward_parameters(aug_inputs.shape)
                aug_inputs = trans(aug_inputs, trans_params)
                aug_masks = trans(aug_masks, trans_params)

            updated_input_batch = torch.cat([non_aug_inputs, aug_inputs], dim=0)
            updated_mask_batch = torch.cat([non_aug_masks, aug_masks], dim=0)
        else:
            aug_inputs = resized_images
            aug_masks = resized_masks

            for trans in self.augmentations:
                trans_params = trans.forward_parameters(aug_inputs.shape)
                aug_inputs = trans(aug_inputs, trans_params)
                aug_masks = trans(aug_masks, trans_params)

            updated_input_batch = aug_inputs
            updated_mask_batch = aug_masks

        return updated_input_batch, updated_mask_batch
