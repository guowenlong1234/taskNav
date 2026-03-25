from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


class CollectDinoPatchEncoder:
    def __init__(
        self,
        rgb_encoder,
        device: torch.device,
        image_size: int = 196,
    ):
        if not hasattr(rgb_encoder, "backbone"):
            raise AttributeError("rgb_encoder must expose a DINOv2 backbone")
        if not hasattr(rgb_encoder, "rgb_transform"):
            raise AttributeError("rgb_encoder must expose rgb_transform")
        self.rgb_encoder = rgb_encoder
        self.device = device
        self.image_size = int(image_size)

    @staticmethod
    def _to_uint8_tensor(images: Union[torch.Tensor, Sequence[np.ndarray]]) -> torch.Tensor:#把一批图片统一转换成 torch.uint8 的 4D tensor，格式要求是 B x H x W x C，其中 C=3
        if torch.is_tensor(images):
            tensor = images.detach().cpu()
        else:
            array = np.stack([np.asarray(img) for img in images], axis=0)
            tensor = torch.from_numpy(array)

        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D image batch, got shape={tuple(tensor.shape)}")
        if tensor.shape[-1] != 3:
            raise ValueError(
                f"Expected image batch in HWC format with 3 channels, got shape={tuple(tensor.shape)}"
            )

        if tensor.dtype == torch.uint8:
            return tensor

        if tensor.is_floating_point():
            maxv = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
            if maxv <= 1.0:
                tensor = tensor * 255.0
            tensor = tensor.clamp(0.0, 255.0).round().to(torch.uint8)
            return tensor

        return tensor.to(torch.uint8)

    def encode_rgb_images(
        self, images: Union[torch.Tensor, Sequence[np.ndarray]]
    ) -> torch.Tensor:
        image_tensor = self._to_uint8_tensor(images).to(self.device, non_blocking=True)
        image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()
        image_tensor = image_tensor.to(torch.float32).div_(255.0)
        if image_tensor.shape[-2:] != (self.image_size, self.image_size):
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        image_tensor = self.rgb_encoder.rgb_transform(image_tensor)

        with torch.no_grad():
            patch_tokens = self.rgb_encoder.backbone.forward_features(image_tensor)[
                "x_norm_patchtokens"
            ]

        return patch_tokens.detach().cpu()
