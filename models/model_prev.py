from typing import Tuple
import PIL
import PIL.Image

import torch
import numpy as np
from scipy import ndimage
from diffusers import StableDiffusion3InpaintPipeline, FluxFillPipeline

import os
import sys

from tools import composite_image, save_inter_mask
from PIL import Image, ImageFilter
from models.data import ImgReader
from lang_sam import LangSAM


class InValidMaskError(Exception):
    pass

class GenerationPipeline:

    def __init__(self, model_id: str, tgt_obj: str, device: str="cuda"):
        self.device = device
        self.tgt_obj = tgt_obj
        self.model_id = model_id

        if model_id == "stabilityai/stable-diffusion-3.5-large":
            self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            ).to(device)
        elif model_id == "black-forest-labs/FLUX.1-Fill-dev":
            self.pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.bfloat16
            ).to(device)

        self.sam = LangSAM(device=device, sam_type="sam2.1_hiera_large")

    def generate(self,
                 prompt: str,
                 img: Image,
                 mask: Image,
                 negative_prompt: str = None,
                 num_inference_steps: int = 50,
                 strength: float = 0.6,
                 guidance_scale: int = 7) -> Image:
        
        width, height = img.size
        
        if self.model_id == "stabilityai/stable-diffusion-3.5-large":
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                mask_image=mask,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale
            ).images[0]
        elif self.model_id == "black-forest-labs/FLUX.1-Fill-dev":
            image = self.pipe(
                prompt=prompt,
                image=img,
                mask_image=mask,
                width=width,
                height=height,
            ).images[0]

        return image
    
    def predict_inter_mask(self, img: Image.Image, mask: tuple[int, int, int, int]) -> np.ndarray:
        """
        img  : PIL.Image
        bbox : (x1, y1, x2, y2)
        return: np.ndarray
        """
        radius = 15
        pad = int(3 * radius)

        x1, y1, x2, y2 = mask
        w, h = img.size

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

        out = self.sam.predict(
            [crop],
            [f"{self.tgt_obj}"]
        )

        masks = out[0]["masks"]

        if isinstance(masks, list):
            return masks

        if masks.ndim == 2:
            masks = masks[None, ...]

        full = np.zeros((masks.shape[0], h, w), dtype=masks.dtype)
        full[:, y1:y2, x1:x2] = masks
        
        return full

    def make_cluster(self, seg: np.ndarray, mask_size: list, min_threshold: int, max_threshold: int) -> tuple[np.ndarray, int]:
        
        seg = (seg > 0).astype(np.uint8) * 255
        x1, y1, x2, y2 = mask_size
        mask = np.zeros_like(seg, dtype=np.uint8)
        mask[:, y1:y2, x1:x2] = 1
        seg[mask == 0] = 0
        label, _ = make_segment(seg[0])
        label, n_seg = remake_segment2(label, min_threshold, max_threshold)

        return label, n_seg

    def run(self,
            gen_info: dict, 
            inp: dict,
            mask: Image,
            negative_prompt: str = None,
            num_inference_steps: int = 50,
            strength: float = 0.6,
            guidance_scale: int = 7)->tuple[PIL.Image.Image, bool, np.ndarray]:

        prompt = inp["prompt"]
        raw = inp["image"]
        min_ratio = gen_info["min_ratio"]
        max_ratio = gen_info["max_ratio"]

        n_seg, cnt = 0, 0
        while 0 == n_seg or n_seg > gen_info["max_obj"]:
            print("Generate Object")
            img = self.generate(  # inpaint
                prompt=prompt,
                img=raw,
                mask=mask,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale
            )

            seg = self.predict_inter_mask(img=img, mask=inp["mask_size"])  # LangSAM
            
            if cnt >= gen_info["max_try"]:  # Maximum Generation try limit
                raise InValidMaskError("Generation failed")
            cnt += 1
            
            if isinstance(seg, list):  # if object not exist, LangSAM return list
                print("no object!!")
                continue
            
            x1, y1, x2, y2 = inp["mask_size"]
            
            tot_pixel = (x2 - x1) * (y2 - y1)

            min_threshold = int(tot_pixel * min_ratio)
            max_threshold = int(tot_pixel * max_ratio)
            label, n_seg = self.make_cluster(seg, inp["mask_size"], min_threshold=min_threshold, max_threshold=max_threshold)  # make cluster, filter noise segment
            
        print("-------------------- fin generation --------------------")

        out = composite_image(img, raw, label, radius=gen_info["blur_radius"])
        return out, label

def make_segment(inp: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=int)
    labels, num_clusters = ndimage.label(inp, structure=structure)
    return labels, num_clusters

def remake_segment2(label, min_threshold, max_threshold):
    counts = np.bincount(label.ravel())
    counts[0] = 0
    
    keep = np.flatnonzero((counts > min_threshold) & (counts < max_threshold))
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[keep] = np.arange(1, len(keep) + 1)
    res = mapping[label]
    tot = len(keep)

    res[res != 0] = 255
    res = res.astype(np.uint8)

    return res, tot


if __name__ == "__main__":

    SEED = 1234
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
