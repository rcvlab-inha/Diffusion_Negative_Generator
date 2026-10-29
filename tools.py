import torch
import numpy as np

import PIL.Image
from PIL import Image, ImageFilter

save_img_cnt = 1
save_mask_cnt = 1

def draw_mask(image: PIL.Image, mask:np.ndarray, color: tuple[int,int,int]=(255,255,255)) -> PIL.Image:
    img_np = np.array(img)

    overlay_color = np.array(color, dtype=np.uint8)
    img_np[mask==255] = overlay_color

    res_img = Image.fromarray(img_np)

    return res_img

def composite_image(obj_img: PIL.Image, bg_img: PIL.Image, mask: np.ndarray, radius: int=5):
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
    out = Image.composite(obj_img, bg_img, mask_img)

    return out

def save_inter_img(image: PIL.Image):
    global save_img_cnt

    image.save(f"inter_img_{save_img_cnt}.png")
    save_img_cnt += 1

def save_inter_mask(mask: np.ndarray):
    """
    must (H, W, C)
    """
    global save_mask_cnt
    mask = mask.astype(np.uint8)
    
    if np.max(mask) == 1:
        mask = mask * 255
    
    mask_img = Image.fromarray(mask)
    mask_img.save(f"inter_mask_{save_mask_cnt}.png")
    save_mask_cnt += 1

def overlay_img(img_pil: Image.Image, segment: np.ndarray, alpha: float = 0.5) -> Image.Image:
    H, W = segment.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[..., :3] = [255, 0, 0]
    overlay[..., 3] = (segment.astype(np.uint8) * int(alpha*255))
    img_rgba = img_pil.convert("RGBA")
    out = Image.alpha_composite(img_rgba, Image.fromarray(overlay, "RGBA")).convert("RGB")
    return out

if __name__ == "__main__":

    pass