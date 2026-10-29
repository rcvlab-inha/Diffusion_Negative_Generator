import os
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

PROMPT_DIR = "./prompt"

class ImgReader:

    def __init__(self, img_pth: Path, tgt_obj: str, ref_size: tuple[int, int], mode: str="RGB"):
        self.img_pth = img_pth
        self.mode = mode
        self.tgt_obj = tgt_obj
        self.down_ratio = 1
        with open(PROMPT_DIR + f"/{self.tgt_obj}.json", "r") as f:
            self.ref = json.load(f)

        self._data_stream = [Path(f) for f in os.listdir(img_pth) if f.endswith((".jpg", ".png", ".jpeg"))]
        self.len = len(self._data_stream)
        self.ref_size = ref_size
        self.check_json_vaild()

    def check_json_vaild(self):
        val_img_pth = self.img_pth / self._data_stream[0]
        val_img = Image.open(val_img_pth).convert(self.mode)
        
        if self.ref["mask"]["maxy"] + self.ref["mask"]["roiy"] > val_img.size[1]:
            raise RuntimeError("The Y area of the mask may extend beyond the image area.\n" \
                               f"{self.ref['mask']['maxy']} (maxy) + {self.ref['mask']['roiy']} (roiy) > {val_img.size[1]} (y size)")
    
    def get_data(self, idx: int) -> dict:
        
        data_pth = self.img_pth / self._data_stream[idx]
        img = Image.open(data_pth).convert(self.mode)
        mask_size = self.set_mask_area(self.ref, img.size)
        down_ratio = 1

        if img.size[0] != self.ref_size[0] or img.size[1] != self.ref_size[1]:
            down_ratio = max(img.size[0] / self.ref_size[0], img.size[1] / self.ref_size[1])
            new_width = int(((img.width / down_ratio) // 16) * 16)
            new_height = int(((img.height / down_ratio) // 16) * 16)
            img = img.resize((new_width, new_height))
            
        # image resizing (make 16x)
        if (img.size[0] % 16) != 0 or (img.size[1] % 16) != 0:
            new_width = int(((img.width) // 16) * 16)
            new_height = int(((img.height) // 16) * 16)
            img = img.resize((new_width, new_height))

        tmp_size = tuple([int(((i / down_ratio) // 16) * 16) for i in mask_size])
        
        print(img.size, tmp_size)
        
        try:
            output = {"image": img, "mask_size": tmp_size, "path": data_pth, "down_ratio": down_ratio, "prompt": self.ref["prompt"], "negative": self.ref["negative"]}

        except:
            output = {"image": img, "mask_size": tmp_size, "path": data_pth, "down_ratio": down_ratio, "prompt": self.ref["prompt"], "negative": None}


        return output

    def update_data_stream(self, res_pth):
        _fin_stream = {Path(f.name) for f in res_pth.iterdir() if f.suffix == ".png" or f.suffix == ".jpg"}
        _data_stream = set(self._data_stream)
        _data_stream = _data_stream - _fin_stream

        self.len = len(_data_stream)
        self._data_stream = list(_data_stream)

    def remake_mask(self):
        mask_size = self.set_mask_area(self.ref, self.ref_size)
        tmp_size = tuple([int(((i / self.down_ratio) // 16) * 16) for i in mask_size])

        return tmp_size

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return self.len
        
    @staticmethod
    def set_mask_area(ref, size: tuple[int, int]):
        minx = ref["mask"]["minx"]
        maxx = ref["mask"]["maxx"]
        miny = ref["mask"]["miny"]
        maxy = ref["mask"]["maxy"]
        roiy = ref["mask"]["roiy"]
        maskx = random.randrange(minx, maxx+1, 10)
        masky = random.randrange(miny, maxy+1, 10)
        stx = random.randrange(0, size[0]+1-maskx, 10)
        sty = random.randrange(roiy, size[1]+1-masky, 10)

        mask_area = (stx, sty, stx+maskx, sty+masky)

        return mask_area

    @staticmethod
    def draw_mask(inp_size: tuple[int, int],
                  mask_size: tuple[int, int, int, int],
                  blur_radius: int=15):

        mask = Image.new('L', inp_size, 0)

        draw = ImageDraw.Draw(mask)
        draw.rectangle(mask_size, fill=255)

        blured_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return blured_mask, mask

if __name__ == '__main__':
    pass