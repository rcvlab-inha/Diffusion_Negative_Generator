import os
import sys
import json
import logging
import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from models.model import GenerationPipeline, InValidMaskError
from models.data import ImgReader

def convert_npy(img):
    img_arr = np.array(img, dtype=np.bool_)
    return img_arr

def handle_exception(exc_type, exc_value, exc_traceback):
    """logging at error"""
    if issubclass(exc_type, KeyboardInterrupt):
        logging.warning("Generating interrupted by user (KeyboardInterrupt).")
    else:
        logging.error("Generating crashed!", exc_info=(exc_type, exc_value, exc_traceback))

    logging.info("Generating did not finish properly.")


if __name__ == "__main__":

    with open("./config/config.json", "r") as f:
        config = json.load(f)

    REF_SIZE = config["REF_SIZE"]
    SEED = config["SEED"]
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    MODEL_ID = config["MODEL_ID"]
    DEVICE = config["DEVICE"]
    tgt_obj = config["tgt_obj"]
    continued = config["continued"]
    gen_info = config["gen_info"]

    print("MODEL_ID :", MODEL_ID)
    print("DEVICE :", DEVICE)
    print("tgt_obj :", tgt_obj)
    print("set continued :", continued)
    print("gen_info :", gen_info)

    img_pth = config["img_pth"]

    print("img_pth :", img_pth)
    
    img_pth = Path(img_pth)
    os.makedirs(img_pth, exist_ok=True)

    save_pth = config["save_pth"]
    print("save_pth :", save_pth)

    save_pth = Path(save_pth)
    os.makedirs(save_pth, exist_ok=True)

    mask_pth = config["mask_pth"]
    print("mask_pth :", mask_pth)

    mask_pth = Path(mask_pth)
    os.makedirs(mask_pth, exist_ok=True)

    sam_mask_pth = config["sam_mask_pth"]
    print("sam_mask_pth :", sam_mask_pth)

    sam_mask_pth = Path(sam_mask_pth)
    os.makedirs(sam_mask_pth, exist_ok=True)

    log_dir = "./logs"
    log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n\n\n\n****************************************************************")
    input("********************press enter before start********************\n"\
          "****************************************************************\n\n\n\n")

    ntime = datetime.datetime.now()
    log_file = os.path.join(
        log_dir,
        f"{ntime.year}_{ntime.month}_{ntime.day}_{ntime.hour}_{ntime.minute}.log"
    )
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.info(f"device = {DEVICE}")
    logging.info(f"model = {MODEL_ID}")
    logging.info(f"target object = {tgt_obj}")
    logging.info(f"set input path = {img_pth}")
    logging.info(f"set result path = {save_pth}")
    logging.info(f"set mask path = {mask_pth}")
    logging.info(f"set continued : {str(continued)}")

    logging.info("Create ImgReader")
    reader = ImgReader(img_pth=img_pth, tgt_obj=tgt_obj, ref_size=REF_SIZE)
    logging.info("Create DataGenerator")
    pipe = GenerationPipeline(model_id=MODEL_ID, tgt_obj=tgt_obj, device=DEVICE)

    if continued:
        logging.info("remove duplicated file")
        pre_len = len(reader)
        reader.update_data_stream(save_pth)
        print(f"change # of target file {pre_len} -> {len(reader)}")
        
    logging.info("Start Generating")
    sys.excepthook = handle_exception

    try:
        for num, data in enumerate(reader):
            logging.info(f"{num} --> {len(reader)}")

            mask, raw_mask = reader.draw_mask(data["image"].size, data["mask_size"], gen_info["blur_radius"])
            while True:
                try:
                    res, sam_mask = pipe.run(
                                gen_info=gen_info,
                                inp=data,
                                mask=mask,
                                negative_prompt=data["negative"],
                                num_inference_steps=50,
                                strength=0.6,
                                guidance_scale=7
                            )
                    break

                except InValidMaskError:
                    print("prev_mask :", data["mask_size"])
                    data["mask_size"] = reader.remake_mask()
                    print("updated_mask :", data["mask_size"])
                    mask, raw_mask = reader.draw_mask(data["image"].size, data["mask_size"], gen_info["blur_radius"])
                    continue
                    

            sam_mask = sam_mask.astype(np.uint8)
            sam_mask_img = Image.fromarray(sam_mask)
            sam_mask_img.save(sam_mask_pth / data["path"].name)
            np.save(sam_mask_pth / data["path"].with_suffix(".npy").name, sam_mask)

            res.save(save_pth / data["path"].name)
            raw_mask.save(mask_pth / data["path"].name)
            img_arr = convert_npy(raw_mask)
            np.save(mask_pth / data["path"].with_suffix(".npy").name, img_arr)

    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        print(e)
        sys.exit(1)