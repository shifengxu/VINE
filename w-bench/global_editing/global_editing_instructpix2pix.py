import os
import PIL
import pandas as pd
import requests
import torch
from tqdm import tqdm
from diffusers.utils import load_image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler


def edit_by_InstructPix2Pix(device, guidance, inputPath_img, inputPath_prmt, outputPath):
    # Model Preparation
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None,
                                                                  requires_safety_checker=False)
    pipe.to(device)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Acquire Data and Process Editing:
    os.makedirs(outputPath, exist_ok=True)
    ID = pd.read_csv(inputPath_prmt).iloc[:, 1].tolist()
    for idx, prompt in tqdm(enumerate(pd.read_csv(inputPath_prmt).iloc[:, 2].tolist())):
        #if guidance == 9 and idx < 649:    # todo ***
            #continue
        image = load_image(os.path.join(inputPath_img, f"{str(idx)}_{str(ID[idx])}_wm.png")).resize((512, 512))

        image = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=guidance,
        ).images[0]

        # Edited images are saved to `edited/`
        path = outputPath + f"{str(idx)}_{str(ID[idx])}.png"
        print(f"\t> Edited image {str(idx)}_{str(ID[idx])} is saved at: `{path}`")
        image.save(path)


if __name__ == '__main__':
    MODE = "INSTRUCT"
    SPEC = "_Pix2Pix"

# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    for WM in ["vine"]:   # todo *** (WM)
        GUIDANCE_RANGE = [5, 6, 7, 8, 9]
        DEVICE = 'cuda:0'   # todo *** (CUDA)

        for GUIDANCE in GUIDANCE_RANGE:
            print(f"\n\n>> Currently processing the CHOICE of {GUIDANCE}...\n")
            INPUT_PATH_IMAGE = f"/home/shilin1/projs/datasets/{WM}_encoded/512/INSTRUCT_1K"          # todo *** (INPUT)
            INPUT_PATH_PROMPT = f"/home/shilin1/projs/datasets/W-Bench/INSTRUCT_1K/prompts.csv"
            OUTPUT_PATH = f"/home/shilin1/projs/datasets/edited_image/{WM}/{MODE}{SPEC}/{GUIDANCE}/"   # todo *** (OUTPUT)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

            print(f"\n>> Processing edited images for [{MODE}], with GUIDANCE={GUIDANCE}, on DEVICE={DEVICE}...")
            edit_by_InstructPix2Pix(
                device=DEVICE,
                guidance=GUIDANCE,
                inputPath_img=INPUT_PATH_IMAGE,
                inputPath_prmt=INPUT_PATH_PROMPT,
                outputPath=OUTPUT_PATH
            )
            """ PROCESS END """


