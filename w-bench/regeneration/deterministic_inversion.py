from utils_det_inversion import container_inversion, load_image
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
repo = 'stabilityai/stable-diffusion-2-1-base'


if __name__ == "__main__":
# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    WMs = ["vine"]  # todo *** (WMs)
    for WM in WMs:
        for STEPS in [15, 25, 35, 45]:
            DEVICE = 'cuda:0'   # todo *** (CUDA)
            INPUT_PATH = f"/home/shilin1/projs/datasets/{WM}_encoded/512/DET_INVERSION_1K"        
            OUTPUT_PATH = f"/home/shilin1/projs/datasets/edited_image/{WM}/DET_INVERSION_1K/{STEPS}"     # todo *** (OUTPUT)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD End ------------------------------------------------------------

            parser = ArgumentParser()
            parser.add_argument('--images_folder_path', type=str, default=INPUT_PATH)
            parser.add_argument('--output_path', type=str, default=OUTPUT_PATH)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            parser.add_argument('--inv_type', type=str, default='dpm')
            parser.add_argument('--dpm_order', type=int, default=2)
            parser.add_argument('--num_steps', type=int, default=STEPS)
            parser.add_argument('--batch_size', type=int, default=1)

            args = parser.parse_args()
            folder_path = args.images_folder_path
            output_folder = args.output_path
            inv_type = args.inv_type
            dpm_order = args.dpm_order
            num_steps = args.num_steps
            batch_size = args.batch_size

            dtype = torch.float32
            device = DEVICE
            ldm_pipe = StableDiffusionPipeline.from_pretrained(repo, safety_checker=None, torch_dtype=dtype)
            ldm_pipe.to(device)

            files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
            
            with torch.no_grad():
                for file in tqdm(files):
                    img_tensor = load_image(os.path.join(folder_path, file))
                    img_tensors = img_tensor.to(device=device, dtype=dtype)  #（1，3，512，512）

                    output_image_tensors = container_inversion(img_tensors, ldm_pipe, inv_type=inv_type, dpm_order=dpm_order, num_steps=num_steps, batch_size=1)
                    # ***
                    output_image_tensors = output_image_tensors.clamp(0,1)

                    numpy_image = output_image_tensors.cpu().permute(0, 2, 3, 1).detach().numpy()
                    numpy_image = (numpy_image * 255).astype(np.uint8)

                    img = Image.fromarray(numpy_image[0], 'RGB')
                    img.save(os.path.join(output_folder, file))
