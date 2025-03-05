import torch, argparse, os
from diffusers import DDIMScheduler
from utils_sto_regeneration import DiffWMAttacker, ReSDPipeline


# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
WMs = ["vine"]    # todo *** (WMs)
for WM in WMs:
    for STEPS in [220, 240]:# [60, 80, 100, 120, 140]:, 220, 240
        DEVICE = 'cuda:0'   # todo *** (CUDA)
        INPUT_PATH = f"/home/shilin1/projs/datasets/{WM}_encoded/512/STO_REGENERATION_1K"   # todo *** (INPUT)
        OUTPUT_PATH = f"/home/shilin1/projs/datasets/edited_image/{WM}/STO_REGENERATION_1K/{STEPS}"   # todo *** (OUTPUT)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

        #print(f"\n* Processing WM={WM}")
        print(f"\n* Processing STEPS={STEPS}")
        parser = argparse.ArgumentParser()
        parser.add_argument("--wm_image_path", type=str, default=None)
        parser.add_argument("--wm_images_folder", type=str, default=INPUT_PATH)
        parser.add_argument("--wm_attacked_folder", type=str, default=OUTPUT_PATH)
        parser.add_argument("--noise_step", type=int, default=STEPS)
        args = parser.parse_args()
        device = DEVICE

        pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        pipe.to(device)
        print('Finished loading model')

        attackers = {
            'diff_attacker_60': DiffWMAttacker(pipe, batch_size=5, noise_step=args.noise_step, captions={}),
        }

        os.makedirs(args.wm_attacked_folder, exist_ok=True)
        if args.wm_image_path is not None:
            wm_img_paths = [args.wm_image_path]
            image_name = os.path.basename(args.wm_image_path)
            att_img_paths = [os.path.join(args.wm_attacked_folder, f'zhao23_{args.noise_step}' + image_name)]
        else:
            wm_images_folder = args.wm_images_folder
            wm_attacked_folder = args.wm_attacked_folder

            image_names = sorted(os.listdir(wm_images_folder))
            wm_img_paths = [os.path.join(wm_images_folder, filename) for filename in image_names]
            att_img_paths = [os.path.join(wm_attacked_folder, filename) for filename in image_names]


        for attacker_name, attacker in attackers.items():
            attackers[attacker_name].attack(wm_img_paths, att_img_paths)

        print('Finished attacking')
        