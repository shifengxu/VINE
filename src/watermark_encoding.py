import os, gc, torch, time, argparse
from transformers import AutoTokenizer, CLIPTextModel
from stega_encoder_decoder import ConditionAdaptor
from model import make_1step_sched
from vine_turbo import VINE_Turbo, VAE_encode, VAE_decode, initialize_unet_no_lora, initialize_vae_no_lora
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms


def main(args, device):
    ### ============= load model =============
    noise_scheduler_1step = make_1step_sched(device)
    timesteps_val = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device=device).long()

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", use_fast=False,)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    fixed_a2b_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.unsqueeze(0).to(device))[0].detach()
    del text_encoder, tokenizer, fixed_a2b_tokens  # free up some memory
    gc.collect()
    torch.cuda.empty_cache()

    sec_encoder = ConditionAdaptor()
    sec_encoder.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'ConditionAdaptor.pth')))
    sec_encoder.to(device)
    sec_encoder.eval()

    unet = initialize_unet_no_lora()
    unet.requires_grad_(False)
    unet.eval()
    unet.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'UNet2DConditionModel.pth')))
    unet.to(device)

    vae_a2b = initialize_vae_no_lora()
    vae_a2b.requires_grad_(False)
    vae_a2b.eval()
    vae_a2b.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'vae.pth')))
    vae_a2b.to(device)
    vae_enc = VAE_encode(vae_a2b)
    vae_dec = VAE_decode(vae_a2b)
    print('\n =================== All Models Loaded Successfully ===================')

    ### ============= load image =============
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
    ])
    t_val_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC), 
    ])
    
    input_image_pil = Image.open(args.input_path).convert('RGB') # 512x512 
    resized_img = t_val_256(input_image_pil) # 256x256
    resized_img = 2.0 * resized_img - 1.0
    input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device) # 512x512
    input_image = 2.0 * input_image - 1.0
    resized_img = resized_img.unsqueeze(0).to(device)

    ### ============= load message =============
    if args.load_text: # text to bits
        if len(args.message) > 12:
            print('Error: Can only encode 100 bits (12 characters)')
            raise SystemExit
        data = bytearray(args.message + ' ' * (12 - len(args.message)), 'utf-8')
        packet_binary = ''.join(format(x, '08b') for x in data)
        watermark = [int(x) for x in packet_binary]
        watermark.extend([0, 0, 0, 0])
        watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)
    else: # random bits
        data = torch.randint(0, 2, (100,))
        watermark = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)

    ### ============= watermark encoding =============
    start_time = time.time()
    encoded_image_256 = VINE_Turbo.forward_with_networks(
        resized_img, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, 
        timesteps_val, fixed_a2b_emb_base, watermark, sec_encoder
    )
    end_time = time.time()
    print('\nEncoding time:', end_time - start_time, 's', '\n (Note that please execute multiple times to get the average time)\n')

    ### ============= resolution scaling to (512x512) =============
    residual_256 = encoded_image_256 - resized_img # 256x256
    residual_512 = t_val_512(residual_256) # 512x512
    encoded_image = residual_512 + input_image # 512x512
    encoded_image = encoded_image * 0.5 + 0.5
    encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

    ### ============= save the output image =============
    output_pil = transforms.ToPILImage()(encoded_image[0])
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    save_loc = os.path.join(args.output_dir, os.path.split(args.input_path)[-1][:-4]+'_wm.png')
    output_pil.save(save_loc)
    print(f'\nWatermarked image saved at: {save_loc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./example/input/2.png', help='path to the input image')
    parser.add_argument('--output_dir', type=str, default='./example/watermarked_img', help='the directory to save the output')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/VINE-R', help='path to the checkpoint')
    parser.add_argument('--message', type=str, default='Hello World!', help='the secret message to be encoded')
    parser.add_argument('--load_text', type=bool, default=True, help='the flag to decide to use inputed text or random bits')
    args = parser.parse_args()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    