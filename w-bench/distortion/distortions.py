import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch, io, os
from distortion_utils import set_random_seed, to_tensor, to_pil


distortion_strength_paras = dict(
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)


def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength


def apply_distortion(
    images,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
    return_image=True,
):
    # Convert images to PIL images if they are tensors
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)
    # Check if strength is relative and convert if needed
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply distortions
    distorted_images = []
    seed = distortion_seed
    for image in images:
        distorted_images.append(
            apply_single_distortion(
                image, distortion_type, strength, distortion_seed=seed
            )
        )
        # If not applying the same distortion, increment the seed
        if not same_operation:
            seed += 1
    # Convert to tensors if needed
    if not return_image:
        distorted_images = to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        assert False

    return distorted_image

if __name__ == '__main__':
    models = [
    # '01_MBRS', 
    # '02_CIN', 
    # '03_PIMoG', 
    # '04_RivaGAN', 
    # '05_SepMark', 
    # '07_RoSteALS', 
    # '08_DWTDCT', 
    # '09_DWTDCTSVD', 
    # '10_HIDDEN', 
    # '12_SSL', 
    # '14_RAW', 
    # '15_StegaStamp', 
    # '17_HIDDEN_CROP', 
    # '18_TrustMark_C',
    # '20_EditGuard', 
    # '21_VINE_R', 
    # '22_VINE_B'
    ]

    steps = {
        'resizedcrop': (0.5, 0.7, 0.9, 1.1, 1.3),
        'erasing': (0.05, 0.1, 0.15, 0.2, 0.25),
        'brightness': (1.2, 1.4, 1.6, 1.8, 2.0),
        'contrast': (1.2, 1.4, 1.6, 1.8, 2.0),
        'blurring': (1, 3, 5, 7, 9),
        'noise': (0.02, 0.04, 0.06, 0.08, 0.1),
        'compression': (10, 25, 40, 55, 70)
    }
    root = '/ntuzfs/shilin/Shilin/watermark/'
    for m in models:
        for c in ['INVERSION']:
            for i in os.listdir(os.path.join(root, m, '512', c)):
                path = os.path.join(root, m, '512', c, i)
                image = Image.open(path)
                #for distortion_type in distortion_strength_paras:
                for distortion_type in ['resizedcrop']:
                    for step in steps[distortion_type]:
                        save_path = os.path.join('/ntuzfs/shilin/Shilin/watermark/edited_result/distortion', m, '512', c, distortion_type, str(step))
                        os.makedirs(save_path, exist_ok=True)
                        image_edited = apply_single_distortion(image, distortion_type, strength=step)
                        image_edited = to_pil(to_tensor([image_edited], norm_type=None).clamp(0, 1), norm_type=None)[0]
                        assert np.min(np.array(image_edited)) >= 0 and np.max(np.array(image_edited)) <= 255
                        assert image_edited.size == (512, 512)
                        save_path = os.path.join(save_path, i)
                        print(save_path)
                        image_edited.save(save_path)