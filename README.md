# [ICLR 2025] Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances

<!-- [![arXiv](https://img.shields.io/badge/arXiv-TF--ICON-green.svg?style=plastic)](https://arxiv.org/abs/2307.12493) -->

[![arXiv](https://img.shields.io/badge/arXiv-VINE-green.svg?style=plastic)](https://arxiv.org/abs/2410.18775) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue.svg?style=plastic)](https://huggingface.co/Shilin-LU) [![HuggingFace](https://img.shields.io/badge/HuggingFace-W--Bench-red.svg?style=plastic)](https://huggingface.co/datasets/Shilin-LU/W-Bench)

Official implementation of [Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances](https://arxiv.org/abs/2410.18775)

> **Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances**<br>
> Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong <br>
> ICLR 2025
> 
>**Abstract**: <br>
Current image watermarking methods are vulnerable to advanced image editing techniques enabled by large-scale text-to-image models. These models can distort embedded watermarks during editing, posing significant challenges to copyright protection. In this work, we introduce W-Bench, the first comprehensive benchmark designed to evaluate the robustness of watermarking methods against a wide range of image editing techniques, including image regeneration, global editing, local editing, and image-to-video generation. Through extensive evaluations of eleven representative watermarking methods against prevalent editing techniques, we demonstrate that most methods fail to detect watermarks after such edits. To address this limitation, we propose VINE, a watermarking method that significantly enhances robustness against various image editing techniques while maintaining high image quality. Our approach involves two key innovations: (1) we analyze the frequency characteristics of image editing and identify that blurring distortions exhibit similar frequency properties, which allows us to use them as surrogate attacks during training to bolster watermark robustness; (2) we leverage a large-scale pretrained diffusion model SDXL-Turbo, adapting it for the watermarking task to achieve more imperceptible and robust watermark embedding. Experimental results show that our method achieves outstanding watermarking performance under various image editing techniques, outperforming existing methods in both image quality and robustness.

---

</div>

![teaser](assets/teaser.png)

---

</div>

![framework](assets/sdxl_encoder.png)

---

<br>

</div>

## News
- [Mar 02, 2025] 😊 We are releasing W-Bench! It will be completed within the next two weeks. Stay tuned!

- [Jan 23, 2025] 🥳 Vine is accepted by ICLR 2025 ([OpenReview](https://openreview.net/forum?id=16O8GCm8Wn))!

- [Oct 24, 2024] 🚀 We release the checkpoints of Vine along with our technical report on [arXiv](https://arxiv.org/abs/2410.18775)!

<br>

</div>

## Contents
  - [Setup](#setup)
    - [Creating a Conda Environment](#creating-a-conda-environment)
    - [Downloading VINE Checkpoints](#downloading-vine-checkpoints)
  - [Demo](#demo)
  - [Inference](#inference)
    - [Watermark Encoding](#watermark-encoding)
    - [Image Editing](#image-editing)
    - [Watermark Decoding](#watermark-decoding)
    - [Quality Metrics Calculation](#quality-metrics-calculation)
  - [W-Bench](#w\-bench)
    - [Regeneration](#regeneration)
    - [Global Editing](#global-editing)
    - [Local Editing](#local-editing)
    - [Image to Video](#image-to-video)
  - [Citation](#citation)


<br>

## Setup

### Creating a Conda Environment

```shell
git clone https://github.com/Shilin-LU/VINE.git
cd VINE
conda env create -f environment.yaml
conda activate vine
cd diffusers
pip install -e .
```

Note that when editing images using MagicBrush and SVD, the environment should use the specific environments listed in their respective sections below. In all other cases, the **vine** environment is sufficient to run all code, including watermark encoding, decoding, regeneration, local editing, and other global editing tasks.

### Downloading VINE Checkpoints

Our models, VINE-B and VINE-R, have been released on HuggingFace ([VINE-B-Enc](https://huggingface.co/Shilin-LU/VINE-B-Enc), [VINE-B-Dec](https://huggingface.co/Shilin-LU/VINE-B-Dec), [VINE-R-Enc](https://huggingface.co/Shilin-LU/VINE-R-Enc), [VINE-R-Dec](https://huggingface.co/Shilin-LU/VINE-R-Dec)) and are also available for download from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/shilin002_e_ntu_edu_sg/Eow35WqqamtKojEB2oX1CiUB1URh40K1xaFp-NsGPa2VBw?e=YCrnJo). 

<br>

## Demo
We provide a complete demo that includes the processes of watermark encoding, image editing, watermark decoding, and quality metrics calculation in `./src/demo.ipynb`. Please refer to it for detailed instructions.

<br>

## Inference

### Watermark Encoding
To encode a message into an image using VINE, please use the following commands:
```shell
python src/watermark_encoding.py  --pretrained_model_name Shilin-LU/VINE-R-Enc \
                                  --input_path ./example/input/2.png           \
                                  --output_dir ./example/watermarked_img       \
                                  --message 'Hello World!'
```

To apply a watermark across the entire W-Bench, please use the following commands:
```shell
python src/watermark_encoding_wbench.py  --pretrained_model_name Shilin-LU/VINE-R-Enc \
                                         --input_dir /path/to/downloaded/W-Bench      \
                                         --output_dir /path/to/output/folder          \
                                         --message 'Your Secret'
```

### Image Editing
A basic example of using [UltraEdit](https://github.com/HaozheZhao/UltraEdit) and Image Inversion for image editing. To edit an image, please use the following commands:
```shell
python src/image_editing.py  --model ultraedit                               \
                             --input_path ./example/watermarked_img/2_wm.png \
                             --output_dir ./example/edited_watermarked_img      
```

To apply other image editing methods listed in W-Bench to a group of images, please refer to [W-Bench](#w\-bench)

### Watermark Decoding
To decode a message from a watermarked image that has been edited, please use the following commands:
```shell
python src/watermark_decoding.py  --pretrained_model_name Shilin-LU/VINE-R-Dec                \
                                  --input_path ./example/edited_watermarked_img/2_wm_edit.png \
                                  --groundtruth_message 'Hello World!'                    
```

To decode all watermarked images, please use the following commands:
```shell
python src/watermark_decoding_wbench.py  --pretrained_model_name Shilin-LU/VINE-R-Dec  \
                                         --groundtruth_message 'Your Secret'           \
                                         --unwm_images_dir /path/to/downloaded/W-Bench \
                                         --wm_images_dir /path/to/watermarked/folder   \
                                         --output_dir /path/to/output/folder 
```

### Quality Metrics Calculation
To calculate the quality metrics for single image (PSNR, SSIM, and LPIPS), please use the following commands:
```shell
python src/quality_metrics.py   --input_path ./example/input/2.png \
                                --wmed_input_path ./example/watermarked_img/2_wm.png                   
```

### Decoding Accuracy Metrics Calculation
A simple implementation for calculating statistical decoding metrics, such as TPR@0.1% FPR, TPR@1% FPR, and AUROC, is available in [this issue](https://github.com/Shilin-LU/VINE/issues/4#issuecomment-2467342137). The full codebase will be released alongside the W-Bench.

<br>

## W-Bench
W-Bench is the first benchmark to evaluate watermarking robustness across four types of image editing techniques, including [regeneration](#regeneration), [global editing](#global-editing), [local editing](#local-editing), and [image-to-video generation](#image-to-video). 11 representative watermarking methods are evaluated on the W-Bench. The W-Bench contains 10,000 samples sourced from datasets such as COCO, Flickr, ShareGPT4V, etc.

The images of W-Bench have been released on [HuggingFace](https://huggingface.co/datasets/Shilin-LU/W-Bench) and are also available on [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/shilin002_e_ntu_edu_sg/EkJ9AIBUNglEt3sRKIBNA9oBI1BNoz2IEj9iizh4uKF-3Q?e=stTbpM). Below is a detailed guide on how to use all the image editing techniques listed in W-Bench.

### Regeneration

#### 1. Stochastic Regeneration

#### 2. Deterministic Regeneration (aka, Image Inversion)

### Global Editing

#### 1. UltraEdit

#### 2. InstructPix2Pix

#### 3. MagicBrush
```shell
# Creating the Environment for MagicBrush
cd w-bench/global_editing
git clone https://github.com/timothybrooks/instruct-pix2pix.git
cd instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p

# Download the MagicBrush Checkpoint
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/osunlp/InstructPix2Pix-MagicBrush/resolve/main/MagicBrush-epoch-52-step-4999.ckpt
```

### Local Editing

#### 1. UltraEdit

#### 2. ControlNet-Inpainting

### Image to Video
```shell
# Creating the Environment for SVD
conda create -n svd python=3.8.5
conda activate svd
cd w-bench/image_to_video/generative-models
pip3 install -r requirements/pt2.txt

# Download the SVD Checkpoint
mkdir checkpoints # path: ./w-bench/image_to_video/generative-models/checkpoints
cd checkpoints
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --repo-type=model --local-dir svd_xt

# Alternatively, you may use the script to download: ./w-bench/image_to_video/generative-models/download_svd_ckpt.py
```
<br>

## Citation
If you find the repo useful, please consider citing:
```
@inproceedings{
  lu2025robust,
  title={Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances},
  author={Shilin Lu and Zihan Zhou and Jiayou Lu and Yuanzhi Zhu and Adams Wai-Kin Kong},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=16O8GCm8Wn}
}
```
