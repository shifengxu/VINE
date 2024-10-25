# Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances

<!-- [![arXiv](https://img.shields.io/badge/arXiv-TF--ICON-green.svg?style=plastic)](https://arxiv.org/abs/2307.12493) -->

[![arXiv](https://img.shields.io/badge/arXiv-VINE-green.svg?style=plastic)](https://arxiv.org/abs/2410.18775)

Official implementation of [Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances](https://arxiv.org/abs/2410.18775)

> **Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances**<br>
> Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong <br>
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

</div>

## Contents
  - [Setup](#setup)
    - [Creating a Conda Environment](#creating-a-conda-environment)
    - [Downloading VINE Checkpoints](#downloading-vine-checkpoints)
  - [Inference](#inference)
    - [Watermark Encoding](#watermark-encoding)
    - [Image Editing](#image-editing)
    - [Watermark Decoding](#watermark-decoding)
    - [Quality Metrics Calculation](#quality-metrics-calculation)
    - [Demo](#demo)
  - [Citation](#citation)


<br>

## Setup

### Creating a Conda Environment

```
git clone https://github.com/Shilin-LU/VINE.git
conda env create -f environment.yaml
conda activate vine
cd diffusers
pip install -e .
```

### Downloading VINE Checkpoints

Our models, VINE-B and VINE-R, have been released and are available for download [here](https://entuedu-my.sharepoint.com/:f:/g/personal/shilin002_e_ntu_edu_sg/Eow35WqqamtKojEB2oX1CiUBF7I-OQBioidUcj68wol-CA?e=O8MDR4). Please place them in the `./ckpt` folder.

## Inference

### Watermark Encoding
To encode a message into an image using VINE, please use the following commands:
```
python src/watermark_encoding.py  --ckpt_path ./ckpt/VINE-R              \
                                  --input_path ./example/input/2.png     \
                                  --output_dir ./example/watermarked_img \
                                  --message 'Hello World!'
                                
```
### Image Editing
We now offer [UltraEdit](https://github.com/HaozheZhao/UltraEdit) and Image Inversion for image editing, with more options to be added in the future. To edit an image, please use the following commands:
```
python src/image_editing.py  --model ultraedit                               \
                             --input_path ./example/watermarked_img/2_wm.png \
                             --output_dir ./example/edited_watermarked_img
                                
```
### Watermark Decoding
To decode a message from a watermarked image that has been edited, please use the following commands:
```
python src/watermark_decoding.py  --ckpt_path ./ckpt/VINE-R                                   \
                                  --input_path ./example/edited_watermarked_img/2_wm_edit.png \
                                  --groundtruth_message 'Hello World!'
                                
```
### Quality Metrics Calculation
To calculate the quality metrics for single image (PSNR, SSIM, and LPIPS), please use the following commands:
```
python src/quality_metrics.py   --input_path ./example/input/2.png \
                                --wmed_input_path ./example/watermarked_img/2_wm.png
                                
```

### Demo
We provide a complete demo that includes the processes of watermark encoding, image editing, watermark decoding, and quality metrics calculation in `./src/demo.ipynb`. Please refer to it for detailed instructions.

## Citation
If you find the repo useful, please consider citing