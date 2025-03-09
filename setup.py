from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

class CustomInstallCommand(install):
    def run(self):
        # 首先运行标准的安装
        install.run(self)
        # 切换到 diffusers 目录并执行 editable 安装
        diffusers_dir = os.path.join(os.path.dirname(__file__), 'diffusers')
        if os.path.exists(diffusers_dir):
            subprocess.check_call(['pip', 'install', '-e', diffusers_dir])
        else:
            print("Warning: diffusers directory not found. Please run 'cd diffusers && pip install -e .' manually.")

setup(
    name="vine",
    version="0.1.0",
    description="A Python package for vine project",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Shilin Lu",
    author_email="shilin002@e.ntu.edu.sg",
    url="https://github.com/Shilin-LU/VINE",
    packages=find_packages(where='vine'),
    install_requires=[
        "einops==0.8.0",
        "numpy==1.26.4",
        "open-clip-torch==2.26.1",
        "opencv-python==4.6.0.66",
        "pillow==10.4.0",
        "scipy==1.11.1",
        "timm>=0.9.2",
        "tokenizers==0.19.1",
        "torch==2.0.1",
        "torchdata==0.6.1",
        "torchmetrics==0.2.0",
        "torchvision==0.15.2",
        "tqdm>=4.65.0",
        "triton==2.0.0",
        "urllib3==1.26.19",
        "xformers==0.0.20",
        "streamlit-keyup==0.2.0",
        "lpips",
        "clean-fid",
        "peft==0.11.1",
        "dominate",
        "gradio==3.43.1",
        "transformers==4.43.3",
        "accelerate==0.30.0",
        "datasets==2.20.0",
        "pilgram",
        "kornia",
        "wandb",
        "scikit-image",
        "scikit-learn",
        "ipykernel",
        "sentencepiece",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    },
)