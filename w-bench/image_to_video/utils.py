import os
import shutil
import re
from tqdm import tqdm


def cluster_svd_images():
    source_folder = f''
    target_folder = f''

    small_folders = ["5", "7", "9", "11", "13", "15", "17", "19"]
    for folder in small_folders:
        os.makedirs(os.path.join(target_folder, folder), exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            number = str(int(filename.split('_')[-1].split('.')[0]))

            if number in small_folders:
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, number, filename.split('_')[0] + "_" + filename.split('_')[1] + '.png')
                shutil.copy(src_path, dst_path)
    print(f'ALL SVD Images Are Copied -> `{target_folder}`')


if __name__ == "__main__":
    cluster_svd_images()
