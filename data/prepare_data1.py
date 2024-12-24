import glob
import os
import sys
import random
import imageio
import pickle
from natsort import natsorted
from tqdm import tqdm
from imresize import imresize
import numpy as np

def get_img_paths(dir_path, wildcard='*.png'):
    return natsorted(glob.glob(os.path.join(dir_path, wildcard)))

def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)

def to_pklv4(obj, path, verbose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if verbose:
        print(f"Wrote {path}")

def random_crop(img, size):
    h, w, c = img.shape
    h_start = np.random.randint(0, h - size)
    h_end = h_start + size
    w_start = np.random.randint(0, w - size)
    w_end = w_start + size
    return img[h_start:h_end, w_start:w_end]

def imread(img_path):
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    return img

def to_pklv4_1pct(obj, path, verbose):
    n = int(round(len(obj) * 0.01))
    path = path.replace(".", "_1pct.")
    to_pklv4(obj[:n], path, verbose=True)

def main(dir_path_1, dir_path_2=None):
    hrs = []
    lqs = []

    img_paths = get_img_paths(dir_path_1)
    for img_path in tqdm(img_paths):
        img = imread(img_path)
        for _ in range(47):
            crop = random_crop(img, 160)
            cropX4 = imresize(crop, scalar_scale=0.25)
            hrs.append(crop)
            lqs.append(cropX4)

    if dir_path_2 is not None:
        img_paths = get_img_paths(dir_path_2)
        for img_path in tqdm(img_paths):
            img = imread(img_path)
            for _ in range(47):
                crop = random_crop(img, 160)
                cropX4 = imresize(crop, scalar_scale=0.25)
                hrs.append(crop)
                lqs.append(cropX4)

    shuffle_combined(hrs, lqs)

    hrs_path = get_hrs_path(dir_path_1)
    to_pklv4(hrs, hrs_path, verbose=True)
    to_pklv4_1pct(hrs, hrs_path, verbose=True)

    lqs_path = get_lqs_path(dir_path_1)
    to_pklv4(lqs, lqs_path, verbose=True)
    to_pklv4_1pct(lqs, lqs_path, verbose=True)

def get_hrs_path(dir_path):
    base_dir = '.'
    name = os.path.basename(dir_path)
    hrs_path = os.path.join(base_dir, 'pkls', f'{name}.pklv4')
    return hrs_path

def get_lqs_path(dir_path):
    base_dir = '.'
    name = os.path.basename(dir_path)
    lqs_path = os.path.join(base_dir, 'pkls', f'{name}_X4.pklv4')
    return lqs_path

def shuffle_combined(hrs, lqs):
    combined = list(zip(hrs, lqs))
    random.shuffle(combined)
    hrs[:], lqs[:] = zip(*combined)

if __name__ == "__main__":
    try:
        dir_path_1 = sys.argv[1]
        dir_path_2 = sys.argv[2] if len(sys.argv) > 2 else None
        assert os.path.isdir(dir_path_1)
        main(dir_path_1, dir_path_2)
    except IndexError:
        print("Please provide at least one directory path.")