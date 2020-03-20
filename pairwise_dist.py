"""Compute pairwise distances between images at input path"""

import csv
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def compute_distances(im1_path, im2_path):
    """
    Returns the distances between the two images at the paths im1_path and im2_path
    """
    im1 = process_im(im1_path)
    im2 = process_im(im2_path)

    euc = euclid_dist(im1, im2)
    kl = kl_dist(im1, im2)

    return euc, kl

def process_im(im_path):
    """
    Opens the image and converts it to 244x244 (network input size)
    The input path should lead to an image which is an internal network representaition
    """
    pil_im =  Image.open(im_path).convert('RGB')
    pil_im=pil_im.resize((224, 224))
    im_as_arr = np.float32(pil_im)
    pil_im.close()
    return im_as_arr

def euclid_dist(im1, im2):
    """
    Returns the Euclidean distance between the two images
    """
    return np.sum((im1-im2)**2)

def kl_dist(im1, im2):
    """
    Returns the KL divergence between the two images
    """
    eps = 0.00001
    im1 = im1 + eps
    im2 = im2 + eps
    return np.sum(np.where(im1 != 0, im1*np.log(im1/im2), 0))