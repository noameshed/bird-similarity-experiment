"""Compute pairwise distances between images at input path"""

import json
import numpy as np
import torch

def compute_euc(im1_path, im2_path):
    """
    Returns the euclidean distance between the two feature representations
    at the paths im1_path and im2_path
    """
    ft1 = process_tensor(im1_path)
    ft2 = process_tensor(im2_path)

    euc = euclid_dist(ft1, ft2)

    return euc.item()

def compute_kl(basepath, im1_path, im2_path):
    """
    Returns the KL divergence between the two label probabilities
    at the paths for the two images using the network at the given path
    """
    # Extract the species and image name information
    spec1 = im1_path.split('/')[0] + '.json'  
    im1 = im1_path.split('/')[1]

    spec2 = im2_path.split('/')[0] + '.json'  
    im2 = im2_path.split('/')[1]

    if spec1 == 'Mimus gilvus.json' or spec2 == 'Mimus gilvus.json':
        return(0)

    # Extract the probability distribution of labels for the given images
    with open(basepath + spec1) as f1:
        try:
            data = json.load(f1)
            dist1 = np.array(data[im1]['confs'])
        except:
            print(spec1, im1)

    with open(basepath + spec2) as f2:
        try:
            data = json.load(f2)
            dist2 = np.array(data[im2]['confs'])
        except:
            print(spec2, im2)
    
    return kl_dist(dist1, dist2)


def process_tensor(im_path):
    """
    Opens the image and converts it to 244x244 (network input size)
    The input path should lead to an image which is an internal network representaition
    """
    tensor = torch.load(im_path)
    return tensor

def euclid_dist(im1, im2):
    """
    Returns the Euclidean distance between the two images
    """
    return torch.sum((im1-im2)**2)

def kl_dist(im1, im2):
    """
    Returns the symmetric KL divergence between the two images
    """
    eps = 0.00001
    im1 = im1 + eps
    im2 = im2 + eps

    # log(0) is undefined
    kl12 = np.sum(np.where(im1 != 0, im1*np.log(im1/im2), 0))
    kl21 = np.sum(np.where(im2 != 0, im2*np.log(im2/im1), 0))
    return kl12 + kl21