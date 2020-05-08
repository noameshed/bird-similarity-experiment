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
    # Get the probability distributions of labels for the given images
    dist1 = extract_output_vec(basepath, im1_path)
    dist2 = extract_output_vec(basepath, im2_path)
    
    return kl_dist(dist1, dist2)

def extract_output_vec(basepath, im_path):
    """
    Get the network prediction (output vector) for the image
    at the given path
    """
    # Extract the species and image name information
    spec = im_path.split('/')[0] + '.json'  
    im = im_path.split('/')[1]

    if spec == 'Mimus gilvus.json':
        return(0)

     # Extract the probability distribution of labels for the given images
    with open(basepath + spec) as f1:
        data = json.load(f1)
        dist = np.array(data[im]['confs'])

    return dist

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
    try:
        return torch.sum((im1-im2)**2)
    except:
        return np.sum((im1-im2)**2)

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
