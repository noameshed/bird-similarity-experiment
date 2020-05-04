# Displays the images seen by both networks and people 
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
import torch
from PIL import Image
from torchvision import datasets, models, transforms
from tqdm import tqdm


def transform():
    # Transform for input images
    t = transforms.Compose([
        transforms.Resize(256),			# images should be 256x256
        transforms.CenterCrop(224),		# crop about the center to 224x224
        transforms.ToTensor(),			# convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    return t

def combine(im1, im2):
    # Combine two images horizontally (assumes the images are the same height)
    h = im1.height
    w = im1.width + im2.width
    imnew = Image.new('RGB', (w, h))
    imnew.paste(im1, (0,0))
    imnew.paste(im2, (im1.width, 0))

    return imnew

def network_bin(score):
    # Converts the network score (continuous) to a bin number (discrete) on the human scale
    cnn_bin = int(np.floor(7*(1-score)) + 1)
    # In the case that a network scores an 8 (perfect score), reduce to 7
    if cnn_bin == 8:
        cnn_bin = 7
    return cnn_bin

if __name__ == '__main__':

    # Get all images in the CSV
    img_path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Aves/'
    exp_data_path = 'C:/Users/noam_/Documents/Cornell/CS7999/bird-similarity-experiment/data_with_cnn_scores/'
    save_path = 'C:/Users/noam_/OneDrive/Cornell/CS7999/thesis/images/pairs_1/'

    size=(600,600)
    birdpairs = {}
    # Go through each participant's data file 
    for fname in os.listdir(exp_data_path):
        df = pd.read_csv(exp_data_path+fname)
        header = df.columns

        prompt = 'bird' if 'bird' in fname else 'image'

        for i, row in df.iterrows():
            # Skip the first 15 trials (practice round)
            if i  < 15:
                continue

            hscore = row['userChoice']
            for j in range(5,len(header)):
                val = row[j]
                layer = header[j]
                cnn_bin = network_bin(val)

                # Check if both scores are 7
                if (cnn_bin == 1) and (hscore == 1):

                    # Open both images
                    im1_path = os.path.join(img_path, row['leftIm'])
                    im2_path = os.path.join(img_path, row['rightIm'])

                    im1 = Image.open(im1_path)
                    im2 = Image.open(im2_path)

                    # Resize images to correct shape
                    im1_square = im1.resize(size, resample = Image.BILINEAR)
                    im2_square = im2.resize(size, resample = Image.BILINEAR)

                    im_combined = combine(im1_square, im2_square)

                    # fig = plt.imshow(im_combined)
                    # plt.axis('off')

                    # plt.show()

                    im_name = row['leftIm'][:-4].replace('/','_') + '_' + row['rightIm'][:-4].replace('/','_') + '.jpg'

                    if not os.path.exists(os.path.join(save_path, prompt, layer)):
                        os.mkdir(os.path.join(save_path, prompt, layer))

                    save_path_complete = os.path.join(save_path, prompt, layer, im_name)
                    im_combined.save(save_path_complete, dpi=(200,200))

                    # Save dictionary of birds with a score of 7 and the networks who gave them that score
                    if im_name not in birdpairs.keys():
                        birdpairs[im_name] = []
                    birdpairs[im_name].append(layer)
