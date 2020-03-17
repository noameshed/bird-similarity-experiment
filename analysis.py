# Analyze human participant data
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from math import floor
from pairwise_dist import compute_distances

#TODO: Make this work when file has multiple score types
#TODO: Make this work for database of CNN scores, not just the scores local to the participant data files

def make_key(row):
    """
    Takes a row of image pair scores and converts to a key, assuming
    that the first two elements of the row are the bird names
    """
    img1 = row[0].split('_')[0]
    img2 = row[1].split('_')[0]

    # Remove '.jpg' if it is at the end of the image name
    if img1[-4:] == '.jpg':
        img1 = img1[:-4]
        img2 = img2[:-4]

    img1 = img1.replace('.jpg', '')
    img2 = img2.replace('.jpg', '')
    
    # Create dictionary key s.t. first name is first alphabetically
    if img1 < img2:
        key = img1 + '_' + img2
    else:
        key = img2 + '_' + img1
    return key

def get_cnn_score(im1_name, im2_name, base_impath):
    """
    Computes the distances scores for the two images on the 3rd, second to last, and output
    layers of AlexNet
    """

    im1_name = im1_name.replace('.jpg', '')
    im1_name = im1_name.replace('.jpg', '')

    im1_path_l3 = base_impath + 'Aves_layer3/' + im1_name + '_layer3_filter0_Guided_BP_color.jpg'
    im2_path_l3 = base_impath + 'Aves_layer3/' + im2_name + '_layer3_filter0_Guided_BP_color.jpg'
    euk3, kl3 = compute_distances(im1_path_l3, im2_path_l3)

    im1_path_l8 = base_impath + 'Aves_layer8/' + im1_name + '_layer8_filter0_Guided_BP_color.jpg'
    im2_path_l8 = base_impath + 'Aves_layer8/' + im2_name + '_layer8_filter0_Guided_BP_color.jpg'
    euk8, kl8 = compute_distances(im1_path_l8, im2_path_l8)

    return euk3, kl3, euk8, kl8

def parse_human_data(datapath, gradientpath):
    """
    Parses all of the experiment participant data and returns a dictionary where
    the key is a bird pair name and the value is a list of scores given to that pair by all participants
    """

    # Look at each participant's results
    human_scores = {}       # All human scores
    cnn_scores = {}
    counter = 0
    for fname in tqdm(os.listdir(datapath)):
        with open(datapath+fname, 'r') as f:
            # Read each row to get score for that bird
            reader = csv.reader(f)
            rownum = 0
            for row in reader:
                if rownum <= 15:     # Skip header row and practice round
                    rownum += 1
                    continue

                key = make_key(row)
                score = int(row[2])     # assumes score is in 3rd column of table
                cnn_score = float(row[3])   
                resp_time = float(row[4])    
            
                # Add the score to the dictionary
                if key not in human_scores.keys():
                    human_scores[key] = ([],[])
                human_scores[key][0].append(score)

                # Add the prompt to the dictionary
                if 'bird' in fname:
                    human_scores[key][1].append('bird')
                else:
                    human_scores[key][1].append('image')

                cnn_scores[key] = cnn_score
                counter += 1

    return cnn_scores, human_scores

def plot_bin_agreement(cnn_scores, human_scores):
    """
    Plots the 7 bins on the x axis and plots the percentage of the time which the humans agreed
    with the cnn score. 

    The human score for each image pair is computed as the average score of all people who rated
    that image pair. Note that individuals only had the option to rate an image pair an integer score 1-7

    Agreement score (per bin):
        (times human and cnn awarded that bin score)/(times cnn gave that bin score)
    """

    agree0 = [0,0,0,0,0,0,0]
    agree1 = [0,0,0,0,0,0,0]
    agree2 = [0,0,0,0,0,0,0]
    cnn_choice = [0,0,0,0,0,0,0]        # How many times did the CNN choose each bin? Should be roughly equal
    human_choice = [0,0,0,0,0,0,0]
    avg_hscores = []
    all_cscores = []            # All of the CNN scores to match all of the human scores
    all_hscores = []
    all_cscores_unique = []     # Each of the CNN scores once 
    pairs = []
    prompts = []

    people_per_pair = []
    for key in cnn_scores.keys():
        
        # Retrieve cnn scores - originally between 0-1 - and convert to be between 0-6
        # CNN scores are originally inverted from human scores, so subtract from 1
        norm_cnn_score = 1-cnn_scores[key]
        all_cscores_unique.append(norm_cnn_score)
        cnn_score = 7*norm_cnn_score
        
        cnn_bin = floor(cnn_score)      # Round CNN score to a bin number
        

        # Retrieve the human score list and compute the average score for this image pair
        avg_human_score = sum(human_scores[key][0])/len(human_scores[key][0])
        norm_human_score = avg_human_score/7
        avg_hscores.append(norm_human_score)
        
        people_per_pair.append(len(human_scores[key][0]))

        pairs.append(key)

        # Count how many times humans agree with the network within a margin of error
        for i, score in enumerate(human_scores[key][0]):
            score -= 1  # To make it from 0-6
            prompts.append(human_scores[key][1][i])
            all_cscores.append(norm_cnn_score)
            all_hscores.append(score)

            human_choice[score] += 1
            cnn_choice[cnn_bin] += 1
            # Depending on which bin the CNN put the label in, compute the human agreement...
            if score == cnn_bin:    # ...within the same bin
                agree0[cnn_bin] += 1
            
            if abs(score-cnn_bin) <= 1:     # ...within 1 bin on either side
                agree1[cnn_bin] += 1

            if abs(score-cnn_bin) <= 2:      # ...within 2 bins on either side
                agree2[cnn_bin] += 1


    print('On average, %f people saw each image pair.' %(sum(people_per_pair)/len(people_per_pair)))

    all_cscores = np.array(all_cscores)
    all_hscores = np.array(all_hscores)
    prompts = np.array(prompts)
    
    # Plot the two image score distributions
    plt.figure()
    xs = range(len(pairs))
    sort_idx = np.argsort(avg_hscores)
    plt.plot(xs, np.array(all_cscores)[sort_idx], 'r-', linewidth=0.25, label='CNN Scores')
    plt.plot(xs, np.array(avg_hscores)[sort_idx], 'b-', linewidth=0.5, label='Human Scores')
    
    plt.title('Similarity Scores Over All Image Pairs')
    plt.legend()
    plt.xlabel('Image Pair')
    plt.ylabel('Similarity Score (0-1)')

    plt.tight_layout()
    plt.show()

    # Scatter plot of human vs CNN scores
    plt.figure()
    idx_birds = np.argwhere(prompts=='bird')
    idx_image = np.argwhere(prompts=='image')
    
    plt.scatter(all_hscores[idx_birds], all_cscores[idx_birds], c='r', marker='x', label='Bird Prompt')
    plt.scatter(all_hscores[idx_image], all_cscores[idx_image], c='b', marker='.', label='Image Prompt')

    plt.title('CNN vs Human Scores - All Image Pairs')
    plt.legend()
    plt.xlabel('Human Scores, Least to Most Similar')
    plt.ylabel('CNN Scores, Least to Most Similar')

    plt.tight_layout()
    plt.show()

    # Plot the distribution of cnn bins 
    plt.figure()
    xs = range(len(cnn_choice))
    plt.bar(xs, cnn_choice)
    plt.title('Distribution of CNN score bins')
    plt.xlabel('Bin number (i.e. Similarity Score) Least to Most Similar')
    plt.ylabel('Number of image pairs')

    plt.tight_layout()
    plt.show()

    # Plot the distribution of human scores
    plt.figure()
    xs = range(len(human_choice))
    plt.bar(xs, human_choice)
    plt.title('Distribution of Human Scores')
    plt.xlabel('Bin number (i.e. Similarity Score) Least to Most Similar')
    plt.ylabel('Number of image pairs')

    plt.tight_layout()
    plt.show()

    # Plot the agreement between the CNN and people
    plt.figure()
    w=0.25
    xs0 = np.arange(len(cnn_choice))-w
    xs1 = [x+w for x in xs0]
    xs2 = [x+w for x in xs1]
    plt.bar(xs0, np.array(agree0)/np.array(cnn_choice), width=w, label='Agree Exactly')
    plt.bar(xs1, np.array(agree1)/np.array(cnn_choice), width=w, label='Agree Within 1 Bin')
    plt.bar(xs2, np.array(agree2)/np.array(cnn_choice), width=w, label='Agree Within 2 Bins')
    
    plt.title('Agreement Between People and AlexNet (Euclidean Dist)')
    plt.legend()
    plt.xlabel('Similarity Score (Least to Most Similar)')
    plt.ylabel('% Agreement')

    plt.tight_layout()
    plt.show()


#TODO: Is there a difference between human responses in the 2 prompts?

if __name__ == "__main__":
    participant_data_files = os.getcwd() + '/data/'
    gradientpath = 'D:/noam_/Cornell/CS7999/iNaturalist/gradients/'     # Path where the images are stored

    # TODO: Compute inter-participant agreement
    cnn_scores, human_scores = parse_human_data(participant_data_files, gradientpath) # Load human scores
    plot_bin_agreement(cnn_scores, human_scores)
