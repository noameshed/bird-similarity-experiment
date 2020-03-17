# Analyze human participant data
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from math import floor

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
    
    # Create dictionary key s.t. first name is first alphabetically
    if img1 < img2:
        key = img1 + '_' + img2
    else:
        key = img2 + '_' + img1
    return key

def cnn_score_lookup_dict(all_pairs_path, experiment_path, scoretype='euclid'):
    """
    Creates a dictionary for looking up the similarity of an image pair based on the CNN score provided
    Dictionary keys are the names of the two images separated by _ and in alphabetical order
    Dictionary values are the pair distance (default 'euclid', but can also choose 'kl' )
    """

    # Make a list of all pairs in the experiment
    allpairs = []
    for fname in os.listdir(experiment_path):
        with open(experiment_path + fname,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                key, img1, img2 = make_key(row)
                allpairs.append(key)

    allpairs = set(allpairs)
    cnn_scores = {}
    for pair in allpairs:
        img1 = pair.split('_')[0]
        img2 = pair.split('_')[1]
        
        species1 = img1.split('/')[0] + '.csv'

        print(img1, img2)

        with open(all_pairs_path+species1,'r') as f:

            df = pd.read_csv(f)
            #img1_list = df['image1']
            #img2_list = df['image2']
            
            found_col1 = df[df['image1'].str.contains(img1)]
            found_col2 = df[df['image2'].str.contains(img2)]

            print(found_col1.empty, found_col2.empty)
            # if not found_col1.empty:
            #     # image1 is in the first column - find the second image in the second column
            #     row = found_col1[found_col1['image2'].str.contains(img2)]
            #     print(row)
            # elif not found_col2.empty:
            #     # image1 is in the first column - find the second image in the second column
            #     row = found_col2[found_col2['image2'].str.contains(img2)]
            #     print(row)
            # else:
            #     # The image is not found anywhere - this should not happen
            #     print('NOT FOUND',found_col1.empty, found_col2.empty)


            # for idx, row in df.iterrows():
            #     if img1 in row['image1']:
            #         print('FOUND', row)




    # Now add the image pairs used in the experiment to the dictionary

    """
    cnn_scores = {}
    for fname in tqdm(os.listdir(all_pairs_path)):
        with open(all_pairs_path+fname, 'r') as f:
            reader = csv.reader(f)
            rownum = 0
            for row in reader:
                if rownum == 0:     # Skip header row
                    rownum += 1
                    continue

                key = make_key(row)
                scoreidx = 2 + row.index(scoretype) # Find which column the normalized score is in
                score = float(row[scoreidx])
                
                # Add information to dictionary if needed
                if key in allpairs:
                    cnn_scores[key] = score
    """
    return cnn_scores

def parse_human_data(datapath):
    """
    Parses all of the experiment participant data and returns a dictionary where
    the key is a bird pair name and the value is a list of scores given to that pair by all participants
    """

    # Look at each participant's results
    human_scores = {}
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
                    human_scores[key] = []
                human_scores[key].append(score)
                cnn_scores[key] = cnn_score
                counter += 1

    return human_scores, cnn_scores

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
    all_hscores = []
    all_cscores = []  
    pairs = []

    people_per_pair = []

    for key in cnn_scores.keys():
        
        # Retrieve cnn scores - originally between 0-1 - and convert to be between 0-6
        # CNN scores are originally inverted from human scores, so subtract from 1
        norm_cnn_score = 1-cnn_scores[key]
        all_cscores.append(norm_cnn_score)
        cnn_score = 7*norm_cnn_score
        
        cnn_bin = floor(cnn_score)      # Round CNN score to a bin number
        

        # Retrieve the human score list and compute the average score for this image pair
        avg_human_score = sum(human_scores[key])/len(human_scores[key])
        norm_human_score = avg_human_score/7
        all_hscores.append(norm_human_score)
        people_per_pair.append(len(human_scores[key]))

        pairs.append(key)

        # Count how many times humans agree with the network within a margin of error
        for score in human_scores[key]:
            score -= 1  # To make it from 0-6

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

    # Plot the two image score distributions
    plt.figure()
    xs = range(len(pairs))
    sort_idx = np.argsort(all_hscores)
    plt.plot(xs, np.array(all_cscores)[sort_idx], 'r-', linewidth=0.25, label='CNN Scores')
    plt.plot(xs, np.array(all_hscores)[sort_idx], 'b-', linewidth=0.5, label='Human Scores')
    
    plt.title('Similarity Scores Over All Image Pairs')
    plt.legend()
    plt.xlabel('Image Pair')
    plt.ylabel('Similarity Score (0-1)')

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
    cnn_score_files = os.getcwd() + '/alexnet_hiddenlayer_distances/alexnet_layer8_dists/'
    experiment_pair_files = os.getcwd() + '/stratified_img_pairs/'
    participant_data_files = os.getcwd() + '/data/'
    savename = 'cnn_score_lookup_alexnetlayer8.json'

    # Create a dictionary of the CNN scores
    #cnn_scores = cnn_score_lookup_dict(cnn_score_files, experiment_pair_files)

    # Save cnn scores as json
    #with open ('cnn_score_lookup_alexnetlayer8.json', 'w') as fp:
    #    json.dump(cnn_scores, fp, sort_keys=True, indent=4)
    
    # with open(os.getcwd() + '/cnn_score_lookup_alexnetlayer8.json') as jsonfile:
    #     cnn_scores = json.load(jsonfile)        # Load CNN scores
    #     human_scores, cnn_scores = parse_human_data(participant_data_files) # Load human scores

    #     #print(len(cnn_scores.keys()), len(human_scores.keys()))       # should be 992, <=992

    #     # Plot agreement between average participants and cnn
    #     plot_bin_agreement(cnn_scores, human_scores)

    #     # TODO: Compute inter-participant agreement

    human_scores, cnn_scores = parse_human_data(participant_data_files) # Load human scores
    plot_bin_agreement(cnn_scores, human_scores)
