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

class Analysis():

    def __init__(self, gradient_path, partic_path):
        # Initialize the Analysis object with relevant paths
        self.gradientpath = gradient_path
        self.participantpath = partic_path

        # Create dictionary for the human and CNN scores 
        self.allscores_path = os.getcwd() + '/data_with_cnn_scores/'
        if not os.path.exists(self.allscores_path):
                os.mkdir(self.allscores_path)

        self.data_dict = {}     #   Format:
                                #     { participantID   : { 
                                #         'prompt_type' : <string>
                                #         'image_pairs' : <np array>
                                #         'resp_times'  : <np array>
                                #         'hscores'     : <np array>
                                #         'cnn_scores'  : <dictionary of scores> }
                                #     ...
                                #     }

        self.cnn_layers = set()

        # Create dictionary by score
        self.scores_dict = {}   #   Format:
                                #     { score (0-7)     : {
                                #         'image_pairs' : <np array>
                                #         'prompts'     : <np array>
                                #         'cnn_scores'  : <dictionary of scores> }
                                #     ...
                                #     }

        # Create dictionary by image pair
        self.image_dict = {}    #   Format:
                                #     { image_pair_key  : {
                                #         'scores'       : <np array of all human scores for that pair>
                                #         'prompts'     : <np array>
                                #         'cnn_scores'  : <dictionary of scores>    }
                                #     }

    def calc_cnn_scores(self, networks):
        """ 
        Computes the CNN scores for each of the image pairs in the participant data
        and stores them in the human score files
        Scores: 
            Euclidean and KL divergence scores for all of the networks/layers in 
            the list 'networks'
        """

        # Extract data from all participant data files
        columns_to_normalize = []
        for pfile in os.listdir(self.participantpath):

            # Open participant file as pandas dataframe
            pdata = pd.read_csv(self.participantpath + pfile)
            pdata.rename(columns={'alexnet_l3_euc':'alexnet_l8_euc'}, inplace=True)
            pdata.to_csv(self.participantpath + pfile, index=False)
            
            # Go through each network layer requested
            for net in networks:            # net format is <network>_<bio-group>_<cnn layer>
                layer = net.split('_')[-1]
                euclist = []
                kllist = []
                
                # Go through each image pair
                for _, row in pdata.iterrows():

                    # Parse row data
                    im1_name = row['leftIm'].replace('.jpg','')
                    im2_name = row['rightIm'].replace('.jpg','')

                    im1_path = self.gradientpath + net + '/' + im1_name + '_' + layer + '_filter0_Guided_BP_color.jpg'
                    im2_path = self.gradientpath + net + '/' + im2_name + '_' + layer + '_filter0_Guided_BP_color.jpg'

                    # Compute the distances for each image pair
                    euc, kl = compute_distances(im1_path, im2_path)
                    euclist.append(euc)
                    kllist.append(kl)
                    
                # Update the dataframe
                num = layer.split('layer')[-1]
                eucname = net.split('_')[0] + '_l' + num + '_euc'
                klname = net.split('_')[0] + '_l' + num + '_kl'

                if net != 'alexnet_aves_layer8':    
                    # All data files already have the scores for alexnet euc. layer 8
                    pdata[eucname] = euclist
                    columns_to_normalize.append(eucname)
                pdata[klname] = kllist
                columns_to_normalize.append(klname)

            # Save the file with the additional data
            pdata.to_csv(self.allscores_path + pfile, index=False)
        
        self.normalize(columns_to_normalize)

    def normalize(self, columns_to_normalize):
        ## Normalize the CNN scores in the data files
         
        # Get the min and max scores for each distance metric
        mins = np.inf*np.ones(len(columns_to_normalize))
        maxs = -1*np.inf*np.ones(len(columns_to_normalize))

        print('Normalizing Scores')
        for pfile in os.listdir(self.allscores_path):    
            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # Find min and max of each distance score column
            filemaxs = np.array(pdata.loc[:,columns_to_normalize].max())
            filemins = np.array(pdata.loc[:,columns_to_normalize].min())

            # Indices to replace with smaller/larger values
            idxmax = filemaxs > maxs
            idxmins = filemins < mins
            
            # Replace with smaller/larger values as appropriate
            maxs[idxmax] = filemaxs[idxmax]
            mins[idxmins] = filemins[idxmins]         

        # Calculate normalized score per column
        for pfile in os.listdir(self.allscores_path):    
            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # Normalize the appropriate columns
            scores = pdata.loc[:,columns_to_normalize]
            normScores = (scores-mins)/(maxs-mins)
            pdata.loc[:,columns_to_normalize] = normScores

            # Save new normalized scores
            pdata.to_csv(self.allscores_path + pfile, index=False)

    def parse_data(self):
        """
        Parses the participant data with associated network costs and stores the data into 
        a dictionary
        """

        # Go through all participant data files
        for pfile in os.listdir(self.allscores_path):    

            # Parse file name for information
            fname = pfile.split('_')
            ID = fname[0]
            self.data_dict[ID] = {}

            if 'birds' in fname[1]:
                self.data_dict[ID]['prompt'] = 'birds'
            else:
                self.data_dict[ID]['prompt'] = 'images'

            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # Store the human response times and scores as-is
            # Ignore the first 15 entries - these are from the practice round
            self.data_dict[ID]['resp_times'] = np.array(pdata['responseTime'])[15:]
            self.data_dict[ID]['hscores'] = np.array(pdata['userChoice'])[15:]   

            # Store the image pair key   
            pairs = []
            for row in pdata[0:2]:
                pairs.append(self.make_key(row))

            self.data_dict[ID]['image_pairs'] = np.array(pairs)[15:]

            # Store the inverted normalized CNN scores for each layer
            # Invert the scores so that a higher score means more similar
            self.data_dict[ID]['cnn_scores'] = {}
            for i in range(4,len(pdata.columns)):   # Go through all CNN data columns
                colname = pdata.columns[i]
                self.data_dict[ID]['cnn_scores'][colname] = 1-np.array(pdata[colname])[15:]  

                self.cnn_layers.add(colname)
              
    def make_scores_dict(self):
        """
        Create a dictionary of human scores with the format:
             { score (0-7)   : {
                 'image_pairs' : <list>
                 'resp_times'  : <list>
                 'prompts'     : <list>  
                 'cnn_scores'  : <dictionary of scores>  }
             ...
             }
        """

        # Initialize the dictionary
        for i in range(1,8):
            self.scores_dict[i] = { 'image_pairs' : [],
                                    'resp_times'  : [],
                                    'prompts'     : [],
                                    'cnn_scores'  : {}
                                    }

        # Go through all participant data files
        for pfile in os.listdir(self.allscores_path):    

            # Check the prompt for that file
            fname = pfile.split('_')
            if 'birds' in fname[1]:
                prompt = 'birds'
            else:
                prompt = 'images'

            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # For each row of data, add to the proper dictionary key:
            #       - the name of the image pair
            #       - the response time
            #       - the scores of all networks & layers provided
            for i, row in pdata.iterrows():
                s = row['userChoice']
                time = row['responseTime']
                pair = self.make_key(row)

                self.scores_dict[s]['image_pairs'].append(pair)
                self.scores_dict[s]['resp_times'].append(time)
                self.scores_dict[s]['prompts'].append(prompt)
                
                for i in range(4,len(pdata.columns)):   # Go through all CNN data columns
                    colname = pdata.columns[i]
                    cnn_score = row[colname]

                    if colname not in self.scores_dict[s]['cnn_scores'].keys():
                        self.scores_dict[s]['cnn_scores'][colname] = []
                    self.scores_dict[s]['cnn_scores'][colname].append(cnn_score)

    def make_image_dict(self):
        """
        Create a score dictionary with image pairs as keys with the following format:
        { image_pair_key  : {
            'scores'       : <np array of all human scores for that pair>
            'prompts'     : <np array>
            'cnn_scores'  : <dictionary of scores>    }
        }
        """

        # Go through all participant data files
        for pfile in os.listdir(self.allscores_path):  

            # Check the prompt for that file
            fname = pfile.split('_')
            if 'birds' in fname[1]:
                prompt = 'birds'
            else:
                prompt = 'images'

            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # For each row of data, create a new entry if it is a new image and 
            # include information on:
            #   - human scores
            #   - prompts (bird or image)
            #   - cnn scores for each cnn/layer provided
            for i, row in pdata.iterrows():
                pair = self.make_key(row)

                # Add to dictionary if not there yet
                if pair not in self.image_dict.keys():
                    self.image_dict[pair] = {   'scores'    : [],
                                                'prompts'   : [],
                                                'resp_times': [],
                                                'cnn_scores': {}
                                            }
                                        
                s = row['userChoice']
                time = row['responseTime']

                # Add human score, prompt, and reponse time to list of images
                self.image_dict[pair]['scores'].append(s)
                self.image_dict[pair]['prompts'].append(prompt)
                self.image_dict[pair]['resp_times'].append(time)

                # Add all the network scores to the dictionar
                for i in range(4,len(pdata.columns)):   # Go through all CNN data columns
                    colname = pdata.columns[i]
                    cnn_score = row[colname]
                    
                    self.image_dict[pair]['cnn_scores'][colname] = cnn_score
                
        print('There are', len(self.image_dict.keys()), ' image pairs')


    def make_key(self, row):
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


if __name__ == "__main__":
    participant_data_files = os.getcwd() + '/data/'
    gradientpath = os.getcwd() + '/gradients/'

    networks = ['alexnet_aves_layer8',      # The folders where you store gradient images
                'alexnet_aves_layer3',      # Format: <network>_<bio-group>_<cnn layer>
                'vgg_aves_layer0',
                'vgg_aves_layer3',
                'vgg_aves_layer14',
                'vgg_aves_layer28']         

    A = Analysis(gradientpath, participant_data_files)
    # A.calc_cnn_scores(networks)         # Only needs to be done once - this makes a copy of your data files with the additional calculations

    A.parse_data()
    A.make_scores_dict()
    A.make_image_dict()
        
    # Get distribution of human scores and store image pairs by score
    # (probability of each score)
    score_distro = []
    for s in A.scores_dict.keys():
        score_distro.append(1.*len(A.scores_dict[s]['image_pairs']))
    score_distro = np.array(score_distro)
    print('Unnormalized Score Distribution (Humans):', score_distro)
    score_distro_norm = score_distro/sum(score_distro)
    print('Normalized Score Distribution (Humans):', score_distro_norm)

    ###############################################################################
    ## Bar plot of number of human scores per category based on prompt
    ###############################################################################
    score_bird = np.zeros(7)
    score_image = np.zeros(7)
    for part in A.data_dict.keys():
        
        # Count the number of times each label was chosen
        if A.data_dict[part]['prompt'] == 'birds':
            for score in A.data_dict[part]['hscores']:
                score_bird[score-1] += 1
        else:
            for score in A.data_dict[part]['hscores']:
                score_image[score-1] += 1

    """
    plt.figure()
    plt.bar(np.arange(len(score_bird))-.25, score_bird/sum(score_bird),     width=.25, label='Bird Prompt',  color='b')
    plt.bar(np.arange(len(score_bird)),     score_image/sum(score_image),   width=.25, label='Image Prompt', color='r')
    plt.xticks(np.arange(len(score_bird))-.25, range(1,1+len(score_bird)))
    plt.title('Distribution of Human Scores by Prompt')
    plt.xlabel('Score (Least to Most Similar)')
    plt.ylabel('Frequency of Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    ###############################################################################
    ## Plot human-human agreement
    ############################################################################### 

    # TRUE AGREEMENT
    agree_true0 = np.zeros(7)
    agree_true1 = np.zeros(7)
    agree_true2 = np.zeros(7)

    total_true = np.zeros(7)

    ## Method 1, select 10,000 random images
    # for i in range(10000):
    #     # Randomly select an image
    #     im = np.random.choice(list(A.image_dict.keys()))
    #     scores = A.image_dict[im]['scores']

    #     if len(scores) >= 2:
    #         # Randomly select two scores from that image (if possible)
    #         rands = np.random.choice(scores, size=2, replace=False)

    #         # If the scores are the same, update distribution
    #         s1 = rands[0]
    #         s2 = rands[1]
    #         if s1 == s2:
    #             agree_true0[s1-1] += 1
    #         if abs(s1-s2) <= 1:         # One-off agreement
    #             agree_true1[s1-1] += 1
    #         if abs(s1-s2) <= 2:         # Two-off agreement
    #             agree_true2[s1-1] += 1
    #         total_true[s1-1] += 1

    ## Method 2, select 100 images per score and compare to another score of the same image
    for s1 in range(1,8):
        for i in range(100):
            
            # Randomly select image pair given that score
            im = np.random.choice(list(A.scores_dict[s1]['image_pairs']))

            # Randomly select another score for that same image pair
            scores = list(A.image_dict[im]['scores']).copy()
            if len(scores) < 2:
                continue

            scores.remove(s1)   # Make sure we're not counting agreement between the same person
            s2 = np.random.choice(scores)

            # Update distribution if they're the same
            if s1 == s2:
                agree_true0[s1-1] += 1
            if abs(s1-s2) <= 1:         # One-off agreement
                agree_true1[s1-1] += 1
            if abs(s1-s2) <= 2:         # Two-off agreement
                agree_true2[s1-1] += 1
            total_true[s1-1] += 1

    ## CHANCE AGREEMENT
    agree_chance = np.zeros(7)
    total_chance = np.zeros(7)
    # For each image, select a chance 'human' image pair score 
    for i in range(10000):

        # Draw 2 scores from the human score distribution
        s1 = np.random.choice(np.arange(1,8), p=score_distro_norm)
        s2 = np.random.choice(np.arange(1,8), p=score_distro_norm)

        # If the 'people' have the same score, update the distribution
        if s1 == s2:
            agree_chance[s1-1] += 1
        total_chance[s1-1] += 1

    ## Plot true vs chance agreement
    # x = np.arange(1., 8)
    # plt.figure()
    # plt.bar(x, agree_true/total_true, width=0.2, label='True agreement')
    # plt.bar(x+0.2, agree/total, width=0.2, label='Chance agreement')
    
    # plt.title('Human-Human Score Agreement')
    # plt.xlabel('Human Score (Least to Most Similar)')
    # plt.ylabel('Frequency of Human Agreement')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    ## Plot human-human agreement by prompt as a scatter plot
    agreement = np.zeros((7,7))
    
    # Go through all image pairs
    for img in A.image_dict.keys():
        # Check if the image pair was shown to someone for both prompts
        prompts = np.array(A.image_dict[img]['prompts'])
        scores = np.array(A.image_dict[img]['scores'])
        if 'birds' in prompts and 'images' in prompts:
            # Count each pair of scores
            idx_b = np.argwhere(prompts=='birds')
            idx_i = np.argwhere(prompts=='images')
            for b in idx_b:
                sb = scores[b]
                for i in idx_i:
                    si = scores[i]
                    agreement[sb-1, si-1] += 1
    
    agreement_normalized = agreement/sum(agreement)

    """
    # Make scatter plot
    xs = []
    ys = []
    size = []
    for x in range(7):
        for y in range(7):
            xs.append(x+1)
            ys.append(y+1)
            size.append(2000*agreement_normalized[x, y]) 

    plt.figure()
    plt.scatter(xs,ys,s=size)
    plt.title('Human-Human Agreement by Prompt')
    plt.xlabel('Bird Prompt')
    plt.ylabel('Image Prompt')
    plt.tight_layout()
    plt.show()
    """

    ## Plot agreement, one-off and two-off agreement
    """
    x = np.arange(1,8)
    plt.figure()
    plt.bar(x,      agree_true0/total_true, width=0.2, label='Exact Agreement')
    plt.bar(x+0.2,  agree_true1/total_true, width=0.2, label='Agree Within 1 Bin')
    plt.bar(x+0.4,  agree_true2/total_true, width=0.2, label='Agree Within 2 Bins')
    plt.title('Human-Human Agreement')
    plt.xlabel('Score (Least to Most Similar)')
    plt.ylabel('Frequency of Agreement')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """    

    ###############################################################################
    ## Plot human-network agreement
    ############################################################################### 
    scoretype = 'alexnet_l8_euc'

    # Continuously draw an image pair from the human score distribution and
    # count the agreement
    agree0_b = np.zeros(7) 
    agree0_i = np.zeros(7)
    agree1_b = np.zeros(7)
    agree1_i = np.zeros(7)
    agree2_b = np.zeros(7)
    agree2_i = np.zeros(7)

    total_b = np.zeros(7)
    total_i = np.zeros(7)
    for i in range(10000):
        # Randomly select a score to draw from
        s = np.random.choice(np.arange(1,8), p=score_distro_norm)
        
        # Randomly select an entry index (i.e. image pair)
        idx = np.random.choice(range(len(A.scores_dict[s]['prompts'])))

        # Get the prompt and CNN score
        prompt = A.scores_dict[s]['prompts'][idx]
        cnn_score = A.scores_dict[s]['cnn_scores'][scoretype][idx]
        cnn_bin = np.floor(7*cnn_score) + 1

        if prompt == 'birds':
            if cnn_bin == s:        # Scores match exactly
                agree0_b[s-1] += 1
            if abs(cnn_bin - s) < 2:
                agree1_b[s-1] += 1
            if abs(cnn_bin - s) < 3:
                agree2_b[s-1] += 1

            total_b[s-1] += 1

        else:   # Prompt is 'images'
            if cnn_bin == s:        # Scores match exactly
                agree0_i[s-1] += 1
            if abs(cnn_bin - s) < 2:
                agree1_i[s-1] += 1
            if abs(cnn_bin - s) < 3:
                agree2_i[s-1] += 1

            total_i[s-1] += 1

    x = np.arange(1., 8)
    
    """
    # Plot bird agreement
    plt.figure()
    plt.bar(x, agree0_b/total_b, width=0.2, label='Agree Exactly')
    plt.bar(x+0.2, agree1_b/total_b, width=0.2, label='Agree Within 1 Bin')
    plt.bar(x+0.4, agree2_b/total_b, width=0.2, label='Agree Within 2 Bins')

    plt.title('Score Agreement By Network & Layer - Bird Prompt')
    plt.xlabel('Human Score (Least to Most Similar)')
    plt.ylabel('Frequency of Network Agreement on This Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot image agreement
    plt.figure()
    plt.bar(x, agree0_i/total_i, width=0.2, label='Agree Exactly')
    plt.bar(x+0.2, agree1_i/total_i, width=0.2, label='Agree Within 1 Bin')
    plt.bar(x+0.4, agree2_i/total_i, width=0.2, label='Agree Within 2 Bins')

    plt.title('Score Agreement By Network & Layer - Image Prompt')
    plt.xlabel('Human Score (Least to Most Similar)')
    plt.ylabel('Frequency of Network Agreement on This Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    ###############################################################################
    ## Scatter plot of CNN scores vs participant scores, stratified by prompt type
    ###############################################################################
    # One plot per score type
    for scoretype in A.cnn_layers:

        # Congregate data for all participants
        xbird = []
        ximage = []
        ybird = []
        yimage = []

        # Split participant data and cnn data by prompt
        for part in A.data_dict.keys():

            x = list(A.data_dict[part]['hscores'])                      # Human Scores
            y = list(A.data_dict[part]['cnn_scores'][scoretype])      # Network Scores

            if A.data_dict[part]['prompt'] == 'birds':
                xbird += x
                ybird += y
            else:
                ximage += x
                yimage += y
        
        """
        plt.figure()
        plt.scatter(xbird, ybird, label='Bird Prompt', c='b', marker='x')
        plt.scatter(ximage, yimage, label='Image Prompt', c='r', marker='.')
        plt.title('Network vs Human Scores by Prompt')
        plt.xlabel('Human Scores (Least to Most Similar)')
        plt.ylabel('Normalized Network Scores (%s, Least to Most Similar)' %scoretype)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """

    ###############################################################################
    ## Bar plot of the distribution of scores between networks
    ############################################################################### 
    plt.figure()
    w = 0.1
    x = np.arange(1.0,8) - w*0.5*len(A.cnn_layers)/2

    # Plot the sc
    for i,scoretype in enumerate(sorted(A.cnn_layers)):

        cnn_bins = []
        for part in A.data_dict.keys():
            cnn_scores = np.array(A.data_dict[part]['cnn_scores'][scoretype])      # Network Scores
            
            # Convert the CNN scores to human score range (1-7)
            cnn_bins += list(np.floor(7.*cnn_scores) + 1.)
        cnn_bins = np.array(cnn_bins)
        
        # Count how many times the network chooses each bin
        y = []
        for s in range(1,8):
            y.append(len(np.argwhere(cnn_bins == s)))
        
        # Plot only the euclidean scores - can change this to see kl scores, all scores, etc.
        if 'euc' in scoretype:      
            plt.bar(x, np.array(y)/sum(y), width=w, label = scoretype)
            x += w

    """    
    plt.title('Score Distribution By Network & Layer - Euclidean Distance')
    plt.xlabel('Score (Human Scale, Least to Most Similar)')
    plt.ylabel('Frequency of Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    