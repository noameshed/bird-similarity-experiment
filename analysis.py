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
        self.allscores_path = partic_path + 'with_cnn_scores/'
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
                                #     { score (0-7)   : {
                                #         'image_pairs' : <np array>
                                #         'prompts'     : <np array>
                                #         'cnn_scores'  : <dictionary of scores> }
                                #     ...
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
                layer = net[-1]
                euclist = []
                kllist = []
                
                # Go through each image pair
                for _, row in pdata.iterrows():

                    # Parse row data
                    im1_name = row['leftIm'].replace('.jpg','')
                    im2_name = row['rightIm'].replace('.jpg','')

                    im1_path = self.gradientpath + net + '/' + im1_name + '_layer' + layer + '_filter0_Guided_BP_color.jpg'
                    im2_path = self.gradientpath + net + '/' + im2_name + '_layer' + layer + '_filter0_Guided_BP_color.jpg'

                    # Compute the distances for each image pair
                    euc, kl = compute_distances(im1_path, im2_path)
                    euclist.append(euc)
                    kllist.append(kl)
                    
                # Update the dataframe
                eucname = net.split('_')[0] + '_l' + layer + '_euc'
                klname = net.split('_')[0] + '_l' + layer + '_kl'

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
    #gradientpath = 'D:/noam_/Cornell/CS7999/iNaturalist/gradients/'     # Path where the images are stored
    gradientpath = os.getcwd() + '/gradients/'

    networks = ['vgg_aves_layer0',         # The folders where you store gradient images
                'alexnet_aves_layer8']         # Format: <network>_<bio-group>_<cnn layer>

    # TODO: Compute inter-participant agreement
    A = Analysis(gradientpath, participant_data_files)
    #A.calc_cnn_scores(networks)         # Only needs to be done once - this makes a copy of your data files with the additional calculations

    A.parse_data()
    A.make_scores_dict()

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
        
        
        plt.figure()
        plt.scatter(xbird, ybird, label='Bird Prompt', c='b', marker='x')
        plt.scatter(ximage, yimage, label='Image Prompt', c='r', marker='.')
        plt.title('Network vs Human Scores by Prompt')
        plt.xlabel('Human Scores (Least to Most Similar)')
        plt.ylabel('Normalized Network Scores (%s, Least to Most Similar)' %scoretype)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
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

    ###############################################################################
    ## Bar plot of the agreement between humans and CNN
    ############################################################################### 
    scoretype = 'alexnet_l8_euc'

    # Get distribution of human scores and store image pairs by score
    # (probability of each score)
    score_distro = []
    for s in A.scores_dict.keys():
        score_distro.append(1.*len(A.scores_dict[s]['image_pairs']))
    print(score_distro)
    score_distro = np.array(score_distro)
    score_distro /= sum(score_distro)
    print(score_distro)


    # Continuously draw an image pair from the human score distribution and
    # count the agreement
    agree0 = np.zeros(7)
    agree1 = np.zeros(7)
    agree2 = np.zeros(7)
    total = np.zeros(7)
    for i in range(10000):
        s = np.random.choice(np.arange(1,8), p=score_distro)
        
        cnn_score = np.random.choice(A.scores_dict[s]['cnn_scores'][scoretype])
        cnn_bin = np.floor(7*cnn_score) + 1

        if cnn_bin == s:        # Scores match exactly
            agree0[s-1] += 1
        if abs(cnn_bin - s) < 2:
            agree1[s-1] += 1
        if abs(cnn_bin - s) < 3:
            agree2[s-1] += 1

        total[s-1] += 1

    print(agree0/total)

    x = np.arange(1., 8)
    
    plt.figure()
    #print(agree0/sum(agree0))
    plt.bar(x, agree0/total, width=0.2, label='Agree Exactly')
    plt.bar(x+0.2, agree1/total, width=0.2, label='Agree Within 1 Bin')
    plt.bar(x+0.4, agree2/total, width=0.2, label='Agree Within 2 Bins')

    plt.title('Score Agreement By Network & Layer')
    plt.xlabel('Human Score (Least to Most Similar)')
    plt.ylabel('Frequency of Network Agreement on This Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    ###############################################################################
    ## Bar plot of the distribution of scores between networks
    ############################################################################### 
    plt.figure()
    x = np.arange(1.0,8) - 0.3
    margin = 0.05
    width = (1.-2.*margin)/7
    for i,scoretype in enumerate(A.cnn_layers):

        cnn = []
        for part in A.data_dict.keys():     # TODO: FIX THIS! IT IS WRONG
            cnn += [A.data_dict[part]['cnn_scores'][scoretype]]      # Network Scores

        # Convert the CNN scores to human score range (1-7)
        cnn_bins = np.floor(7*cnn) + 1

        # Count how many times the network chooses each bin
        y = []
        for i in range(7):
            y.append(len(np.argwhere(cnn_bins == i+1)))

        
        plt.bar(x, np.array(y)/sum(y), width=.2, label = scoretype)
        x += 0.2

    
    plt.title('Score Distribution By Network & Layer')
    plt.xlabel('Score (Human Scale, Least to Most Similar)')
    plt.ylabel('Frequency of Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
