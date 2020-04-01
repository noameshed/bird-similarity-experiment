import json
import numpy as np
import os
import pandas as pd
from pairwise_dist import compute_distances

class Analysis():
    # This class is made to analyse the human participant data from the bird image similarity experiments

    def __init__(self, gradient_path, partic_path):
        # Initialize the Analysis object with relevant paths
        self.gradientpath = gradient_path
        self.participantpath = partic_path

        # Create dictionary for the human and CNN scores 
        self.allscores_path = os.getcwd() + '/data_with_cnn_scores/'
        if not os.path.exists(self.allscores_path):
                os.mkdir(self.allscores_path)

        self.partic_dict = {}     #   Format:
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
                                #         'scores'      : <np array of all human scores for that pair>
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

    def make_partic_dict(self):
        """
        Parses the participant data with associated network costs and stores the data into 
        a dictionary
        """

        # Go through all participant data files
        for pfile in os.listdir(self.allscores_path):    

            # Parse file name for information
            fname = pfile.split('_')
            ID = fname[0]
            self.partic_dict[ID] = {}

            if 'birds' in fname[1]:
                self.partic_dict[ID]['prompt'] = 'birds'
            else:
                self.partic_dict[ID]['prompt'] = 'images'

            # Open file as pandas dataframe
            pdata = pd.read_csv(self.allscores_path + pfile)

            # Store the human response times and scores as-is
            # Ignore the first 15 entries - these are from the practice round
            self.partic_dict[ID]['resp_times'] = np.array(pdata['responseTime'])[15:]
            self.partic_dict[ID]['hscores'] = np.array(pdata['userChoice'])[15:]   

            # Store the image pair key   
            pairs = []
            for row in pdata[0:2]:
                pairs.append(self.make_key(row))

            self.partic_dict[ID]['image_pairs'] = np.array(pairs)[15:]

            # Store the normalized CNN scores for each layer
            self.partic_dict[ID]['cnn_scores'] = {}
            for i in range(4,len(pdata.columns)):   # Go through all CNN data columns
                colname = pdata.columns[i]
                self.partic_dict[ID]['cnn_scores'][colname] = np.array(pdata[colname])[15:]  

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
                # Skip the first 15 rows (practice round)
                if i < 15:
                    continue
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
                # Skip the first 15 rows (practice round)
                if i < 15:
                    continue
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
                
        print('There are', len(self.image_dict.keys()), 'image pairs')

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