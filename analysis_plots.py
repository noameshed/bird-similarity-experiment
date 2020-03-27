
import numpy as np
import os
import matplotlib.pyplot as plt
from analysis import Analysis

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

    A.make_partic_dict()
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
    for part in A.partic_dict.keys():
        
        # Count the number of times each label was chosen
        if A.partic_dict[part]['prompt'] == 'birds':
            for score in A.partic_dict[part]['hscores']:
                score_bird[score-1] += 1
        else:
            for score in A.partic_dict[part]['hscores']:
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
        for part in A.partic_dict.keys():

            x = list(A.partic_dict[part]['hscores'])                      # Human Scores
            y = list(A.partic_dict[part]['cnn_scores'][scoretype])      # Network Scores

            if A.partic_dict[part]['prompt'] == 'birds':
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
        for part in A.partic_dict.keys():
            cnn_scores = np.array(A.partic_dict[part]['cnn_scores'][scoretype])      # Network Scores
            
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
    