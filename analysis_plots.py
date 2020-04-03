
import numpy as np
import os
import matplotlib.pyplot as plt
from analysis import Analysis

def human_score_distro(A, plot=False):

	# Compute normalized distribution of human scores
	hscore_distro = []
	for s in A.scores_dict.keys():
		hscore_distro.append(1.*len(A.scores_dict[s]['image_pairs']))
	hscore_distro = np.array(hscore_distro)
	print('Unnormalized Score Distribution (Humans):', hscore_distro)
	hscore_distro = hscore_distro/sum(hscore_distro)
	print('Normalized Score Distribution (Humans):', hscore_distro)

	###############################################################################
	## Bar plot of number of human scores overall
	###############################################################################
	
	if plot:
		plt.figure()
		plt.bar(np.arange(1,8.), hscore_distro, width=.25, color='b')
		plt.title('Distribution of Human Scores')
		plt.xlabel('Score (Least to Most Similar)')
		plt.ylabel('Frequency of Score')
		plt.tight_layout()
		plt.show()

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
	
	if plot:
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
	
	return hscore_distro

def network_score_distro(A, plot=False):
	###############################################################################
	## Bar plot of the distribution of network scores
	############################################################################### 

	# Compute normalized distribution of scores of each network
	cnnscore_distro = {}
	for net in A.cnn_layers:
		# Initialize empty distribution 
		cnnscore_distro[net] = np.zeros(7)

		# Update the distribution with each image pair score
		for impair in A.image_dict.keys():
			# Invert score so that higher scores are 'more similar'
			cnn_score = 1. - A.image_dict[impair]['cnn_scores'][net]

			# Convert to discrete score 1-7
			cnn_bin = int(np.floor(7*cnn_score) + 1)
			if cnn_bin == 8:
				cnn_bin -= 1
			cnnscore_distro[net][cnn_bin-1] += 1

		# Normalize scores
		cnnscore_distro[net] = cnnscore_distro[net]/np.sum(cnnscore_distro[net])

	if plot:

		plt.figure()
		w = 0.1
		x = np.arange(1.0,8) - w*0.5*len(A.cnn_layers)/2

		# Plot the score distribution for the networks
		for i,net in enumerate(sorted(A.cnn_layers)):
			# Plot only the euclidean scores - can change this to see kl scores, all scores, etc.
			if 'euc' in net:  
				y = cnnscore_distro[net]
					
				plt.bar(x, np.array(y), width=w, label = net)
				x += w

		plt.title('Score Distribution By Network & Layer - Euclidean Distance')
		plt.xlabel('Score (Human Scale, Least to Most Similar)')
		plt.ylabel('Frequency of Score')
		plt.legend()
		plt.tight_layout()
		plt.show()

	return cnnscore_distro
		
def human_human_agreement(A, hscore_distro):
		###############################################################################
	## Plot human-human agreement
	############################################################################### 

	# TRUE AGREEMENT
	agree_true0 = np.zeros(7)
	agree_true1 = np.zeros(7)
	agree_true2 = np.zeros(7)

	total_true = np.zeros(7)

	## Method 1, select 10,000 random images
	
	for _ in range(10000):
	    # Randomly select an image
	    im = np.random.choice(list(A.image_dict.keys()))
	    scores = A.image_dict[im]['scores']

	    if len(scores) >= 2:
	        # Randomly select two scores from that image (if possible)
	        rands = np.random.choice(scores, size=2, replace=False)

	        # If the scores are the same, update distribution
	        s1 = rands[0]
	        s2 = rands[1]
	        if s1 == s2:
	            agree_true0[s1-1] += 1
	        if abs(s1-s2) <= 1:         # One-off agreement
	            agree_true1[s1-1] += 1
	        if abs(s1-s2) <= 2:         # Two-off agreement
	            agree_true2[s1-1] += 1
	        total_true[s1-1] += 1
	

	## Method 2, select 100 images per score and compare to another score of the same image
	"""
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
	"""

	## CHANCE AGREEMENT
	agree_chance = np.zeros(7)
	total_chance = np.zeros(7)
	# For each image, select a chance 'human' image pair score 
	for i in range(10000):

		# Draw 2 scores from the human score distribution
		s1 = np.random.choice(np.arange(1,8), p=hscore_distro)
		s2 = np.random.choice(np.arange(1,8), p=hscore_distro)

		# If the 'people' have the same score, update the distribution
		if s1 == s2:
			agree_chance[s1-1] += 1
		total_chance[s1-1] += 1

	# Plot true vs chance agreement
	x = np.arange(1., 8)
	plt.figure()
	plt.bar(x, agree_true0/total_true, width=0.2, color='r',label='True agreement')
	plt.bar(x+0.2, agree_chance/total_chance, width=0.2, color='b',label='Chance agreement')
	
	plt.title('Inter-Participant Agreement')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()

	## Plot exact, one-off, and two-off agreement
	x = np.arange(1,8)
	plt.figure()
	plt.bar(x,      agree_true0/total_true, width=0.2, label='Exact Agreement')
	plt.bar(x+0.2,  agree_true1/total_true, width=0.2, label='Agree Within 1 Bin')
	plt.bar(x+0.4,  agree_true2/total_true, width=0.2, label='Agree Within 2 Bins')
	plt.title('Inter-Participant Agreement Within 1 or 2 Points')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()

def human_human_pairs(A):
	## Plot frequency of human-human score pairs
	allscores = np.zeros((7,7))
	
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
					allscores[sb-1, si-1] += 1
	
	allscores_normalized = allscores/sum(allscores)
	
	# Plot
	xs = []
	ys = []
	size = []
	colors = []
	for x in range(7):
		for y in range(7):
			xs.append(x+1)
			ys.append(y+1)
			size.append(2000*allscores_normalized[x, y]) 
			if x == y:
				colors.append('r')
			else:
				colors.append('b')

	plt.figure()
	plt.scatter(xs,ys,s=size, c=colors)
	plt.title('Image Scores by Prompt')
	plt.xlabel('Bird Prompt Scores')
	plt.ylabel('Image Prompt Scores')
	plt.tight_layout()
	plt.show()
	
def human_network_agreement_combined(A, hscore_distro):
	###############################################################################
	## Plot human-network agreement - all networks together
	############################################################################### 

	# plt.figure()
	x = np.arange(1.,8)
	w = 0.05

	agree_b = np.zeros(7)
	total_b = np.zeros(7)
	agree_i = np.zeros(7)
	total_i = np.zeros(7)
	for scoretype in sorted(A.cnn_layers):

		agree = np.zeros(7)
		total = np.zeros(7)
		# Plot agreement per network
		for _ in range(10000):
			# Randomly select score from human distribution
			s = np.random.choice(np.arange(1,8), p=hscore_distro)
			
			# Randomly select an entry index (i.e. image pair)
			idx = np.random.choice(range(len(A.scores_dict[s]['prompts'])))

			# Get the prompt and CNN score. Invert and discretize CNN score
			prompt = A.scores_dict[s]['prompts'][idx]
			cnn_score = 1 - A.scores_dict[s]['cnn_scores'][scoretype][idx]
			cnn_bin = int(np.floor(7*cnn_score) + 1)

			if cnn_bin == 8:	# Edge case where the network score was highest possible
				cnn_bin -=	1

			if cnn_bin == s:        # Scores match exactly
				agree[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 2:
				agree[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 3:
				agree[cnn_bin-1] += 1
			total[cnn_bin-1] += 1

			if prompt == 'birds':
				if cnn_bin == s:        # Scores match exactly
					agree_b[cnn_bin-1] += 1
				total_b[cnn_bin-1] += 1

			else:			# Prompt is 'images'
				if cnn_bin == s:        # Scores match exactly
					agree_i[cnn_bin-1] += 1
				total_i[cnn_bin-1] += 1

		# Plot this network's agreement scores
		plt.bar(x, agree/total, width=w, label=scoretype)
		x += w

	plt.title('Human-Network Agreement (P(H ^ N | N))')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	plt.figure()
	x = np.arange(1.,8)
	plt.bar(x, agree_b/total_b, width=0.25, label='Bird Prompt')
	plt.bar(x+0.25, agree_i/total_i, width=0.25, label='Image Prompt')
	plt.title('Human-Network Agreement By Prompt')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
def human_network_agreement_separate(A, hscore_distro):
	###############################################################################
	## Plot human-network agreement - one network at a time
	############################################################################### 

	scoretype = 'vgg_l28_kl'
	network_description = 'VGG16 L28 KL'

	# TRUE AGREEMENT
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
		# Randomly select score from human distribution
		s = np.random.choice(np.arange(1,8), p=hscore_distro)
		
		# Randomly select an entry index (i.e. image pair)
		idx = np.random.choice(range(len(A.scores_dict[s]['prompts'])))

		# Get the prompt and CNN score
		prompt = A.scores_dict[s]['prompts'][idx]
		cnn_score = 1 - A.scores_dict[s]['cnn_scores'][scoretype][idx]
		cnn_bin = int(np.floor(7*cnn_score) + 1)
		if cnn_bin == 8:	# Edge case where the network score was highest possible
			cnn_bin -=	1

		if prompt == 'birds':
			if cnn_bin == s:        # Scores match exactly
				agree0_b[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 2:
				agree1_b[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 3:
				agree2_b[cnn_bin-1] += 1

			total_b[cnn_bin-1] += 1

		else:   # Prompt is 'images'
			if cnn_bin == s:        # Scores match exactly
				agree0_i[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 2:
				agree1_i[cnn_bin-1] += 1
			if abs(cnn_bin - s) < 3:
				agree2_i[cnn_bin-1] += 1

			total_i[cnn_bin-1] += 1
		

	x = np.arange(1., 8)

	# Plot bird agreement over a few bins
	plt.figure()
	total = total_b + total_i
	plt.bar(x, (agree0_b+agree0_i)/(total_i+total_b), width=0.2, label='Agree Exactly')
	plt.bar(x+0.2, (agree1_b+agree1_i)/(total_i+total_b), width=0.2, label='Agree Within 1 Bin')
	plt.bar(x+0.4, (agree2_b+agree2_i)/(total_i+total_b), width=0.2, label='Agree Within 2 Bins')

	plt.title('Human-' + network_description + ' Agreement')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()

	# CHANCE AGREEMENT
	agree_chance = np.zeros(7)
	total_chance = np.zeros(7)
	# For each image, select a chance 'human' image pair score 
	for i in range(10000):

		# Draw 2 scores from the score distributions
		s1 = np.random.choice(np.arange(1,8), p=hscore_distro)					# Human
		s2 = np.random.choice(np.arange(1,8), p=cnnscore_distro[scoretype])		# Network
				# cnnscore_distro is already inverted to be from least to most similar

		# If the person and network have the same score, update the distribution
		if s1 == s2:
			agree_chance[s2-1] += 1
		total_chance[s2-1] += 1



	## Plot true vs chance agreement
	x = np.arange(1., 8)
	plt.figure()
	plt.bar(x, (agree0_b+agree0_i)/(total_b+total_i), width=0.2, label='True agreement')
	plt.bar(x+0.2, agree_chance/total_chance, width=0.2, label='Chance agreement')
	
	plt.title('Human-' + network_description + ' Agreement (P(H ^ N | N))')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	## Plot human-network agreement by prompt 
	plt.figure()
	plt.bar(x, agree0_b/total_b, width=0.2, label='Bird Prompt')
	plt.bar(x+0.2, agree0_i/total_i, width=0.2, label='Image Prompt')
	plt.bar(x+0.4, agree_chance/total_chance, width=0.2, label='Agree By Chance')

	plt.title('Human-' + network_description + ' Agreement By Prompt')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Human Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()	
	
def human_network_pairs(A):
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
			y = list(1 - A.partic_dict[part]['cnn_scores'][scoretype])      # Network Scores (inverted)

			if A.partic_dict[part]['prompt'] == 'birds':
				xbird += x
				ybird += y
			else:
				ximage += x
				yimage += y
		
		# Add some jitter to the scores to make them easier to see
		xbird += np.random.uniform(low=-.25, high=0.25, size=len(xbird))
		ximage += np.random.uniform(low=-.25, high=0.25, size=len(ximage))

		plt.figure()
		plt.scatter(xbird, ybird, label='Bird Prompt', c='b', marker='x')
		plt.scatter(ximage, yimage, label='Image Prompt', c='r', marker='.')
		plt.title('Network (' + scoretype +') vs Human Scores by Prompt')
		plt.xlabel('Human Scores (Least to Most Similar)')
		plt.ylabel('Normalized Network Scores (Least to Most Similar)')
		plt.legend()
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	participant_data_files = os.getcwd() + '/data/'
	feature_path = os.getcwd() + '/features/'
	network_labels_path = 'C:/Users/noam_/Documents/Cornell/CS7999/novelty-detection/alexnet_inat_results/Aves/'

	networks = ['alexnet_conv_1',
				'alexnet_conv_2',
				'alexnet_conv_3',
				'alexnet_conv_4',
				'alexnet_conv_5',
				'vgg16_block_1',
				'vgg16_block_2',
				'vgg16_block_3',
				'vgg16_block_4',
				'vgg16_block_5'
				]         

	A = Analysis(feature_path, participant_data_files, network_labels_path)
	A.calc_cnn_scores(networks)         # Only needs to be done once  -
										# this makes a copy of your data files with the normalized network distance scores
	"""
	A.make_partic_dict()
	A.make_scores_dict()
	A.make_image_dict()

	###############################################################################
	## All plots
	############################################################################### 
	
	# Score distributions
	hscore_distro = human_score_distro(A, plot=False)
	cnnscore_distro = network_score_distro(A, plot=False)

	# Human-human agreement and trends
	# human_human_pairs(A)
	# human_human_agreement(A, hscore_distro)

	# Human-network agreement and trends
	# human_network_agreement_combined(A, hscore_distro)
	human_network_agreement_separate(A, hscore_distro)
	# human_network_pairs(A)

	"""