import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr

from analysis import Analysis
from PIL import Image


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
		plt.bar(np.arange(len(score_bird))-.5, score_bird/sum(score_bird),     width=.25, label='Bird Prompt',  color='r')
		plt.bar(np.arange(len(score_bird))-0.25,     score_image/sum(score_image),   width=.25, label='Image Prompt', color='b')
		plt.bar(np.arange(len(score_bird)), hscore_distro, width=.25, label='Average', color='mediumorchid')
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

	# Plot the true network distribution, not the binned one calculated above
	if plot:
		colors = ['red', 'green','blue']
		for i,net in enumerate(['alexnet', 'vgg16', 'resnet']):
			# plt.figure()
			x = []
			y = []

			# Collect all scores for all image pairs
			for p in A.image_dict.keys():
				
				for layer in A.image_dict[p]['cnn_scores']:

					if net in layer:
						num = layer.split('_')[-1]		# network name is <net>_<block/conv>_<num>
						if num == 'output':
							num = 6
						x.append(1 - A.image_dict[p]['cnn_scores'][layer])	# invert score
						y.append(int(num))

			# Compute average lines
			x = np.array(x)
			y = np.array(y)

			avgs = np.zeros(6)
			for j in range(1,7):
				idx = np.argwhere(y==j)
				avgs[j-1] = np.sum(x[idx])/len(x[idx])

			y = list(y)
			y += np.random.uniform(low=-.2, high=0.2, size=len(y))
			plt.scatter(x, y, c=colors[i], s=1)

			# Plot average lines
			for k, a in enumerate(avgs):
				plt.plot([a, a], [k+0.5, k+1.5], c='black')
				

			netname = net.split('_')[0].capitalize()
			plt.title('Score Distribution for ' + netname, fontsize=14)
			plt.xlabel('Score (Least to Most Similar)', fontsize=12)
			plt.ylabel('Network Layer', fontsize=12)
			plt.yticks(ticks=np.arange(6)+1, labels=['1','2','3','4','5','output'], fontsize=10)
			# plt.legend()
			plt.tight_layout()
			plt.show()

	return cnnscore_distro

def response_time(A):
	# Plot people's response times as the trial goes on

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	xs = np.arange(1,316)
	avg_ys = np.zeros(315)
	# Loop through all participants
	for fname in os.listdir(A.participant_path):
		pdata = pd.read_csv(A.participant_path+fname)
		ys = pdata['responseTime']#[15:]		# Skip the 15 question practice round
		avg_ys += ys

		plt.plot(xs,ys, c='silver', linewidth=0.5)

	# Plot average time
	avg_ys = avg_ys/len(os.listdir(A.participant_path))
	plt.plot(xs, avg_ys, c='red')

	# Plot lines of best fit
	plt.plot(np.unique(xs[:15]), np.poly1d(np.polyfit(xs[:15],avg_ys[:15],1))(np.unique(xs[:15])), 'b')	
	plt.plot(np.unique(xs[15:]), np.poly1d(np.polyfit(xs[15:],avg_ys[15:],1))(np.unique(xs[15:])), 'b')

	# Plot vertical line after 15 questions
	plt.plot([15, 15],[0,40], 'b-.', linewidth=0.75)
	plt.text(17, 30, s='Practice Round End', fontsize=10)
	ax.labelsize = 12
	plt.title('Participant Response Times Throughout the Trial', fontsize=15)
	plt.xlabel('Question Number (1-300)', fontsize=15)
	plt.ylabel('Response Time (s)', fontsize=15)
	plt.tight_layout()
	plt.show()

def human_human_pairs(A):
	## Plot frequency of human-human score pairs
	allscores = np.zeros((7,7))
	
	birdscores = []
	imagescores = []
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
					allscores[sb[0]-1, si[0]-1] += 1

					# Track all pairs of points for correlation score
					birdscores.append(sb[0])
					imagescores.append(si[0])

	# Compute correlation between bird and image prompt
	# corr = pearsonr(birdscores, imagescores)
	# print('Correlation:', corr)	
	# print(allscores)
	# print(np.sum(allscores, axis=1))
	# allscores_normalized = allscores/np.sum(allscores, axis=0)
	allscores_normalized = (allscores.T/np.sum(allscores, axis=1)).T	#np.sum(allscores) gets sum of entire matrix
							# axis=0 gives bird sums (vertical)
							# axis=1 gives image sums (horizontal)
	print(allscores_normalized)
	
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
	plt.title('Conditional Probability of Bird Scores Given Image Scores')
	plt.xlabel('Bird Prompt Scores')
	plt.ylabel('Image Prompt Scores')
	plt.tight_layout()
	plt.show()

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

def human_network_agreement_combined(A, hscore_distro):
	###############################################################################
	## Plot human-network agreement - all networks together
	############################################################################### 

	# plt.figure()
	
	w = 0.05
	agree_b = np.zeros(7)
	total_b = np.zeros(7)
	agree_i = np.zeros(7)
	total_i = np.zeros(7)
	for net in ['alexnet','vgg16','resnet']:
		# plt.figure()
		x = np.arange(1.,8) - w*(3)
		for scoretype in sorted(A.cnn_layers):
			if net in scoretype and 'euc' not in scoretype:
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
					# if abs(cnn_bin - s) < 2:
					# 	agree[cnn_bin-1] += 1
					# if abs(cnn_bin - s) < 3:
					# 	agree[cnn_bin-1] += 1
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
		netname = net.capitalize()
		plt.title('Human-'+netname+' Agreement (P(H ^ N | N))', fontsize=12)
		plt.xlabel('Score (Least to Most Similar)', fontsize=10)
		plt.ylabel('Frequency of Human Agreement', fontsize=10)
		plt.legend()
		plt.tight_layout()
		plt.show()
	
	plt.figure()
	x = np.arange(1.,8)
	plt.bar(x, agree_b/total_b, width=0.25, color='r', label='Bird Prompt')
	plt.bar(x+0.25, agree_i/total_i, width=0.25, color='b', label='Image Prompt')
	plt.title('Human-Network Agreement By Prompt (Avg over all networks)')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Frequency of Agreement')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
def human_network_agreement_separate(A, hscore_distro, species_data):
	###############################################################################
	## Plot human-network agreement - one network at a time
	############################################################################### 

	avg = np.zeros(7)
	avg_in = np.zeros(7)
	total_in = np.zeros(7)
	avg_notin = np.zeros(7)
	total_notin = np.zeros(7)
	avg_chance_novelty = np.zeros(7)
	avg_chance_novelty_total = np.zeros(7)

	avg_chance = np.zeros(7)
	avg_total = np.zeros(7)
	avg_chance_total = np.zeros(7)

	# Initialize info about class novelty for network
	imagenet = np.array(species_data['in imagenet'])
	cub = np.array(species_data['in cub'])
	species = np.array(species_data['name'])

	# Only look at species not in CUB training set
	good_idx = np.argwhere(cub == 'No')
	# good_idx = np.arange(len(imagenet))
	
	in_imagenet = species[np.argwhere(imagenet[good_idx] == 'Yes')][:,0]
	notin_imagenet = species[np.argwhere(imagenet[good_idx] == 'No')][:,0]

	for scoretype in sorted(A.cnn_layers):

		namesplit = scoretype.split('_')
		namesplit[0] = namesplit[0].capitalize()
		network_description = ' '.join(namesplit)

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

		for i in range(1000):
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
					avg[cnn_bin-1] += 1
				if abs(cnn_bin - s) < 2:
					agree1_b[cnn_bin-1] += 1
				if abs(cnn_bin - s) < 3:
					agree2_b[cnn_bin-1] += 1

				total_b[cnn_bin-1] += 1

			else:   # Prompt is 'images'
				if cnn_bin == s:        # Scores match exactly
					agree0_i[cnn_bin-1] += 1
					avg[cnn_bin-1] += 1
				if abs(cnn_bin - s) < 2:
					agree1_i[cnn_bin-1] += 1
				if abs(cnn_bin - s) < 3:
					agree2_i[cnn_bin-1] += 1

				total_i[cnn_bin-1] += 1
			avg_total[cnn_bin-1] += 1


			# Check for novelty
			imnames = A.scores_dict[s]['image_pairs'][idx]
			im1 = imnames.split('_')[0]
			im2 = imnames.split('_')[1]

			spec1 = im1.split('/')[0]
			spec2 = im2.split('/')[0]

			if spec1 in in_imagenet and spec2 in in_imagenet:
				if cnn_bin == s:
					avg_in[cnn_bin-1] += 1
				total_in[cnn_bin-1] += 1
			elif spec1 in notin_imagenet and spec2 in notin_imagenet:
				if cnn_bin == s:
					avg_notin[cnn_bin-1] += 1
				total_notin[cnn_bin-1] += 1
			

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
		for i in range(1000):

			# Draw 2 scores from the score distributions
			s1 = np.random.choice(np.arange(1,8), p=hscore_distro)					# Human
			s2 = np.random.choice(np.arange(1,8), p=cnnscore_distro[scoretype])		# Network
					# cnnscore_distro is already inverted to be from least to most similar

			# If the person and network have the same score, update the distribution
			if s1 == s2:
				agree_chance[s2-1] += 1
				avg_chance[s2-1] += 1
			total_chance[s2-1] += 1
			avg_chance_total[s2-1] += 1

			# Check for novelty
			imnames = A.scores_dict[s]['image_pairs'][idx]
			im1 = imnames.split('_')[0]
			im2 = imnames.split('_')[1]

			spec1 = im1.split('/')[0]
			spec2 = im2.split('/')[0]

			if spec1 in in_imagenet and spec2 in in_imagenet:
				if s1 == s2:
					avg_chance_novelty[s2-1] += 1
				avg_chance_novelty_total[s2-1] += 1
			elif spec1 in notin_imagenet and spec2 in notin_imagenet:
				if s1 == s2:
					avg_chance_novelty[s2-1] += 1
				avg_chance_novelty_total[s2-1] += 1


		## Plot true vs chance agreement
		x = np.arange(1., 8)
		plt.figure()
		plt.bar(x, (agree0_b+agree0_i)/(total_b+total_i), width=0.2, label='True agreement')
		plt.bar(x+0.2, agree_chance/total_chance, width=0.2, label='Chance agreement')
		
		plt.title('Human-' + network_description + ' Agreement')
		plt.xlabel('Score (Least to Most Similar)')
		plt.ylabel('Frequency of Human Agreement')
		plt.legend()
		plt.tight_layout()
		plt.show()
		
		# Plot human-network agreement by prompt 
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

	# Plot average agreement by chance
	plt.figure()
	plt.bar(x, avg/avg_total, width=0.2, color='r', label='Agree')
	plt.bar(x+0.2, avg_chance/avg_chance_total, width=0.2, color='grey', label='Agree By Chance')

	plt.title('Average Human-Network Agreement')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Human-Network Agreement')
	plt.ylim(top=0.5)
	plt.legend()
	plt.tight_layout()
	plt.show()	

	# Plot average agreement by novelty
	print(avg_chance_novelty_total)
	plt.figure()
	plt.bar(x, avg_in/total_in, width=0.2, color='r', label='Both Known')
	plt.bar(x+0.2, avg_notin/total_notin, width=0.2, color='b', label='Both Novel')
	plt.bar(x+0.4, avg_chance_novelty/avg_chance_novelty_total, width=0.2, color='grey', label='Agree By Chance')
	plt.title('Average Human-Network Agreement By Novelty')
	plt.xlabel('Score (Least to Most Similar)')
	plt.ylabel('Human-Network Agreement')
	plt.ylim(top=0.5)

	plt.legend()
	plt.tight_layout()
	plt.show()
	
def human_network_pairs(A):
	###############################################################################
	## Scatter plot of CNN scores vs participant scores, stratified by prompt type
	###############################################################################
	# One plot per score type
	w=0.2
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
				xbird += list(np.array(x)-w)
				ybird += y
			else:
				ximage += list(np.array(x)+w)
				yimage += y
		
		# Add some jitter to the scores to make them easier to see
		xbird += np.random.uniform(low=-w, high=w, size=len(xbird))
		ximage += np.random.uniform(low=-w, high=w, size=len(ximage))

		plt.figure()
		plt.xticks(ticks=np.arange(1,8), labels=np.arange(1,8), fontsize=10)
		plt.scatter(xbird, ybird, label='Bird Prompt', c='b', marker='.', s=10)
		plt.scatter(ximage, yimage, label='Image Prompt', c='r', marker='.', s=10)
		plt.title('Network (' + scoretype +') vs Human Scores by Prompt', fontsize=12)
		plt.xlabel('Human Scores (Least to Most Similar)', fontsize=10)
		
		plt.yticks(fontsize=10)
		plt.ylabel('Normalized Network Scores (Least to Most Similar)', fontsize=10)
		plt.legend()
		plt.tight_layout()
		plt.show()

def vis_agreement_pairs(A):
	"""
	Saves image pairs where a given network and human participants gave scores of:
		human		network
		1			1
		7			7
	"""

	hscore = 7	
	for netlayer in A.scores_dict[hscore]['cnn_scores']:

		all_cscores = 1. - np.array(A.scores_dict[hscore]['cnn_scores'][netlayer])	# invert network scores

		all_impairs = np.array(A.scores_dict[hscore]['image_pairs'])
		cscore_bins = (np.floor(7*all_cscores) + 1.).astype(int)
		
		idx_match = np.argwhere(cscore_bins==1)
		
		# Open and save the two images together
		for pair in all_impairs[idx_match]:
			pair = pair[0]		# Each pair name is stored in its own list
			impath = os.getcwd() + '/images/Aves/'
			im1_path = impath + pair.split('_')[0] + '.jpg'
			im2_path = impath + pair.split('_')[1] + '.jpg'

			# Open images
			im1 = Image.open(im1_path)
			im2 = Image.open(im2_path)

			# Reshape if needed
			h = max(im1.height, im2.height)
			w1 = round(im1.width*h/im1.height)
			w2 = round(im2.width*h/im2.height)
			im1 = im1.resize((w1,h))
			im2 = im2.resize((w2,h))

			# Combine images
			newim = Image.new('RGB',(im1.width + im2.width, h))
			newim.paste(im1, (0,0))
			newim.paste(im2, (im1.width,0))

			# Save in directory
			savepath = os.getcwd() + '/impair_agreement/h7_c1_'+netlayer+'/'
			if not os.path.exists(savepath):
				os.mkdir(savepath)
			
			fname = pair.replace('/','_') + '.jpg'
			newim.save(savepath + fname)

def novelty(A, species_data):
	"""
	Some analysis taking into account whether the network was trained
	on the image class or not 
	"""
	#TODO: Incorporate the 'novelty' part of the image dict for all sections of this analysis
	imagenet = np.array(species_data['in imagenet'])
	cub = np.array(species_data['in cub'])
	species = np.array(species_data['name'])

	# Only look at species not in CUB training set
	# good_idx = np.argwhere(cub == 'No')
	# good_idx = np.arange(len(imagenet))
	
	in_imagenet = species[np.argwhere(imagenet == 'Yes')][:,0]
	notin_imagenet = species[np.argwhere(imagenet == 'No')][:,0]

	in_cub = species[np.argwhere(cub == 'Yes')][:,0]
	notin_cub = species[np.argwhere(cub == 'No')][:,0]

	##############################################################
	# Plots scores per layer, split by known/novel classes
	##############################################################

	print('Species In ImageNet (not in CUB):', len(in_imagenet), '\nNot in ImageNet:',len(notin_imagenet))
	
	## Plot network scores stratified by whether the image is in imagenet or not
	colors_in = ['darkred', 'darkgreen','midnightblue']
	colors_notin = ['lightsalmon', 'mediumaquamarine', 'cornflowerblue']
	
	for i,net in enumerate(['alexnet', 'vgg16', 'resnet']):
		# plt.figure()
		x_in = []
		x_notin = []
		y_in = []
		y_notin = []

		# Collect all scores for all image pairs
		for p in A.image_dict.keys():
			
			for layer in A.image_dict[p]['cnn_scores']:

				if net in layer:
					num = layer.split('_')[-1]		# network name is <net>_<block/conv>_<num>
					if num == 'output':
						num = 6
					if 'euc' == num:	#TODO: Fix this
						continue

					# Get network score
					s = 1 - A.image_dict[p]['cnn_scores'][layer]

					# Get species information
					im1 = p.split('_')[0]
					im2 = p.split('_')[1]

					spec1 = im1.split('/')[0]
					spec2 = im2.split('/')[0]

					# Check novelty
					# spec1_known  = (spec1 in in_imagenet and 'resnet' not in layer) \
					# 	or (spec1[i] in in_cub and 'resnet'  in layer)
					# spec2_known = (spec2 in in_imagenet and 'resnet' not in layer) \
					# 	or (spec2 in in_cub and 'resnet'  in layer)

					spec1_known = spec1 in in_imagenet
					spec2_known = spec2 in in_imagenet

					# print(spec1 in in_imagenet and spec2 in in_imagenet)
					if spec1_known and spec2_known:
						x_in.append(s)
						y_in.append(int(num))
					elif not spec1_known and not spec2_known:
						x_notin.append(s)	
						y_notin.append(int(num))
		
		# Both images are in ImageNet
		y_in += np.random.uniform(low=-.2, high=0.2, size=len(y_in))
		plt.scatter(x_in, y_in, c=colors_in[i], s=1, label='Both In ImageNet')

		# Both images are not in ImageNet
		y_notin += np.random.uniform(low=-.2, high=0.2, size=len(y_notin))
		plt.scatter(x_notin, y_notin+0.4, c=colors_notin[i], s=1, label='Neither in ImageNet')

		netname = net.split('_')[0].capitalize()
		plt.title('Score Distribution for ' + netname, fontsize=14)
		plt.xlabel('Score (Least to Most Similar)', fontsize=12)
		plt.ylabel('Network Layer', fontsize=12)
		plt.yticks(ticks=np.arange(6)+1, labels=['1','2','3','4','5','output'], fontsize=10)
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.show()
	
	
	#####################################################################################
	# Plot human-network score pairs based on whether or not the species is in imagenet #
	#####################################################################################
	for layer in A.cnn_layers:

		# Agregate data for all participants
		# x axis = human scores
		# y axis = network scores
		y_in = [[],[],[],[],[],[],[]]
		y_onein = [[],[],[],[],[],[],[]]
		y_notin = [[],[],[],[],[],[],[]]

		# Split participant data and cnn data by prompt
		for part in A.partic_dict.keys():

			imgs = [i.split('_') for i in A.partic_dict[part]['image_pairs']]

			spec1 = np.array([i[0].split('/')[0] for i in imgs])
			spec2 = np.array([i[1].split('/')[0] for i in imgs])

			hscores = list(A.partic_dict[part]['hscores'])                    # Human Scores
			cscores = list(1 - A.partic_dict[part]['cnn_scores'][layer])      # Network Scores (inverted)

			for i in range(len(spec1)):

				# spec1_known  = (spec1[i] in in_imagenet and 'resnet' not in layer) \
				# 		or (spec1[i] in in_cub and 'resnet'  in layer)
				# spec2_known = (spec2[i] in in_imagenet and 'resnet' not in layer) \
				# 		or (spec2[i] in in_cub and 'resnet'  in layer)

				spec1_known = spec1[i] in in_imagenet
				spec2_known = spec2[i] in in_imagenet

				# If both species in imagenet, add to list
				if spec1_known and spec2_known:
					x = hscores[i]-1
					y_in[x].append(cscores[i])

				# If exactly one species in imagenet, add to list
				elif ((spec1_known and not spec2_known) or
					 (not spec1_known and spec2_known)):
					x = hscores[i]-1
					y_onein[x].append(cscores[i])

				# If neither species in imagenet, add to list
				elif not spec1_known and not spec2_known:
					x = hscores[i]-1
					y_notin[x].append(cscores[i])

		# Plot score pairs for known, novel, and one known/one novel cases
		# plt.figure()
		plt.xticks(ticks=np.arange(1,8), labels=np.arange(1,8), fontsize=10)
		w=0.1
		for x in range(7):	# Plot by human score
			# Add jitter so points are more visible
			x_in = np.random.uniform(low=x+1-w, high=x+1+w, size=len(y_in[x]))		
			x_onein = np.random.uniform(low=x+1-w, high=x+1+w, size=len(y_onein[x]))	
			x_notin = np.random.uniform(low=x+1-w, high=x+1+w, size=len(y_notin[x]))	

			Plot score pairs based on network familiarity
			plt.scatter(x_in+2*w, y_in[x],  c='b', marker='.', s=8)
			plt.scatter(x_onein, y_onein[x],  c='g', marker='.',s=8)
			plt.scatter(x_notin-2*w, y_notin[x], c='r', marker='.', s=8)


		plt.title('Network (' + layer +') vs Human Scores - Novelty', fontsize=12)
		plt.xlabel('Human Scores (Least to Most Similar)', fontsize=10)
		plt.yticks(fontsize=10)
		plt.ylabel('Normalized Network Scores (Least to Most Similar)', fontsize=10)
		plt.legend(labels=['Both Known', 'One Known', 'Both Novel'],loc='lower right')
		plt.tight_layout()
		plt.show()

		# Compute correlation based on network familiarity
		x = []
		y = []
		for i in range(7):
			y += y_in[i]
			x += list(i*np.ones(len(y_in[i])))
		print('Pearson correlation In ImageNet:', layer, pearsonr(x, y))
		print('Spearman Rank correlation In ImageNet:', layer, spearmanr(x, y))
		
		x = []
		y = []
		for i in range(7):
			y += y_onein[i]
			x += list(i*np.ones(len(y_onein[i])))
		print('Pearson correlation One In ImageNet:', layer, pearsonr(x, y))
		print('Spearman Rank correlation One In ImageNet:', layer, spearmanr(x, y))
		
		x = []
		y = []
		for i in range(7):
			y += y_notin[i]
			x += list(i*np.ones(len(y_notin[i])))
		print('Pearson correlation Not In ImageNet:', layer, pearsonr(x, y))
		print('Spearman Rank correlation Not In ImageNet:', layer,spearmanr(x, y))

		# Plot only the average score
		y_in_avg = np.zeros(7)
		y_onein_avg = np.zeros(7)
		y_notin_avg = np.zeros(7)
		plt.figure()
		for x in range(7):	# Plot by human score
			y_in_avg[x] = np.mean(y_in[x])
			y_onein_avg[x] = np.mean(y_onein[x])
			y_notin_avg[x] = np.mean(y_notin[x])
	
		plt.plot(np.arange(7), y_in_avg, c='b', label='Both Known')
		plt.plot(np.arange(7), y_onein_avg, c='g', label='One Known')
		plt.plot(np.arange(7), y_notin_avg, c='r', label='Both Novel')
		plt.xticks(np.arange(7), np.arange(1,8))

		plt.title('Network (' + layer +') vs Human Scores - Novelty', fontsize=12)
		plt.xlabel('Human Scores (Least to Most Similar)', fontsize=10)
		plt.ylim(0,1)
		
		plt.yticks(fontsize=10)
		plt.ylabel('Normalized Network Scores (Least to Most Similar)', fontsize=10)
		plt.legend(loc='lower right')
		plt.tight_layout()
		plt.show()

def species_analysis(A, species_data):
	"""
	Extracts statistics about species similarity in human/network choices
	"""

	#####################################################################################
	# How often the species is the same for all peoples' scores
	#####################################################################################
	same_spec_people = np.zeros(7)
	same_spec_people_image = np.zeros(7)
	same_spec_people_bird = np.zeros(7)
	total = np.zeros(7)
	total_image = np.zeros(7)
	total_bird = np.zeros(7)
	for p in A.image_dict.keys():

		# Get species information
		im1 = p.split('_')[0]
		im2 = p.split('_')[1]

		spec1 = im1.split('/')[0]
		spec2 = im2.split('/')[0]

		for i, hscore in enumerate(A.image_dict[p]['scores']):
			if spec1 == spec2:
				same_spec_people[hscore-1] += 1

			if A.image_dict[p]['prompts'][i] == 'birds':
				# print('bird')
				if spec1 == spec2:
					same_spec_people_bird[hscore-1] += 1
				total_bird[hscore-1] += 1
			elif A.image_dict[p]['prompts'][i] == 'images':
				if spec1 == spec2:
					same_spec_people_image[hscore-1] += 1
				total_image[hscore-1] += 1


			total[hscore-1] += 1

	print(same_spec_people)
	print('Same Species for Humans:', same_spec_people/total)
	print('Same Species Human Bird', same_spec_people_bird/total_bird)
	print('Same Species Human Images', same_spec_people_image/total_image)
	plt.figure()
	# plt.plot(np.arange(1,8), same_spec_people/total, marker='o', color='r')
	plt.plot(np.arange(1,8), same_spec_people_bird/total_bird, marker='o', color='r', label='Bird Prompt')
	plt.plot(np.arange(1,8), same_spec_people_image/total_image, marker='o', color='b', label='Image Prompt')
	plt.legend()
	plt.title('Same-Species Frequency: Humans')
	plt.xlabel('Human Scores')
	plt.ylabel('Frequency of Same Species')
	plt.xticks(np.arange(1,8), np.arange(1,8))
	plt.hlines(np.arange(0,1,0.1), 1, 7, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.tight_layout()
	plt.show()

	#####################################################################################
	# How often the species is the same for all networks' scores
	#####################################################################################
	same_spec_networks = np.zeros((len(networks), 7))
	total = np.zeros((len(networks),7))
	for p in A.image_dict.keys():

		# Get species information
		im1 = p.split('_')[0]
		im2 = p.split('_')[1]

		spec1 = im1.split('/')[0]
		spec2 = im2.split('/')[0]

		for i, j in enumerate(range(len(networks))):
			n = networks[j]
			cnn_score = 1. - A.image_dict[p]['cnn_scores'][n]
			cnn_bin = int(np.floor(7*cnn_score) + 1)
			if cnn_bin == 8:
				cnn_bin = 7
			
			if spec1 == spec2:
				same_spec_networks[i, cnn_bin-1] += 1

			total[i,cnn_bin-1] += 1

	# Convert to plottable format
	# nets, scores = same_spec_networks.shape
	# xs = []
	# ys = []
	# size = []
	# colors = []
	# labels = []
	# cm = plt.cm.get_cmap('RdYlBu')
	# for y, j in enumerate(np.arange(18)):
	# 	for x in range(scores):
			
	# 		xs.append(x)
	# 		ys.append(y)
	# 		size.append(1500*same_spec_networks[y,x]/total[y,x])
	# 		frac = float(same_spec_networks[y,x]/total[y,x])
	# 		colors.append(frac)

	# 	labels.append(networks[j])
			
	# Scatter Plot
	# plt.figure()
	# fig = plt.gcf()
	# ax = fig.gca()

	# sc = ax.scatter(xs,ys,s=size, c=colors)
	# sc.set_clim([0, 1])
	# cb = plt.colorbar(sc)
	# plt.title('ResNet-18: Fraction of Matching Species')
	# plt.xlabel('Network Bin Scores')
	# plt.ylabel('Network Layers')
	# plt.xticks(np.arange(7), np.arange(1,8))
	# plt.yticks(np.arange(len(labels)), labels = labels)
	# plt.tight_layout()
	# plt.show()

	# Line plot
	# AlexNet
	print(networks)
	same_spec_alex = np.sum(same_spec_networks[:6], axis=0)
	total_alex = np.sum(total[:6], axis=0)
	# VGG
	same_spec_vgg = np.sum(same_spec_networks[6:12], axis=0)
	total_vgg = np.sum(total[6:12], axis=0)
	# ResNet
	same_spec_res = np.sum(same_spec_networks[12:], axis=0)
	total_resnet = np.sum(total[12:], axis=0)
	# Average
	same_spec_avglayer = np.sum(same_spec_networks, axis=0)
	total_avglayer = np.sum(total, axis=0)

	plt.figure()
	plt.plot(np.arange(1,8), same_spec_alex/total_alex, marker='o', color='r', label='AlexNet')
	plt.plot(np.arange(1,8), same_spec_vgg/total_vgg, marker='o', color='purple', label='VGG16')
	plt.plot(np.arange(1,8), same_spec_res/total_resnet, marker='o', color='teal', label='ResNet-18')
	plt.plot(np.arange(1,8), same_spec_avglayer/total_avglayer, marker='o', color='grey', label='Average')
	plt.title('Same-Species Frequency: Networks')
	plt.xlabel('Binned Network Scores')
	plt.ylabel('Frequency of Same Species')
	plt.legend()
	plt.xticks(np.arange(1,8), np.arange(1,8))
	plt.hlines(np.arange(0,1,0.1), 1, 7, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.tight_layout()
	plt.show()

	#####################################################################################
	# When people and networks give the same score, how often are the species the same
	#####################################################################################
	same_score = np.zeros(7)
	total = np.zeros(7)
	for p in A.image_dict.keys():

		# Get species information
		im1 = p.split('_')[0]
		im2 = p.split('_')[1]

		spec1 = im1.split('/')[0]
		spec2 = im2.split('/')[0]

		for i, j in enumerate(np.arange(18)):
			n = networks[j]
			cnn_score = 1. - A.image_dict[p]['cnn_scores'][n]
			cnn_bin = int(np.floor(7*cnn_score) + 1)
			if cnn_bin == 8:
				cnn_bin = 7
			
			# Loop through human scores
			for hscore in A.image_dict[p]['scores']:
				if hscore == cnn_bin:		# If people and networks agree
					if spec1 == spec2:		# If species are the same
						same_score[cnn_bin-1] += 1

					total[cnn_bin-1] += 1

	# print(same_score/total)

	plt.figure()
	plt.plot(np.arange(1,8), same_score/total, marker='o', color='r')
	plt.title('Same-Species Frequency: Human-Network Agreement Average')
	plt.xlabel('Scores')
	plt.ylabel('Frequency of Same Species')
	plt.xticks(np.arange(1,8), np.arange(1,8))
	plt.hlines(np.arange(0,1,0.1), 1, 7, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.tight_layout()
	plt.show()


	###################################################################
	# Split by network layer averages
	###################################################################

	colors = ['lightcoral','orangered', 'gold','lightgreen','turquoise', 'mediumpurple']
	layers = ['1','2','3','4','5','output']
	# plt.figure()
	for i, l in enumerate(layers):
		same_score = np.zeros(7)
		total = np.zeros(7)
		c = colors[i]
		for p in A.image_dict.keys():

			# Get species information
			im1 = p.split('_')[0]
			im2 = p.split('_')[1]

			spec1 = im1.split('/')[0]
			spec2 = im2.split('/')[0]

			for i, j in enumerate(np.arange(18)):
				n = networks[j]
				if l not in n:
					continue
				# if 'resnet' not in n:
				# 	continue

				cnn_score = 1. - A.image_dict[p]['cnn_scores'][n]
				cnn_bin = int(np.floor(7*cnn_score) + 1)
				if cnn_bin == 8:
					cnn_bin = 7
				
				# Loop through human scores
				for hscore in A.image_dict[p]['scores']:
					if hscore == cnn_bin:		# If people and networks agree
						if spec1 == spec2:		# If species are the same
							same_score[cnn_bin-1] += 1

						total[cnn_bin-1] += 1

		# plt.plot(np.arange(1,8), same_score/total, color=c, marker='o', label=l)

	# plt.title('Same-Species Frequency: Human-Network Agreement by Layer')
	# plt.xlabel('Scores')
	# plt.ylabel('Frequency of Same Species')
	# plt.xticks(np.arange(1,8), np.arange(1,8))
	# plt.hlines(np.arange(0,1,0.1), 1, 7, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	# plt.legend()
	# plt.tight_layout()
	# plt.show()

	#####################################################################################
	# How often people and networks give the same score when the species are the same 
	# and the network has never seen it before
	#####################################################################################

	same_score_known = np.zeros(7)
	same_score_novel = np.zeros(7)
	total_known = np.zeros(7)
	total_novel = np.zeros(7)

	for p in A.image_dict.keys():
		# Get species information
		im1 = p.split('_')[0]
		im2 = p.split('_')[1]

		spec1 = im1.split('/')[0]
		spec2 = im2.split('/')[0]
		# Counting each individual's score
		
		for i, j in enumerate(range(len(networks))):
			n = networks[j]
			# if 'vgg' not in n:
			# 	continue

			
			cnn_score = 1. - A.image_dict[p]['cnn_scores'][n]
			cnn_bin = int(np.floor(7*cnn_score) + 1)
			if cnn_bin == 8:
				cnn_bin = 7
			
			# Loop through human scores
			for hscore in A.image_dict[p]['scores']:
				if hscore == cnn_bin:		# If people and networks agree
				
					# Check novelty respective to the network 
					# Note ResNet is trained on CUB, the others are trained on ImageNet
					im1known = ('imagenet' in A.image_dict[p]['novelty']['im1'] and 'resnet' not in n) \
						or ('cub' in A.image_dict[p]['novelty']['im1'] == 'cub' and 'resnet'  in n)
					im2known = ('imagenet' in A.image_dict[p]['novelty']['im2'] and 'resnet' not in n) \
						or ( 'cub' in A.image_dict[p]['novelty']['im2'] == 'cub' and 'resnet'  in n)

					if im1known and im2known:
						if spec1 == spec2:		# If species are the same
							same_score_known[cnn_bin-1] += 1
					total_known[cnn_bin-1] +=1
					if not im1known and not im2known:
						if spec1 == spec2:		# If species are the same
							same_score_novel[cnn_bin-1] += 1
					total_novel[cnn_bin-1] += 1
		"""

		# Counting if at least one person/network agreed

		# Get list of all network scores
		cnn_scores = set()
		for n in networks:			
			cnn_score = 1. - A.image_dict[p]['cnn_scores'][n]
			cnn_bin = int(np.floor(7*cnn_score) + 1)
			if cnn_bin == 8:
				cnn_bin = 7
			cnn_scores.add(cnn_bin)
		human_scores = set(A.image_dict[p]['scores'])
		agree_scores = cnn_scores.intersection(human_scores)
		print(agree_scores)

		# Check novelty respective to the network 
		# Note ResNet is trained on CUB, the others are trained on ImageNet
		im1known = ('imagenet' in A.image_dict[p]['novelty']['im1'] and 'resnet' not in n) \
			or ('cub' in A.image_dict[p]['novelty']['im1'] == 'cub' and 'resnet'  in n)
		im2known = ('imagenet' in A.image_dict[p]['novelty']['im2'] and 'resnet' not in n) \
			or ( 'cub' in A.image_dict[p]['novelty']['im2'] == 'cub' and 'resnet'  in n)

		for s in agree_scores:
			if im1known and im2known:
				if spec1 == spec2:
					same_score_known[cnn_bin-1] += 1
				total_known[cnn_bin-1] +=1
			if not im1known and not im2known:
				if spec1 == spec2:		# If species are the same
					same_score_novel[cnn_bin-1] += 1
				total_novel[cnn_bin-1] += 1
		"""
			


	# plt.figure()
	print('known',same_score_known/total_known)
	print('novel',same_score_novel/total_novel)


	plt.plot(np.arange(1,8), same_score_known/total_known, marker='o', color='r', label='Both Known')
	plt.plot(np.arange(1,8), same_score_novel/total_novel, marker='o', color='b', label='Both Novel')
	plt.legend()
	plt.title('Same-Species Frequency: Human-Network Agreement by Novelty')
	plt.xlabel('Scores')
	plt.ylabel('Frequency of Same Species')
	plt.xticks(np.arange(1,8), np.arange(1,8))
	plt.hlines(np.arange(0,1,0.1), 1, 7, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	participant_data_files = os.getcwd() + '/data/'
	feature_path = os.getcwd() + '/features/'
	network_labels_path = 'C:/Users/noam_/Documents/Cornell/CS7999/novelty-detection/'

	# List of all network 
	networks = ['alexnet_conv_1',
				'alexnet_conv_2',
				'alexnet_conv_3',
				'alexnet_conv_4',
				'alexnet_conv_5',
				'alexnet_output',
				'vgg16_block_1',
				'vgg16_block_2',
				'vgg16_block_3',
				'vgg16_block_4',
				'vgg16_block_5',
				'vgg16_output',
				'resnet18_block_1',
				'resnet18_block_2',
				'resnet18_block_3',
				'resnet18_block_4',
				'resnet18_block_5',
				'resnet_output'
				]         

	A = Analysis(feature_path, participant_data_files, network_labels_path)
	# A.calc_cnn_scores(networks)         # Only needs to be done once  -
										# this makes a copy of your data files with the normalized network distance scores
	
	A.make_partic_dict()
	A.make_scores_dict()
	A.make_image_dict()

	df = pd.read_csv(os.getcwd() + '/test_species.csv', encoding="ISO-8859-1")

	###############################################################################
	## All plots
	############################################################################### 
	
	# Score distributions
	hscore_distro = human_score_distro(A, plot=False)
	cnnscore_distro = network_score_distro(A, plot=False)		# The result is the cnn score distro in bins 1-7

	# TODO: Clean up and split up some of these functions to make them clearer

	# Human-human agreement and trends
	# human_human_pairs(A)
	# human_human_agreement(A, hscore_distro)
	# response_time(A)

	# Human-network agreement and trends
	# human_network_agreement_combined(A, hscore_distro)
	# human_network_agreement_separate(A, hscore_distro, df)
	# human_network_pairs(A)
	# vis_agreement_pairs(A)

	# Analysis based on network correctness
	novelty(A, df)
	# species_analysis(A, df)
