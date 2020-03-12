# bird-similarity-experiment
Psychology experiment code for a study on how people rate similarity between bird images

## Image Similarity from CNN
The data in the stratified_img_pairs contains the similarity scores between pairs of bird images from approximately 60 different bird species. The similarity scores were calculated as the Euclidean distance between the last hidden layer representation of these images in an Alexnet. Alexnet is pre-trained on the ImageNet dataset.

These scores are sorted, normalized, and split into 7 bins, as reflected in the 7 data files. Each file has a different range of scores in sorted order, described in the file name. Example: subset_0_scores_0.0-0.14.csv has scores from 0.0 to 0.14. Note that lower scores in the CNN correspond to a higher similarity, while higher scores for a human correspond to a higher similarity.

## Images
The images in this experiment are all taken from approximately 60 species of birds in the iNaturalist image dataset.

## Experiment
This experiment uses PsychoPy as a framework. Participants are asked to rate either the image similarities or the bird similarities on a sliding scale from 1 to 7. The images are sorted into score bins (described above), and for any given trial (bird pair) one of those bins is randomly selected. Then, a bird pair is randomly selected from that file. This way, the participants see images with uniformly distributed scores as rated by the CNN.
