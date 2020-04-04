# Test layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, datasets
from torchsummary import summary
import torch.utils.data as data
from tqdm import tqdm
import os
from transfer_model import Transfer

BATCH_SIZE = 1
ROOT = "D:/noam_/Cornell/CS7999/iNaturalist/"
TEST_DATA_PATH = os.path.join(ROOT, "experiment classes/Aves/")
SAVE_DATA_PATH = os.path.join(ROOT, "features/resnet_block1/")
TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Author: Andrew Jong
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

if __name__ == '__main__':
    """
    # Print AlexNet network summary
    print('ALEXNET')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alex = models.alexnet(pretrained=True).to(device)
    summary(alex,(3,244,244))
    print(nn.Sequential(*list(alex.children())))

    # Print VGG16 network summary
    print('VGG16')
    vgg = models.vgg16(pretrained=True).to(device)
    summary(vgg,(3,244,244))
    print(nn.Sequential(*list(vgg.children())))
    """

    # Print VGG16 network summary
    print('ResNet18')
    CUB_path = 'model_train_sgd-lr_1e-01-gamma_1e-01-epoch_100-stage_2-decay_1e-03-bs_64-gpu_2.pth'
    # model_state_dict = torch.load(CUB_path, map_location='cuda:0')
    resnet = Transfer(num_channel=32, num_class=100, resnet=True)
    model_state = torch.load(CUB_path, map_location='cuda:0')
    resnet.load_state_dict(model_state)

    # summary(model,(3,244,244))
    print(nn.Sequential(*list(resnet.children())))
    
    # Extract the partial model up to the point of interest
    model_ft = models.vgg16(pretrained=True)
    partial_model = nn.Sequential(*list(model_ft.children())[0][:3])
    print('Partial model:\n',partial_model)
    
    # Set up model to not update parameters
    for param in partial_model.parameters():
        param.requires_grad = False

    test_data = ImageFolderWithPaths(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

    for inputs, labels, paths in tqdm(test_data_loader):
        # Parse file name
        path_info = paths[0].strip().split(TEST_DATA_PATH)[1].split('\\')
        foldername = path_info[0] + '/'
        filename = path_info[1].split('.jpg')[0] + '.pt'

        # Create sub-directory for saving results
        if not os.path.exists(SAVE_DATA_PATH + foldername):
            os.mkdir(SAVE_DATA_PATH + foldername)

        # Extract features
        output = partial_model(inputs)

        # Save results
        torch.save(output, SAVE_DATA_PATH + foldername + filename)

        # Example of loading results:
        # reading = torch.load(SAVE_DATA_PATH + foldername + filename)
    