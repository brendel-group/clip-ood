import numpy as np
import os
from tqdm import tqdm
import pickle
import open_clip
import torchvision.datasets as datasets
import torch

def encode_image(model, image):
    ''' encode an image using CLIP
    '''

    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding /= embedding.norm(dim=-1, keepdim=True)

    return embedding


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path
