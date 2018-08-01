import torch
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
import time

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):

        self.output_size = (output_size, output_size)

    def __call__(self, image, gt_image):

        h, w = image.shape[1],image.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        traindata_tmp = image[:,top: top + new_h, left: left + new_w]
        gtdata_tmp = gt_image[:,top: top + new_h, left: left + new_w]

        return traindata_tmp, gtdata_tmp



class LFDataset(data.Dataset):
    """Light Field dataset."""
    def __init__(self, traindata_tmp, gtdata_tmp, batch_size=32, view_n=9, crop_size=33,  If_flip=True, If_rotation=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.traindata_tmp = traindata_tmp
        self.gtdata_tmp = gtdata_tmp
        self.crop_size = crop_size
        self.transform = transform
        self.batch_size = batch_size
        self.view_n = view_n
        self.RandomCrop = RandomCrop(crop_size)
        self.If_flip = If_flip
        self.If_rotation = If_rotation


    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):

        # flip
        if self.If_flip:
            if np.random.random() <= 0.5:
                random_tmp = np.random.random()
                if random_tmp <= (1.0 / 3):
                    self.traindata_tmp = np.flip(self.traindata_tmp, 1)
                    self.gtdata_tmp = np.flip(self.gtdata_tmp, 1)
                else:
                    self.traindata_tmp = np.flip(self.traindata_tmp, 2)
                    self.gtdata_tmp = np.flip(self.gtdata_tmp, 2)

        # rotation
        if self.If_rotation:
            if np.random.random() <= 0.5:
                random_tmp = np.random.random()
                if random_tmp <= 0.5:
                    self.traindata_tmp = np.rot90(self.traindata_tmp, 1, (1,2))
                    self.gtdata_tmp = np.rot90(self.gtdata_tmp, 1, (1,2))
                else:
                    self.traindata_tmp = np.rot90(self.traindata_tmp, 2, (1, 2))
                    self.gtdata_tmp = np.rot90(self.gtdata_tmp, 2, (1, 2))

        if self.transform:
            self.traindata_tmp, self.gtdata_tmp = self.transform(self.traindata_tmp, self.gtdata_tmp)

        # for test
        if self.crop_size==1:
            traindata = self.traindata_tmp
            gtdata = self.gtdata_tmp
        # for train
        else:
            traindata = np.zeros((self.view_n, self.crop_size, self.crop_size), dtype=np.float32)
            gtdata = np.zeros((1, self.crop_size, self.crop_size), dtype=np.float32)
            traindata[:, :, :], gtdata[:, :, :] = self.RandomCrop(self.traindata_tmp, self.gtdata_tmp)

        return torch.from_numpy(traindata.copy()), torch.from_numpy(gtdata.copy())
