import torch
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data

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
        gtdata_tmp = gt_image[:,top+17: top + new_h-17, left+17: left + new_w-17]

        return traindata_tmp, gtdata_tmp



class LFDataset(data.Dataset):
    """Light Field dataset."""
    def __init__(self, image_path='', batch_size=32, view_n=9, crop_size=33, image_h=512, image_w=512, If_flip=True, If_rotation=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.crop_size = crop_size
        self.image_path = image_path
        self.transform = transform
        self.batch_size = batch_size
        self.view_n = view_n
        self.image_h = image_h
        self.image_w = image_w
        self.RandomCrop = RandomCrop(crop_size)
        self.If_flip = If_flip
        self.If_rotation = If_rotation


    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):

        RGB = [0.299, 0.587, 0.114]

        traindata_tmp = np.zeros((self.view_n, self.image_h, self.image_w), dtype=np.float32)
        gtdata_tmp = np.zeros((1, self.image_h, self.image_w), dtype=np.float32)

        tmp = np.float32(imageio.imread(self.image_path + '/input_Cam040.png'))
        gtdata_tmp[0, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255

        i = 0
        for seq in range(36, 45, 1):

            img = Image.open(self.image_path + '/input_Cam0%.2d.png' % seq)
            new_img = img.resize((int(self.image_h / 2), int(self.image_w / 2)), Image.BICUBIC)
            img = new_img.resize((self.image_h, self.image_w), Image.BICUBIC)
            tmp = np.asarray(img)
            traindata_tmp[i, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255
            i += 1

        # flip
        if self.If_flip:
            if np.random.random() <= 0.5:
                random_tmp = np.random.random()
                if random_tmp <= (1.0 / 3):
                    traindata_tmp = np.flip(traindata_tmp, 1)
                    gtdata_tmp = np.flip(gtdata_tmp, 1)
                else:
                    traindata_tmp = np.flip(traindata_tmp, 2)
                    gtdata_tmp = np.flip(gtdata_tmp, 2)

        # rotation
        if self.If_rotation:
            if np.random.random() <= 0.5:
                random_tmp = np.random.random()
                if random_tmp <= 0.5:
                    traindata_tmp = np.rot90(traindata_tmp, 1, (1,2))
                    gtdata_tmp = np.rot90(gtdata_tmp, 1, (1,2))
                else:
                    traindata_tmp = np.rot90(traindata_tmp, 2, (1, 2))
                    gtdata_tmp = np.rot90(gtdata_tmp, 2, (1, 2))

        if self.transform:
            traindata_tmp, gtdata_tmp = self.transform(traindata_tmp, gtdata_tmp)

        # for test
        if self.crop_size==1:
            traindata = traindata_tmp
            gtdata = gtdata_tmp[:,17:495,17:495]
        # for train
        else:
            traindata = np.zeros((self.view_n, self.crop_size, self.crop_size), dtype=np.float32)
            gtdata = np.zeros((1, self.crop_size-34, self.crop_size-34), dtype=np.float32)
            traindata[:, :, :], gtdata[:, :, :] = self.RandomCrop(traindata_tmp, gtdata_tmp)

        return torch.from_numpy(traindata),torch.from_numpy(gtdata)