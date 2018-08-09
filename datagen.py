import torch
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

        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        traindata_tmp = image[:, top: top + new_h, left: left + new_w]
        gtdata_tmp = gt_image[:, top*2: top*2 + new_h*2, left*2: left*2 + new_w*2]

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

        # for test
        if self.crop_size == 1:
            train_data = self.traindata_tmp
            gt_data = self.gtdata_tmp
        # for train
        else:
            train_data = np.zeros((self.view_n, self.crop_size, self.crop_size), dtype=np.float32)
            gt_data = np.zeros((1, self.crop_size*2, self.crop_size*2), dtype=np.float32)
            train_data[:, :, :], gt_data[:, :, :] = self.RandomCrop(self.traindata_tmp, self.gtdata_tmp)

        # flip
        if self.If_flip:
            random_tmp = np.random.random()
            if random_tmp >= (2.0 / 3):
                train_data = np.flip(train_data, 1)
                gt_data = np.flip(gt_data, 1)
            elif random_tmp <= (1.0 / 3):
                train_data = np.flip(train_data, 2)
                gt_data = np.flip(gt_data, 2)

        # rotation
        if self.If_rotation:
            random_tmp = np.random.random()
            if random_tmp <= (1.0 / 4):
                train_data = np.rot90(train_data, 1, (1,2))
                gt_data = np.rot90(gt_data, 1, (1,2))
            elif random_tmp >= (3.0 / 4):
                train_data = np.rot90(train_data, 2, (1, 2))
                gt_data = np.rot90(gt_data, 2, (1, 2))
            elif (random_tmp >= (1.0 / 4)) & (random_tmp <= (2.0 / 4)):
                train_data = np.rot90(train_data, 3, (1, 2))
                gt_data = np.rot90(gt_data, 3, (1, 2))

        if self.transform:
            train_data, gt_data = self.transform(train_data, gt_data)

        return torch.from_numpy(train_data.copy()), torch.from_numpy(gt_data.copy())
