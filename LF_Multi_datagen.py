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

    def __call__(self, train_data_0, train_data_90, train_data_45, train_data_135, gt_image):

        h, w = train_data_0.shape[1],train_data_0.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        train_data_0_tmp = train_data_0[:, top: top + new_h, left: left + new_w]
        train_data_90_tmp = train_data_90[:, top: top + new_h, left: left + new_w]
        train_data_45_tmp = train_data_45[:, top: top + new_h, left: left + new_w]
        train_data_135_tmp = train_data_135[:, top: top + new_h, left: left + new_w]

        gtdata_tmp = gt_image[:, top: top + new_h, left: left + new_w]

        return train_data_0_tmp, train_data_90_tmp, train_data_45_tmp, train_data_135_tmp, gtdata_tmp


class LF_Multi_Dataset(data.Dataset):
    """Light Field dataset."""
    def __init__(self, train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp, batch_size=32, view_n=9,
                 crop_size=33, if_flip=True, if_rotation=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_data_0 = train_data_0
        self.train_data_90 = train_data_90
        self.train_data_45 = train_data_45
        self.train_data_135 = train_data_135
        self.gt_data_tmp = gt_data_tmp

        self.crop_size = crop_size
        self.transform = transform
        self.batch_size = batch_size
        self.view_n = view_n
        self.RandomCrop = RandomCrop(crop_size)
        self.if_flip = if_flip
        self.if_rotation = if_rotation

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):

        # for test
        if self.crop_size == 1:
            train_data_0 = self.train_data_0
            train_data_90 = self.train_data_90
            train_data_45 = self.train_data_45
            train_data_135 = self.train_data_135
            gt_data = self.gt_data_tmp
        # for train
        else:
            train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
                self.RandomCrop(self.train_data_0, self.train_data_90, self.train_data_45, self.train_data_135,
                                self.gt_data_tmp)

        # flip
        if self.if_flip:
            random_tmp = np.random.random()
            if random_tmp >= (2.0 / 3):
                train_data_0 = np.flip(train_data_0, 1)
                train_data_90 = np.flip(train_data_90, 1)
                train_data_45 = np.flip(train_data_45, 1)
                train_data_135 = np.flip(train_data_135, 1)
                gt_data = np.flip(gt_data, 1)
            elif random_tmp <= (1.0 / 3):
                train_data_0 = np.flip(train_data_0, 2)
                train_data_90 = np.flip(train_data_90, 2)
                train_data_45 = np.flip(train_data_45, 2)
                train_data_135 = np.flip(train_data_135, 2)
                gt_data = np.flip(gt_data, 2)

        if self.if_rotation:
            random_tmp = np.random.random()
            if random_tmp <= (1.0 / 4):
                train_data_0 = np.rot90(train_data_0, 1, (1, 2))
                train_data_90 = np.rot90(train_data_90, 1, (1, 2))
                train_data_45 = np.rot90(train_data_45, 1, (1, 2))
                train_data_135 = np.rot90(train_data_135, 1, (1, 2))
                gt_data = np.rot90(gt_data, 1, (1, 2))

                train_tmp = train_data_0
                train_data_0 = train_data_90
                train_data_90 = train_tmp

                train_tmp = train_data_45
                train_data_45 = train_data_135
                train_data_135 = train_tmp

            elif random_tmp >= (3.0 / 4):
                train_data_0 = np.rot90(train_data_0, 2, (1, 2))
                train_data_90 = np.rot90(train_data_90, 2, (1, 2))
                train_data_45 = np.rot90(train_data_45, 2, (1, 2))
                train_data_135 = np.rot90(train_data_135, 2, (1, 2))
                gt_data = np.rot90(gt_data, 2, (1, 2))

            elif (random_tmp >= (1.0 / 4)) & (random_tmp <= (2.0 / 4)):
                train_data_0 = np.rot90(train_data_0, 3, (1, 2))
                train_data_90 = np.rot90(train_data_90, 3, (1, 2))
                train_data_45 = np.rot90(train_data_45, 3, (1, 2))
                train_data_135 = np.rot90(train_data_135, 3, (1, 2))
                gt_data = np.rot90(gt_data, 3, (1, 2))

                train_tmp = train_data_0
                train_data_0 = train_data_90
                train_data_90 = train_tmp

                train_tmp = train_data_45
                train_data_45 = train_data_135
                train_data_135 = train_tmp

        # if self.transform:

        return torch.from_numpy(train_data_0.copy()), torch.from_numpy(train_data_90.copy()), torch.from_numpy(train_data_45.copy()), \
               torch.from_numpy(train_data_135.copy()), torch.from_numpy(gt_data.copy())
