import imageio
import numpy as np
from PIL import Image
"""
separate images input for hci new dataset
'/input_Cam040.png'

"""


def hci_new_input(image_path, view_n):

    RGB = [0.299, 0.587, 0.114]

    tmp = np.float32(imageio.imread(image_path + '/input_Cam040.png'))
    image_h = len(tmp[0])
    image_w = len(tmp[1])

    train_data_tmp = np.zeros((view_n, image_h, image_w), dtype=np.float32)
    gt_data_tmp = np.zeros((1, image_h, image_w), dtype=np.float32)

    gt_data_tmp[0, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255

    i = 0
    for seq in range(36, 45, 1):
        img = Image.open(image_path + '/input_Cam0%.2d.png' % seq)
        new_img = img.resize((int(image_h / 2), int(image_w / 2)), Image.BICUBIC)
        img = new_img.resize((image_h, image_w), Image.BICUBIC)
        # save the training image
        # if seq == 40:
        #     img.convert('L').save('%.2d_old.png' % epoch)
        tmp = np.asarray(img)
        train_data_tmp[i, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255
        i += 1

    return train_data_tmp, gt_data_tmp