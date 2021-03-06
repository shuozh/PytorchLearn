import imageio
import numpy as np
from PIL import Image
import os
"""
separate images input for hci dataset
'lf.png'

"""


def hci_input(image_path, view_n):

    RGB = [0.299, 0.587, 0.114]

    # print('Image loading in  {} '.format(image_path))

    if os.path.exists(image_path + '/lf.bmp'):
        tmp = np.float32(imageio.imread(image_path + '/lf.bmp'))
    elif os.path.exists(image_path + '/lf.png'):
        tmp = np.float32(imageio.imread(image_path + '/lf.png'))

    # print(np.max(tmp))

    image_h = tmp.shape[0]/view_n
    image_w = tmp.shape[1]/view_n

    if image_w % 2:
        new_tmp = tmp[9:-9, 18:-27, :]
        tmp = new_tmp
        image_h = tmp.shape[0] / view_n
        image_w = tmp.shape[1] / view_n

    gt_data_tmp = np.zeros((1, int(image_h), int(image_w)), dtype=np.float32)
    gt_data_tmp[0, :, :] = (RGB[0] * tmp[4::9, 4::9, 0] + RGB[1] * tmp[4::9, 4::9, 1] + RGB[2] * tmp[4::9, 4::9, 2]) / 255

    train_data_tmp = np.zeros((view_n, int(image_h/2), int(image_w/2)), dtype=np.float32)
    grey_tmp = RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]

    for i in range(0, 9, 1):
        img_tmp = grey_tmp[4::9, i::9]
        img = Image.fromarray(img_tmp)
        new_img = img.resize((int(image_w/2), int(image_h/2)), Image.BICUBIC)
        new_img.save('test.png')
        # img = new_img.resize((int(image_w), int(image_h)), Image.BICUBIC)
        tmp = np.asarray(new_img)
        train_data_tmp[i, :, :] = tmp/255

    del grey_tmp, tmp, img
    return train_data_tmp, gt_data_tmp