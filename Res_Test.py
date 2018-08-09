import torch
import torch.utils.data
from PIL import Image
from LF_RES import LFRES
from torchvision import transforms
from datagen import LFDataset
from math import log10
from func_hci_input import hci_input
from SSIM import compute_ssim
import os
from LF_ER import EDSR
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def main():

    # dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/boxes/'
    dir_Test_LFimages = 'hci_training/StillLife/'
    view_n = 9

    ''' Define Model(set parameters)'''
    filter_num = 64

    # model = LFRES(filter_num, view_n)
    model = EDSR()
    model.cuda()
    criterion = torch.nn.MSELoss()   # or L1Loss?

    state_dict = torch.load('LFSR80.pkl')
    model.load_state_dict(state_dict)

    # print(model)

    # params = model.state_dict()
    # for k, v in params.items():
    #     print(k)    #打印网络中的变量名
    # print(params['layer_conv.S1_c9.weight'])   #打印conv1的weight
    # print(params['layer_conv.S1_c9.bias'])   #打印conv1的bias

    # Image loading
    train_data_tmp, gt_data_tmp = hci_input(dir_Test_LFimages, view_n)

    train_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=1, view_n=view_n, crop_size=1, If_flip=False,
                              If_rotation=False, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    with torch.no_grad():
        for i, (train_data, gt_data) in enumerate(train_loader):
            train_data, gt_data = train_data.cuda(), gt_data.cuda()

            # Forward pass: Compute predicted y by passing x to the model
            gt_pred = model(train_data)

            # Compute and print loss
            loss = criterion(gt_pred, gt_data)
            psnr = 10 * log10(1 / loss.item())

            ssim = compute_ssim((gt_pred[0, 0, :, :].cpu()), (gt_data[0, 0, :, :].cpu()))

            # save the super-resolution images
            output = gt_pred[0, :, :, :]
            print(np.max(output.cpu().numpy()))
            print(np.min(output.cpu().numpy()))
            img = transforms.ToPILImage()(output.cpu())
            # print(np.max(img))
            img.save('pre.png')

            output = gt_data[0, :, :, :]
            print(np.max(output.cpu().numpy()))
            print(np.min(output.cpu().numpy()))
            img = transforms.ToPILImage()(output.cpu())
            img.save('gt.png')

            print('===>SR PSNR: {:.4f} dB, SSIM: {:.4f}  Loss: {:.6f}'.format(psnr, ssim, loss.item()))

            # for evaluating the original resize image
            output = train_data[0, 4:5, :, :]
            img = transforms.ToPILImage()(output.cpu())
            image_h = output.shape[1]
            image_w = output.shape[2]
            new_img = img.resize((int(image_w*2), int(image_h*2)), Image.BICUBIC)
            new_img.save('compare.png')

            gt_img = gt_data[0, 0, :, :]
            compare_loss = (np.array(new_img)/255 - gt_img.cpu().numpy()) ** 2
            compare_loss = compare_loss.sum()/(int(image_w*2)*int(image_h*2))
            psnr = 10 * log10(1 / compare_loss)
            ssim = compute_ssim(np.array(new_img)/255, gt_img.cpu().numpy())

            print('===>Compared PSNR: {:.4f} dB, SSIM: {:.4f}  Loss: {:.6f}'.format(psnr, ssim, compare_loss))


if __name__ == '__main__':
    main()
