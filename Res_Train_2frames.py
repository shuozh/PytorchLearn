import torch
import torch.utils.data
from LF_RES import LFRES
from datagen import LFDataset
import time
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import imageio
from math import log10
# from util import LrMultiStep

# dir_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/train/img/additional/' #training
# dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/'

dir_LFimages = 'training/'
dir_Test_LFimages = 'testing/'


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
image_h = 512
image_w = 512
view_n = 9


''' Define Model(set parameters)'''
filt_num = 64

model = LFRES(filt_num, view_n)
model.cuda()
print(model)

criterion = torch.nn.MSELoss()   # MSELoss or L1Loss?

learning_rate = 0.1 ** 5
base_lr = 0.001 #0.0003
momentum = 0.9
weight_decay = 0.0001
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

lr_steps = [4000, 5001, 10001]
lr_mults = [1, 1, 1]
last_iter = -1
# lr_scheduler = LrMultiStep(optimizer, lr_steps, lr_mults, last_iter=last_iter)

crop_size=35
current_iter = 1
epoch = 1


for epoch in range(current_iter, 1000001):

    for root, dirs, files in os.walk(dir_LFimages): # 当前路径下所有子目录

        total_loss = 0
        for image_path in dirs:

            if os.path.exists(os.path.join(root, image_path, '/input_Cam040.png')):
                break

            model.train()
            time_start = time.time()

            # Image Reading for training
            RGB = [0.299, 0.587, 0.114]

            traindata_tmp = np.zeros((view_n, image_h, image_w), dtype=np.float32)
            gtdata_tmp = np.zeros((1, image_h, image_w), dtype=np.float32)
            img_tmp = np.zeros((image_h, image_w, 3), dtype=np.float32)

            tmp = np.float32(imageio.imread(root + image_path + '/input_Cam040.png'))
            gtdata_tmp[0, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255

            seq = 44
            img = Image.open(root + image_path + '/input_Cam0%.2d.png' % seq)
            new_img = img.resize((int(image_h / 2), int(image_w / 2)), Image.BICUBIC)
            img = new_img.resize((image_h, image_w), Image.BICUBIC)
            tmp = np.asarray(img)

            for i in range(1,9):
                img_tmp[:,0:image_w-i,:] = tmp[:,i:image_w,:]
                traindata_tmp[i, :, :] = (RGB[0] * img_tmp[:, :, 0] + RGB[1] * img_tmp[:, :, 1] + RGB[2] * img_tmp[:, :, 2]) / 255

            seq = 40
            img = Image.open(root + image_path + '/input_Cam0%.2d.png' % seq)
            new_img = img.resize((int(image_h / 2), int(image_w / 2)), Image.BICUBIC)
            img = new_img.resize((image_h, image_w), Image.BICUBIC)
            tmp = np.asarray(img)
            traindata_tmp[0, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255


            train_dataset = LFDataset(traindata_tmp, gtdata_tmp, batch_size=32, view_n=view_n, crop_size=crop_size, If_flip=True, If_rotation=True, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

            # lr_scheduler.step(iter)
            # current_lr = lr_scheduler.get_lr()[0]

            for i, (traindata, gtdata_tmp) in enumerate(train_loader):
                traindata, gtdata_tmp = traindata.cuda(), gtdata_tmp.cuda()

                # Forward pass: Compute predicted y by passing x to the model
                gt_pred = model(traindata)

                # Compute and print loss
                loss = criterion(gt_pred, gtdata_tmp)
                total_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                time_end = time.time()

            #print('Train Epoch: {} Iter: {} Lr: {} Loss: {:.6f} Time: {:.2f}s'.format(epoch, current_iter, list(map(lambda group: group['lr'], optimizer.param_groups)), loss.item(), time_end - time_start))
            time_start = time.time()
            current_iter += 1

        print('Train Epoch: {} Average Loss: {:.6f} '.format(epoch, total_loss / len(dirs)))
        break

    avg_psnr = 0
    avg_loss = 0

    for root, dirs, files in os.walk(dir_Test_LFimages): # 当前路径下所有子目录

        if len(dirs) == 0:
            break

        for image_path in dirs:

            if os.path.exists(os.path.join(root, image_path, '/input_Cam040.png')):
                continue

            # Image Reading for testing
            RGB = [0.299, 0.587, 0.114]

            traindata_tmp = np.zeros((view_n, image_h, image_w), dtype=np.float32)
            gtdata_tmp = np.zeros((1, image_h, image_w), dtype=np.float32)

            tmp = np.float32(imageio.imread(root + image_path + '/input_Cam040.png'))
            gtdata_tmp[0, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :, 2]) / 255

            i = 0
            for seq in range(36, 45, 1):
                img = Image.open(root + image_path + '/input_Cam0%.2d.png' % seq)
                new_img = img.resize((int(image_h / 2), int(image_w / 2)), Image.BICUBIC)
                img = new_img.resize((image_h, image_w), Image.BICUBIC)
                tmp = np.asarray(img)
                # save the training image
                # if seq == 40:
                #     img.convert('L').save('%.2d_old.png' % epoch)
                traindata_tmp[i, :, :] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] + RGB[2] * tmp[:, :,2]) / 255
                i += 1


            test_dataset = LFDataset(traindata_tmp, gtdata_tmp, batch_size=1, view_n=view_n, crop_size=1, If_flip=False, If_rotation=False, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

            for i, (traindata, gtdata_tmp) in enumerate(test_loader):
                traindata, gtdata_tmp, = traindata.cuda(), gtdata_tmp.cuda()
                gt_pred = model(traindata)
                loss = criterion(gt_pred, gtdata_tmp)

                #print('Test Loss: {:.6f} '.format(loss.item()))

                avg_loss += loss
                psnr = 10 * log10(1 / loss.item())
                avg_psnr += psnr


                # output = gt_pred[0, :, :, :]
                # img = transforms.ToPILImage()(output.cpu())
                # img.save('%.2d_pre.png' % epoch)
                #
                # output = gtdata_tmp[0, :, :, :]
                # img = transforms.ToPILImage()(output.cpu())
                # img.save('%.2d_gt.png' % epoch)

        break

    print('===> Avg. PSNR: {:.4f} dB Avg. Loss: {:.6f}'.format(avg_psnr/len(dirs), avg_loss/len(dirs)))
    torch.save(model.state_dict(), 'model/LFSR' + str(epoch) + '.pkl')





