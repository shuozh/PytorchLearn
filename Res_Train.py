import torch
import torch.utils.data
from LF_RES import LFRES
from datagen import LFDataset
import time
import os
from torchvision import transforms
from math import log10
from initializers import weights_init_xavier
from func_hci_new_input import hci_new_input
from func_hci_input import hci_input

# from util import LrMultiStep

# dir_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/train/img/additional/' #training
# dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/'

dir_LFimages = 'training/'
dir_Test_LFimages = 'hci_testing/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
view_n = 9


''' Define Model(set parameters)'''
filter_num = 64

model = LFRES(filter_num, view_n)
model.apply(weights_init_xavier)
model.cuda()
print(model)

criterion = torch.nn.MSELoss()   # MSELoss or L1Loss?
criterion_test = torch.nn.MSELoss()

learning_rate = 0.1 ** 5
base_lr = 0.001 # 0.0003
momentum = 0.9
weight_decay = 0.0001
# optimizer = torch.optim.RMSprop(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

lr_steps = [4000, 5001, 10001]
lr_mults = [1, 1, 1]
last_iter = -1
# lr_scheduler = LrMultiStep(optimizer, lr_steps, lr_mults, last_iter=last_iter)

crop_size = 35
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

            # Image loading
            train_data_tmp, gt_data_tmp = hci_new_input(root+image_path, view_n)

            train_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=64, view_n=view_n, crop_size=crop_size, If_flip=True, If_rotation=True, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

            # lr_scheduler.step(iter)
            # current_lr = lr_scheduler.get_lr()[0]

            for i, (train_data, gt_data) in enumerate(train_loader):
                train_data, gt_data = train_data.cuda(), gt_data.cuda()

                # Forward pass: Compute predicted y by passing x to the model
                gt_pred = model(train_data)

                # Compute and print loss
                loss = criterion(gt_pred, gt_data)
                total_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                time_end = time.time()

            # print('Train Epoch: {} Iter: {} Lr: {} Loss: {:.6f} Time: {:.2f}s'.format(epoch, current_iter,
            # list(map(lambda group: group['lr'], optimizer.param_groups)), loss.item(), time_end - time_start))
            time_start = time.time()
            current_iter += 1

        print('Train Epoch: {} Average Loss: {:.6f} '.format(epoch, total_loss / len(dirs)))
        break

    avg_psnr = 0
    avg_loss = 0

    # 当前路径下所有子目录
    for root, dirs, files in os.walk(dir_Test_LFimages):
        model.eval()
        if len(dirs) == 0:
            break
        for image_path in dirs:

            # Image loading
            train_data_tmp, gt_data_tmp = hci_input(root + image_path, view_n)

            test_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=1, view_n=view_n, crop_size=1, If_flip=False, If_rotation=False, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

            del train_data_tmp, gt_data_tmp

            for i, (train_data, gt_data) in enumerate(test_loader):
                train_data, gt_data, = train_data.cuda(), gt_data.cuda()
                gt_pred = model(train_data)
                loss = criterion_test(gt_pred, gt_data)

                print('Test Loss: {:.6f} '.format(loss.item()))

                avg_loss += loss
                psnr = 10 * log10(1 / loss.item())
                avg_psnr += psnr

                # output = gt_pred[0, :, :, :]
                # img = transforms.ToPILImage()(output.cpu())
                # img.save('%.2d_pre.png' % epoch)
                #
                # output = gt_data[0, :, :, :]
                # img = transforms.ToPILImage()(output.cpu())
                # img.save('%.2d_gt.png' % epoch)

        break

    print('===> Avg. PSNR: {:.4f} dB Avg. Loss: {:.6f}'.format(avg_psnr/len(dirs), avg_loss/len(dirs)))
    if epoch % 100 == 0:
        torch.save(model.state_dict(), 'model/LFSR' + str(epoch) + '.pkl')





