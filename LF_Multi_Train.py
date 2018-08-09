import torch
import torch.utils.data
from LF_Multi_Res import LF_Multi_Res
from LF_Multi_datagen import LF_Multi_Dataset
import time
import os
import csv
from math import log10
from initializers import weights_init_xavier
from LF_Multi_hci_input import multi_hci_input

# from util import LrMultiStep

# dir_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/train/img/additional/' #training
# dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main():

    dir_LFimages = 'hci_training/'
    dir_Test_LFimages = 'hci_testing/'
    view_n = 9

    ''' Define Model(set parameters)'''
    filter_num = 64

    model = LF_Multi_Res(filter_num, view_n, conv_depth=10)
    model.apply(weights_init_xavier)
    model.cuda()
    print(model)

    criterion = torch.nn.MSELoss()   # MSELoss or L1Loss?

    base_lr = 0.001 # 0.0003
    momentum = 0.9
    weight_decay = 0.0001
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    lr_steps = [4000, 5001, 10001]
    lr_mults = [1, 1, 1]
    last_iter = -1
    # lr_scheduler = LrMultiStep(optimizer, lr_steps, lr_mults, last_iter=last_iter)

    # test_res(dir_Test_LFimages, model, criterion, view_n=view_n)
    current_iter = 1
    f_train = open('model_train_loss.csv', 'a')
    writer_train = csv.writer(f_train)
    f_test = open('model_test_loss.csv', 'a')
    writer_test = csv.writer(f_test)
    for epoch in range(current_iter, 1000001):
        current_iter, train_loss = train_res(dir_LFimages, model, epoch, criterion, optimizer, current_iter, view_n)
        writer_train.writerow([epoch, 'train loss', train_loss])
        if epoch % 5 == 0:
            test_psnr, test_loss = test_res(dir_Test_LFimages, model, criterion, view_n)
            writer_test.writerow([epoch, 'test loss', test_loss, 'test psnr', float('%.4f' % test_psnr)])
        if epoch % 100 == 0:
            torch.save(model.state_dict(), 'model/LFSR' + str(epoch) + '.pkl')


def train_res(dir_LFimages, model, epoch, criterion, optimizer, current_iter, view_n):

    crop_size = 35
    time_start = time.time()
    for root, dirs, files in os.walk(dir_LFimages):  # 当前路径下所有子目录

        total_loss = 0
        for image_path in dirs:

            if os.path.exists(os.path.join(root, image_path, '/input_Cam040.png')):
                break

            model.train()

            # Image loading
            train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp = \
                multi_hci_input(root + image_path, view_n)

            train_dataset = LF_Multi_Dataset(train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp,
                                             batch_size=64, view_n=view_n, crop_size=crop_size, if_flip=True,
                                             if_rotation=True, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

            # lr_scheduler.step(iter)
            # current_lr = lr_scheduler.get_lr()[0]

            for i, (train_data_0, train_data_90, train_data_45, train_data_135, gt_data) in enumerate(train_loader):
                train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
                    train_data_0.cuda(), train_data_90.cuda(), train_data_45.cuda(), train_data_135.cuda(), gt_data.cuda()

                # Forward pass: Compute predicted y by passing x to the model
                gt_pred = model(train_data_0, train_data_90, train_data_45, train_data_135)

                # Compute and print loss
                loss = criterion(gt_pred, gt_data)
                total_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print('Train Epoch: {} Iter: {} Lr: {} Loss: {:.6f} Time: {:.2f}s'.format(epoch, current_iter,
            # list(map(lambda group: group['lr'], optimizer.param_groups)), loss.item(), time_end - time_start))

            current_iter += 1

        time_end = time.time()
        print('Train Epoch: {} Average Loss: {:.6f} Time: {:.2f}s'.format(epoch, total_loss / len(dirs), time_end - time_start))
        return current_iter, total_loss / len(dirs)


def test_res(dir_Test_LFimages, model, criterion, view_n):

    avg_psnr = 0
    avg_loss = 0

    # 当前路径下所有子目录
    for root, dirs, files in os.walk(dir_Test_LFimages):
        model.eval()
        if len(dirs) == 0:
            break
        for image_path in dirs:

            # Image loading
            train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp = \
                multi_hci_input(root + image_path, view_n)

            test_dataset = LF_Multi_Dataset(train_data_0, train_data_90, train_data_45, train_data_135, gt_data_tmp,
                                            batch_size=1, view_n=view_n, crop_size=1, if_flip=False, if_rotation=False,
                                            transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

            with torch.no_grad():
                for i, (train_data_0, train_data_90, train_data_45, train_data_135, gt_data) in enumerate(test_loader):
                    train_data_0, train_data_90, train_data_45, train_data_135, gt_data = \
                        train_data_0.cuda(), train_data_90.cuda(), train_data_45.cuda(), train_data_135.cuda(), gt_data.cuda()

                    # Forward pass: Compute predicted y by passing x to the model
                    gt_pred = model(train_data_0, train_data_90, train_data_45, train_data_135)

                    loss = criterion(gt_pred, gt_data)

                    print('Test Loss: {:.6f} in {}'.format(loss.item(), image_path))

                    avg_loss += loss.item()
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

    print('===> Avg. PSNR: {:.4f} dB Avg. Loss: {:.6f}'.format(avg_psnr / len(dirs), avg_loss / len(dirs)))
    return avg_psnr / len(dirs), avg_loss / len(dirs)


if __name__ == '__main__':
    main()