import torch
import torch.utils.data
from LF_RES import LFRES
from LF_ER import EDSR
from datagen import LFDataset
import time
import os
import csv
from torchvision import transforms
from math import log10
from initializers import weights_init_xavier
from func_hci_input import hci_input
import util
# from util import LrMultiStep

# dir_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/train/img/additional/' #training
# dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def main():

    dir_LFimages = 'hci_training/'
    dir_Test_LFimages = 'hci_testing/'
    view_n = 9

    ''' Define Model(set parameters)'''
    filter_num = 64
    crop_size = 35

    # model = LFRES(filter_num, view_n)
    model = EDSR()
    model.apply(weights_init_xavier)
    model.cuda()
    print(model)

    # state_dict = torch.load('LFSR490.pkl')
    # model.load_state_dict(state_dict)

    criterion = torch.nn.MSELoss()   # MSELoss or L1Loss?
    # criterion_test = torch.nn.L1Loss()

    base_lr = 0.0003 # 0.0003
    momentum = 0.9
    weight_decay = 0.0001
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    lr_steps = [4000, 5001, 10001]
    lr_mults = [1, 1, 1]
    last_iter = -1
    # lr_scheduler = LrMultiStep(optimizer, lr_steps, lr_mults, last_iter=last_iter)

    # test_res(dir_Test_LFimages, model, criterion, view_n=view_n)
    error_last = 1e2
    PSNR_Best = 0
    current_iter = 1
    f_train = open('model_train_loss.csv', 'a')
    writer_train = csv.writer(f_train)
    f_test = open('model_test_loss.csv', 'a')
    writer_test = csv.writer(f_test)
    for epoch in range(current_iter, 1000001):
        current_iter, error_last, train_loss = train_res(dir_LFimages, model, epoch, criterion, optimizer, current_iter, view_n, crop_size, error_last)
        writer_train.writerow([epoch, 'train loss', train_loss])
        if epoch % 1 == 0:
            PSNR_Best, test_psnr, test_loss = test_res(dir_Test_LFimages, model, criterion, view_n, PSNR_Best)
            writer_test.writerow([epoch, 'test loss', test_loss, 'test psnr', float('%.4f' % test_psnr)])
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'model/LFSR' + str(epoch) + '.pkl')


def train_res(dir_LFimages, model, epoch, criterion, optimizer, current_iter, view_n, crop_size, error_last):

    scheduler = util.make_scheduler(optimizer)
    lr = scheduler.get_lr()[0]
    model.train()

    for root, dirs, files in os.walk(dir_LFimages):  # 当前路径下所有子目录
        time_start = time.time()
        total_loss = 0

        for image_path in dirs:
            # Image loading
            train_data_tmp, gt_data_tmp = hci_input(root + image_path, view_n)
            train_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=32, view_n=view_n, crop_size=crop_size,
                                      If_flip=True, If_rotation=True, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

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

                # skipping batch that has large error
                skip_threshold = 1e2
                if loss.item() < skip_threshold * error_last:
                    loss.backward()
                    optimizer.step()
                else:
                    print('Skip batch {} in {} ! (Loss: {})'.format(current_iter + 1, image_path, loss.item()))

                if loss.item() > 100 * error_last:
                    print('Bad batch {} in {} ! (Loss: {})'.format(current_iter + 1, image_path, loss.item()))

                error_last = loss.item()

            # print('Train Epoch: {} Iter: {} Loss: {:.6f} in {}'.format(epoch, current_iter, loss.item(), image_path))
            current_iter += 1

        time_end = time.time()
        print('=========================================================================')
        print('Train Epoch: {} Learning rate: {:.2e} Time: {:.2f}s Average Loss: {:.6f} '
              .format(epoch, lr, time_end - time_start, total_loss / len(dirs)))
        # print('=========================================================================')
        return current_iter, error_last, total_loss / len(dirs)


def test_res(dir_Test_LFimages, model, criterion, view_n, PSNR_Best):

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

            test_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=1, view_n=view_n, crop_size=1,
                                     If_flip=False, If_rotation=False, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

            del train_data_tmp, gt_data_tmp

            with torch.no_grad():
                for i, (train_data, gt_data) in enumerate(test_loader):
                    train_data, gt_data, = train_data.cuda(), gt_data.cuda()
                    gt_pred = model(train_data)
                    loss = criterion(gt_pred, gt_data)
                    # loss = criterion_test(gt_pred, gt_data)
                    avg_loss += loss.item()
                    psnr = 10 * log10(1 / loss.item())
                    avg_psnr += psnr

                    print('Test Loss: {:.6f}, PSNR: {:.4f} in {}'.format(loss.item(), psnr, image_path))

        break

    if (avg_psnr/len(dirs)) > PSNR_Best:
        PSNR_Best = avg_psnr/len(dirs)

    print('===> Avg. PSNR: {:.4f} dB / BEST {:.4f} dB Avg. Loss: {:.6f}'.format(avg_psnr / len(dirs), PSNR_Best, avg_loss / len(dirs)))
    return PSNR_Best, avg_psnr / len(dirs), avg_loss / len(dirs)


if __name__ == '__main__':
    main()