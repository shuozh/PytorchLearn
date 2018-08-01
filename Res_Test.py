import torch
import torch.utils.data
from LF_RES import LFRES
from datagen import LFDataset
from math import log10
from func_hci_new_input import hci_new_input
from func_hci_input import hci_input

# dir_Test_LFimages = '/home/hawkeye948-3/cz/EPINET_DATA/test/boxes/'
dir_Test_LFimages = 'hci_testing/boxes/'
view_n = 9

''' Define Model(set parameters)'''
filter_num = 64

model = LFRES(filter_num, view_n)
model.cuda()
criterion = torch.nn.MSELoss()   # or L1Loss?

state_dict = torch.load('model/LFSR500.pkl')
model.load_state_dict(state_dict)

# print(model)

# params = model.state_dict()
# for k, v in params.items():
#     print(k)    #打印网络中的变量名
# print(params['layer_conv.S1_c9.weight'])   #打印conv1的weight
# print(params['layer_conv.S1_c9.bias'])   #打印conv1的bias

# Image loading
train_data_tmp, gt_data_tmp = hci_input(dir_Test_LFimages, view_n)

train_dataset = LFDataset(train_data_tmp, gt_data_tmp, batch_size=1, view_n=view_n, crop_size=1, If_flip=False, If_rotation=False, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

for i, (train_data, gt_data) in enumerate(train_loader):
    train_data, gt_data = train_data.cuda(), gt_data.cuda()

    # Forward pass: Compute predicted y by passing x to the model
    gt_pred = model(train_data)

    # for evaluating the original resize image
    # gt_compare = train_data_tmp[4, :, :]
    # gt_compare = torch.from_numpy(gt_compare.copy())
    # gt_compare = gt_compare.cuda()
    # gt_pred[0,0,:,:] = gt_compare

    # Compute and print loss
    loss = criterion(gt_pred, gt_data)

    psnr = 10 * log10(1 / loss.item())

    # save the super-resolution images
    # output = gt_pred[0, :, :, :]
    # img = transforms.ToPILImage()(output.cpu())
    # img.save('pre.png')
    #
    # output = gt_data[0, :, :, :]
    # img = transforms.ToPILImage()(output.cpu())
    # img.save('gt.png')

    print('===> Avg. PSNR: {:.4f} dB Avg. Loss: {:.6f}'.format(psnr, loss.item()))



