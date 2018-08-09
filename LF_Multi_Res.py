import torch
import torch.nn as nn


def layer1_multistream(filt_num, in_channels):
    seq = nn.Sequential()
    ''' Conv layer : Conv - Relu '''

    seq.add_module('S1_c', nn.Conv2d(in_channels, filt_num, kernel_size=3, stride=1, padding=1))
    seq.add_module('S1_relu', nn.ReLU())

    for i in range(5):
        seq.add_module('S1_c%d' % i, nn.Conv2d(filt_num, filt_num, kernel_size=3, stride=1, padding=1))
        seq.add_module('S1_relu%d' % i, nn.ReLU())
    return seq


def layer2_merged(filt_num, conv_depth):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''

    seq = nn.Sequential()

    for i in range(conv_depth):
        seq.add_module('S2_c%d' % i, nn.Conv2d(filt_num, filt_num, kernel_size=3, stride=1, padding=1))
        seq.add_module('S2_relu%d' % i, nn.ReLU())
    return seq


def layer3_last(filt_num):
    ''' last layer : Conv '''

    seq = nn.Sequential()
    seq.add_module('S3_last', nn.Conv2d(filt_num, 1, kernel_size=3, stride=1, padding=1))

    return seq


class LF_Multi_Res(nn.Module):
    def __init__(self, filt_num, in_channels, conv_depth):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LF_Multi_Res, self).__init__()

        self.layer1_multistream = layer1_multistream(filt_num, in_channels)
        self.layer2_merged = layer2_merged(int(4 * filt_num), conv_depth)
        self.layer3_last = layer3_last(int(4 * filt_num))

    def forward(self, train_data_0, train_data_90, train_data_45, train_data_135):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        residual = train_data_0[:, 4:5, :, :]

        ''' 4-Stream layer : Conv - Relu'''
        mid_0d = self.layer1_multistream(train_data_0)
        mid_90d = self.layer1_multistream(train_data_90)
        mid_45d = self.layer1_multistream(train_data_45)
        mid_135d = self.layer1_multistream(train_data_135)

        ''' Merge layers '''
        mid_merged = torch.cat((mid_0d, mid_90d, mid_45d, mid_135d), 1)

        ''' Merged layer : Conv - Relu'''
        mid_merged_ = self.layer2_merged(mid_merged)

        ''' Last Conv layer : Conv - Relu - Conv '''
        output = self.layer3_last(mid_merged_)

        output += residual

        return output


