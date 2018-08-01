import torch
import torch.nn as nn
import torch.nn.init as init

def layer_conv(filt_num, in_channels):
    seq = nn.Sequential()
    ''' Conv layer : Conv - Relu '''
    
    seq.add_module('S1_c', nn.Conv2d(in_channels, filt_num, kernel_size=3, stride=1, padding=1))
    seq.add_module('S1_relu', nn.ReLU())

    for i in range(10): #16
        seq.add_module('S1_c%d' % i, nn.Conv2d(filt_num, filt_num, kernel_size=3, stride=1, padding=1))
        seq.add_module('S1_relu%d' % i, nn.ReLU())
    return seq


class LFRES(nn.Module):
    def __init__(self, filt_num, in_channels):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LFRES, self).__init__()

        self.layer_conv = layer_conv(filt_num, in_channels)
        self.layer_resi = nn.Conv2d(filt_num, 1, kernel_size=3, stride=1, padding=1)

        init.xavier_normal_(self.layer_resi.weight.data, gain=nn.init.calculate_gain('relu'))


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """


        residual = x
        mid_output = self.layer_conv(x)
        output = self.layer_resi(mid_output)

        resi_out = residual[:,4:5,:,:] #4:5 0:1
        output += resi_out

        return output

