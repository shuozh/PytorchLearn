import common
import torch.nn as nn


class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR, self).__init__()

        # n_resblock = args.n_resblocks
        n_resblock = 8
        # n_feats = args.n_feats
        n_feats = 64
        kernel_size = 3
        # scale = args.scale[0]
        scale = 2
        act = nn.ReLU(True)
        # n_colors = args.n_colors
        n_colors = 1
        n_view = 9
        # args.rgb_range
        rgb_range = 255

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module

        m_head = [conv(n_view, n_feats, kernel_size)]
        central_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        # res_scale = args.res_scale
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.central_head = nn.Sequential(*central_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    # def forward(self, x):
    #     # x = self.sub_mean(x)
    #     x = self.head(x)
    #
    #     res = self.body(x)
    #     res += x
    #
    #     x = self.tail(res)
    #     # x = self.add_mean(x)
    #
    #     return x

    def forward(self, x):

        central_x = x[:, 4:5, :, :]
        res_x = self.central_head(central_x)

        y = self.head(x)
        res = self.body(y)

        res += res_x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

