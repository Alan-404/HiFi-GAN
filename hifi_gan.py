import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: Union[list, tuple] = (1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilations[0],
                               padding=get_padding(kernel_size, dilations[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilations[1],
                               padding=get_padding(kernel_size, dilations[1])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilations[2],
                               padding=get_padding(kernel_size, dilations[2]))
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(torch.nn.Module):
    def __init__(self, n_mel_channels: int, upsample_rates: Union[list, tuple], upsample_kernel_sizes: Union[list, tuple], upsample_initial_channel: int, resblock_kernel_sizes: Union[list, tuple], resblock_dilation_sizes: Union[list, tuple]):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(n_mel_channels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)


    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
    
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period

        # norm_func = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(kernel_size, 1), stride=(stride, 1) , padding=(get_padding(5, 1), 0)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(kernel_size, 1),stride=(stride, 1) , padding=(get_padding(5, 1), 0)),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(kernel_size, 1),stride=(stride, 1) , padding=(get_padding(5, 1), 0)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(kernel_size, 1),stride=(stride, 1) , padding=(get_padding(5, 1), 0)),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(kernel_size, 1),stride=1, padding=(2, 0))
        ])

        self.conv_post = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 1), stride=1, padding=(1,0))

    def forward(self, x: torch.Tensor):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    
class MultiPeriodDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period=2),
            PeriodDiscriminator(period=3),
            PeriodDiscriminator(period=5),
            PeriodDiscriminator(period=5),
            PeriodDiscriminator(period=11)
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        num_pad = y_hat.size(-1) - y.size(-1)
        padded_y = F.pad(y, pad=(0, num_pad), mode='reflect')

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(padded_y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    
class ScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=41, stride=2, groups=4, padding=20),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=41, stride=2, groups=16, padding=20),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=41, stride=2, groups=16, padding=20),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, groups=16, padding=20),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=1, groups=16, padding=20),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2)
        ])
    
        self.conv_post = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    
class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        num_pad = y_hat.size(-1) - y.size(-1)
        y = F.pad(y, pad=(0, num_pad), mode='reflect')
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def get_padding(kernel_size: int, dilation: int):
    return int((kernel_size * dilation - dilation)/2)