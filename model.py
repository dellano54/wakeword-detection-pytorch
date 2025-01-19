import torch.nn as nn
import torch

class Dropout1d(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.p
        if self.inplace:
            x.mul_(mask)
        else:
            x = x * mask
        return x / (1 - self.p)

class CONV_LAYER(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: float = 0.3):
        super().__init__()
        '''
        This an base conv1d module which uses convolution operation through the time dimension reducing the size
        INPUTS:

            in_channels: input channels size
            out_channels: output channels size
            kernel_size: kernel_size of the model
            stride: the skip size between convolution
            dropout: dropout in input

        ARCH:

            conv1d -> BatchNorm -> silu -> dropout
        '''

        self.Conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=kernel_size*stride)
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.drop = Dropout1d(dropout)

    def forward(self, x):
        x = self.Conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    

class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x
    
class StackedConvNet(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, pool_size, embed_dim, num_layers):
        super(StackedConvNet, self).__init__()
        layers = []
        current_channels = in_channels
        initial_kernel_size = 6  # starting kernel size
        initial_stride = 4  # starting stride
        kernel_decay_factor = 2
        stride_decay_factor = 2

        for layer_idx in range(num_layers):
            kernel_size = max(2, initial_kernel_size // (kernel_decay_factor ** layer_idx))
            stride = max(1, initial_stride // (stride_decay_factor ** layer_idx))
            layers.append(CONV_LAYER(current_channels, intermediate_channels, kernel_size=kernel_size, stride=stride))
            current_channels = intermediate_channels
            intermediate_channels = max(out_channels, intermediate_channels // 2)  # decay intermediate channels

        # Last layer
        layers.append(nn.Conv1d(current_channels, embed_dim, kernel_size=1, stride=1))

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.pix_shuf = PixelShuffle1D(embed_dim)

        self.class_fc = nn.Linear(pool_size*embed_dim, 1)

    def forward(self, x):
        return self.class_fc(self.pix_shuf(self.pool(self.network(x))).squeeze(1))
    

