import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# Channel Attention Module simples
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn

# Spatial Attention Module simples
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.weight * noise

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, pixel_norm=True):
        super().__init__()
        convs = [EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)]
        convs.append(NoiseInjection(out_channel))  
        convs.append(nn.GroupNorm(8, out_channel))  
        convs.append(nn.LeakyReLU(0.2)) 
        convs.append(ChannelAttention(out_channel))
        convs.append(EqualConv2d(out_channel, out_channel, kernel_size, padding=padding))
        convs.append(nn.GroupNorm(8, out_channel))
        convs.append(nn.LeakyReLU(0.2))
        convs.append(SpatialAttention())
        convs.append(nn.Dropout2d(0.1))  
        self.conv = nn.Sequential(*convs)
    def forward(self, input):
        return checkpoint(self.conv, input, use_reentrant=False)

def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=128, tanh=True):
        super().__init__()
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            nn.GroupNorm(8, in_channel),
            nn.LeakyReLU(0.2))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1)
        self.progression_32 = ConvBlock(in_channel, in_channel // 2, 3, 1)
        self.progression_64 = ConvBlock(in_channel // 2, in_channel // 4, 3, 1)
        self.progression_128 = ConvBlock(in_channel // 4, in_channel // 8, 3, 1)
        self.progression_256 = ConvBlock(in_channel // 8, in_channel // 8, 3, 1)

        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel // 2, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel // 4, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel // 8, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel // 8, 3, 1)
        self.max_step = 6
    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out
    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out
    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step
        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)
        if step == 1:
            return self.output(out_4, out_8, self.to_rgb_8, self.to_rgb_8, alpha)
        out_16 = self.progress(out_8, self.progression_16)
        if step == 2:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)
        out_32 = self.progress(out_16, self.progression_32)
        if step == 3:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)
        out_64 = self.progress(out_32, self.progression_64)
        if step == 4:
            return self.output(out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha)
        out_128 = self.progress(out_64, self.progression_128)
        if step == 5:
            return self.output(out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha)
        out_256 = self.progress(out_128, self.progression_256)
        if step == 6:
            return self.output(out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha)

class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(feat_dim//4, feat_dim//4, 3, 1),
                                          ConvBlock(feat_dim//4, feat_dim//2, 3, 1),
                                          ConvBlock(feat_dim//2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          #ConvBlock(feat_dim+1, feat_dim, 3, 1, 4, 0)
                                          ConvBlock(feat_dim+1, feat_dim, 3, 1)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//2, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

       # self.linear = EqualLinear(feat_dim, 1)
        self.linear = EqualLinear(feat_dim * 4 * 4, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.view(out.size(0), -1)  # Flatten antes da camada linear
        out = self.linear(out)
        return out