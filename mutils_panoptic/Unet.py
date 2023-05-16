# Source: https://github.com/choosehappy/PytorchDigitalPathology/blob/ ...
# master/segmentation_epistroma_unet/

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        external_concat_layer=None,
        external_concat_nc=None,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            external_concat_layer (int): TODO: explain me!
            external_concat_nc (int): TODO: explain me!
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth

        # Mohamed: possibly enable concatenating ecternal convolutional maps
        # to the intermediate feature maps. I use this feature concatenate
        # features fromt he low magnification UNet to the HPF UNet. See
        # HookNet paper for details. This obly applies to the deconv. path
        self.ecl = external_concat_layer
        self.ecnc = external_concat_nc
        self.econcat = self.ecl is not None

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):

            # Mohamed: account for external concat
            if self.econcat and (depth - i - 2 == self.ecl):
                in_size = prev_channels + self.ecnc
                stage2_in_size = prev_channels
            else:
                in_size = stage2_in_size = prev_channels

            self.up_path.append(UNetUpBlock(
                in_size=in_size,
                out_size=2**(wf+i),
                up_mode=up_mode,
                padding=padding,
                batch_norm=batch_norm,
                stage2_in_size=stage2_in_size,
            ))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, fetch_layers: List[int] = None, cx: Tensor = None):
        if self.econcat:
            assert cx is not None, "Need external tensor to be concatenated!!"
        fetch = fetch_layers is not None
        rd = 0  # real depth (encoder + decoder)
        xf = []
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
            # Mohamed: maybe fetch intermediate layer
            if fetch:
                if rd in fetch_layers:
                    xf.append(0. + x)
            rd += 1

        for i, up in enumerate(self.up_path):

            # Mohamed: maybe concat external tensor (extra channels)
            if self.econcat and (i == self.ecl):
                x = torch.cat([x, cx], dim=1)

            x = up(x, blocks[-i-1])

            # Mohamed: maybe fetch intermediate layer
            if fetch:
                if rd in fetch_layers:
                    xf.append(0. + x)
            rd += 1

        # final layer
        x = self.last(x)

        if fetch:
            return x, xf
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(
            self, in_size, out_size, up_mode, padding, batch_norm,
            stage2_in_size=None):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1))

        stage2_in_size = stage2_in_size or in_size
        self.conv_block = UNetConvBlock(
            stage2_in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(
            diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
