# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Neural network layers"""

import copy
from typing import Optional, Tuple

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class GatherExcite(nn.Module):
    """Gather-Excite module (https://arxiv.org/abs/1810.12348),

    a generalization of the Squeeze-and-Excitation module
    (https://arxiv.org/abs/1709.01507).

    Args:
        channels: Number of input channels (= number of output channels)
        extent: extent factor that determines how much the gather operator
            output is smaller than its input. The special value ``extent=0``
            activates global gathering (so the gathered information has no
            spatial extent).
        param_gather: If ``True``, the gather operator is parametrized
            according to https://arxiv.org/abs/1810.12348.
        param_excite: If ``True``, the excitation operator is parametrized
            according to https://arxiv.org/abs/1810.12348 (also equivalent to
            the original excitation operator proposed in
            https://arxiv.org/abs/1709.01507).
        reduction:  Channel reduction rate of the parametrized excitation
            operator.
        spatial_shape: Spatial shape of the module input. This needs to be
            specified if ``param_gather=0 and extent=0`` (parametrized global
            gathering).
    """
    def __init__(
            self,
            channels: int,
            extent: int = 0,
            param_gather: bool = False,
            param_excite: bool = True,
            reduction: int = 16,
            spatial_shape: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()
        if extent == 1:
            raise NotImplementedError('extent == 1 doesn\'t make sense.')
        if param_gather:
            if extent == 0:  # Global parametrized gather operator
                if spatial_shape is None:
                    raise ValueError(
                        'With param_gather=True, extent=0, you will need to specify spatial_shape.')
                self.gather = nn.Sequential(
                    nn.Conv3d(channels, channels, spatial_shape),
                    nn.BatchNorm3d(channels),
                    nn.ReLU()
                )
            else:
                # This will make the model much larger with growing extent!
                # TODO: This is ugly and I'm not sure if it should even be supported
                assert extent in [2, 4, 8, 16]
                num_convs = int(torch.log2(torch.tensor(extent, dtype=torch.float32)))
                self.gather = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(channels, channels, 3, stride=2, padding=1),
                        nn.BatchNorm3d(channels),
                        nn.ReLU()
                    ) for _ in range(num_convs)
                ])
        else:
            if extent == 0:
                self.gather = nn.AdaptiveAvgPool3d(1)  # Global average pooling
            else:
                self.gather = nn.AvgPool3d(extent)
        if param_excite:
            self.excite = nn.Sequential(
                nn.Conv3d(channels, channels // reduction, 1),
                nn.ReLU(),
                nn.Conv3d(channels // reduction, channels, 1)
            )
        else:
            self.excite = Identity()

        if extent == 0:
            self.interpolate = Identity()  # Use broadcasting instead of interpolation
        else:
            self.interpolate = torch.nn.functional.interpolate

    def forward(self, x):
        y = self.gather(x)
        y = self.excite(y)
        y = torch.sigmoid(self.interpolate(y, x.shape[2:]))
        return x * y


class AdaptiveConv3d(nn.Module):
    """Equivalent to ``torch.nn.Conv3d`` except that if
    ``kernel_size[0] == 1``, ``torch.nn.Conv2d`` is used internally in
    order to improve computation speed.

    This is a workaround for https://github.com/pytorch/pytorch/issues/7740.

    Current limitations:
    - Expects ``kernel_size`` to be passed as a keyword arg, not positional."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        ks = kwargs['kernel_size']
        if isinstance(ks, tuple) and ks[0] == 1:
            kwargs['kernel_size'] = ks[1:]
            kwargs['stride'] = kwargs.get('stride', (0, 1, 1))[1:]
            kwargs['padding'] = kwargs.get('padding', (0, 0, 0))[1:]
            kwargs['dilation'] = kwargs.get('dilation', (1, 1, 1))[1:]
            self.conv = nn.Conv2d(*args, **kwargs)
            self.forward = self.forward2d
        else:
            self.conv = nn.Conv3d(*args, **kwargs)
            self.forward = self.forward3d

    def forward2d(self, x):
        n, c, d, h, w = x.shape
        transp = x.transpose(1, 2)  # -> (N, D, C, H, W)
        view2d = transp.reshape(n * d, c, h, w)  # -> (N * D, C, H, W)
        out2dtransp = self.conv(view2d)
        h, w = out2dtransp.shape[-2:]  # H and W can be changed due to convolution
        c = self.conv.out_channels
        out3dtransp = out2dtransp.reshape(n, d, c, h, w)  # -> (N, D, C, H, W)
        out3d = out3dtransp.transpose(1, 2)  # -> (N, C, D, H, W)

        return out3d

    def forward3d(self, x):
        return self.conv(x)

    def forward(self, x): raise NotImplementedError()  # Chosen by __init__()


class AdaptiveConvTranspose3d(nn.Module):
    """Equivalent to ``torch.nn.ConvTranspose3d`` except that if
    ``kernel_size[0] == 1``, ``torch.nn.ConvTranspose2d`` is used internally in
    order to improve computation speed.

    This is a workaround for https://github.com/pytorch/pytorch/issues/7740.

    Current limitations:
    - Expects ``kernel_size`` to be passed as a keyword arg, not positional."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        ks = kwargs['kernel_size']
        if isinstance(ks, tuple) and ks[0] == 1:
            kwargs['kernel_size'] = ks[1:]
            kwargs['stride'] = kwargs.get('stride', (0, 1, 1))[1:]
            kwargs['padding'] = kwargs.get('padding', (0, 0, 0))[1:]
            kwargs['dilation'] = kwargs.get('dilation', (1, 1, 1))[1:]
            self.conv = nn.ConvTranspose2d(*args, **kwargs)
            self.forward = self.forward2d
        else:
            self.conv = nn.ConvTranspose3d(*args, **kwargs)
            self.forward = self.forward3d

    def forward2d(self, x):
        n, c, d, h, w = x.shape
        transp = x.transpose(1, 2)  # -> (N, D, C, H, W)
        view2d = transp.reshape(n * d, c, h, w)  # -> (N * D, C, H, W)
        out2dtransp = self.conv(view2d)
        h, w = out2dtransp.shape[-2:]  # H and W can be changed due to convolution
        c = self.conv.out_channels
        out3dtransp = out2dtransp.reshape(n, d, c, h, w)  # -> (N, D, C, H, W)
        out3d = out3dtransp.transpose(1, 2)  # -> (N, C, D, H, W)

        return out3d

    def forward3d(self, x):
        return self.conv(x)

    def forward(self, x): raise NotImplementedError()  # Chosen by __init__()


# Small helper functions for abstracting over 2D and 3D networks


def get_conv(dim=3, adaptive=False):
    if dim == 3:
        return AdaptiveConv3d if adaptive else nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_convtranspose(dim=3, adaptive=False):
    if dim == 3:
        return AdaptiveConvTranspose3d if adaptive else nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_maxpool(dim=3):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_batchnorm(dim=3):
    if dim == 3:
        return nn.BatchNorm3d
    elif dim == 2:
        return nn.BatchNorm2d
    else:
        raise ValueError('dim has to be 2 or 3')


def planar_kernel(x):
    if isinstance(x, int):
        return (1, x, x)
    else:
        return x


def planar_pad(x):
    if isinstance(x, int):
        return (0, x, x)
    else:
        return x


def conv3(in_channels, out_channels, kernel_size=3, stride=1,
          padding=1, bias=True, planar=False, dim=3, adaptive=False):
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim, adaptive)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )


def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3, adaptive=False):
    kernel_size = 2
    stride = 2
    scale_factor = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
        scale_factor = planar_kernel(scale_factor)
    if mode == 'transpose':
        return get_convtranspose(dim, adaptive)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    else:
        # out_channels is always going to be the same
        # as in_channels
        mode = 'trilinear' if dim == 3 else 'bilinear'
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            conv1(in_channels, out_channels, dim=dim)
        )


def conv1(in_channels, out_channels, dim=3):
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=1,
    )


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'lin':
            return Identity()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)