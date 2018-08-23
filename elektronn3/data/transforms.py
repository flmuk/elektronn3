# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""
Transformations (data augmentation, normalization etc.) for semantic segmantation.

Important note: The transformations here have a similar interface to
torchvsion.transforms, but there are two key differences:

1. They all map (inp, target) pairs to (transformed_inp, transformed_target)
  pairs instead of just inp to inp. Most transforms don't change the target, though.
2. They exclusively operate on numpy.ndarray data instead of PIL or torch.Tensor data.
"""

from typing import Sequence, Tuple, Optional, Dict, Any, Union

import numpy as np

from elektronn3.data import random_blurring


# Transformation = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class Identity:
    def __call__(self, inp, target):
        return inp, target


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> Compose([
        >>>     Normalize(mean=(155.291411,), std=(41.812504,)),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize:
    """Normalizes inputs with supplied per-channel means and stds."""
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        normalized = np.empty_like(inp)
        if not inp.shape[0] == self.mean.shape[0] == self.std.shape[0]:
            raise ValueError('mean and std must have the same length as the C '
                             'axis (number of channels) of the input.')
        for c in range(inp.shape[0]):
            normalized[c] = (inp[c] - self.mean[c]) / self.std[c]
        return normalized, target


class AdditiveGaussianNoise:
    """Adds random gaussian noise to the input.

    Args:
        sigma: Sigma parameter of the gaussian distribution to draw from
        channels: If ``channels`` is ``None``, the noise is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, noise is only applied
            to the specified channels.
        rng: Optional random state for deterministic execution
    """
    def __init__(
            self,
            sigma: float = 0.1,
            channels: Optional[Sequence[int]] = None,
            rng: Optional[np.random.RandomState] = None
    ):
        self.sigma = sigma
        self.channels = channels
        self.rng = np.random.RandomState() if rng is None else rng

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise = np.zeros_like(inp)
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        for c in channels:
            noise[c] = self.rng.normal(0, self.sigma, inp[c].shape)
        noisy_inp = inp + noise
        return noisy_inp, target


class RandomBlurring:  # Warning: This operates in-place!

    _default_scheduler = random_blurring.ScalarScheduler(
        value=0.1,
        max_value=0.5,
        growth_type="lin",
        interval=500000,
        steps_per_report=1000
    )
    _default_config = {
        "probability": 0.5,
        "threshold": _default_scheduler,
        "lower_lim_region_size": [3, 6, 6],
        "upper_lim_region_size": [8, 16, 16],
        "verbose": False,
    }

    def __init__(
            self,
            config: Dict[str, Any],
            patch_shape: Optional[Sequence[int]] = None
    ):
        self.config = {**self._default_config, **config}
        # TODO: support random state
        if patch_shape is not None:
            random_blurring.check_random_data_blurring_config(patch_shape, **config)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        # In-place, overwrites inp!
        assert inp.ndim == 4, 'Currently only (C, D, H, W) inputs are supported.'
        random_blurring.apply_random_blurring(inp_sample=inp, **self.config)
        return inp, target


class RandomCrop:
    def __init__(self, size: Sequence[int]):
        # TODO: support random state
        self.size = np.array(size)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        ndim_spatial = len(self.size)  # Number of spatial axes E.g. 3 for (C,D,H.W)
        img_shape = inp.shape[-ndim_spatial:]
        # Number of nonspatial axes (like the C axis). Usually this is one
        ndim_nonspatial = inp.ndim - ndim_spatial
        # Calculate the "lower" corner coordinate of the slice
        coords_lo = np.array([
            np.random.randint(0, img_shape[i] - self.size[i] + 1)
            for i in range(ndim_spatial)
        ])
        coords_hi = coords_lo + self.size  # Upper ("high") corner coordinate.
        # Calculate necessary slice indices for reading the file
        nonspatial_slice = [  # Slicing all available content in these dims.
            slice(0, inp.shape[i]) for i in range(ndim_nonspatial)
        ]
        spatial_slice = [  # Slice only the content within the coordinate bounds
            slice(coords_lo[i], coords_hi[i]) for i in range(ndim_spatial)
        ]
        full_slice = nonspatial_slice + spatial_slice
        inp_cropped = inp[full_slice]
        if target is None:
            return inp_cropped, target

        if target.ndim == inp.ndim - 1:  # inp: (C, [D,], H, W), target: ([D,], H, W)
            full_slice = full_slice[1:]  # Remove C axis from slice because target doesn't have it
        target_cropped = target[full_slice]
        return inp_cropped, target_cropped


class SqueezeTarget:
    """Squeeze a specified dimension in target tensors.

    (This is just needed as a workaround for the example neuro_data_cdhw data
    set, because its targets have a superfluous first dimension.)"""
    def __init__(self, dim, inplace=True):
        self.dim = dim

    def __call__(
            self,
            inp: np.ndarray,  # Returned without modifications
            target: np.ndarray,
    ):
        return inp, target.squeeze(axis=self.dim)


# TODO: Handle target striding and offsets via transforms?

#Jittering points clouds (introducing Gaussian noise) and rotating point clouds.
#Adopted from https://github.com/hxdengBerkeley/PointCNN.Pytorch/blob/master/provider.py

class JitterScalePointCloud:
    """ Randomly jitter points with Gaussian noise bettwen [-clip, clip]). jittering is per point.
        scaling will be applied to the jittered points
               Input:
                 Nx3 array, original batch of point clouds
               Return:
                 Nx3 array, jittered batch of point clouds
           """
    def __init__(self, sigma: float = 0.1, clip: float = 0.05, scale: np.ndarray = None):
        self.sigma = sigma
        self.clip = clip
        self.scale = scale

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        N, C = inp.shape
        assert (self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        jittered_data += inp
        if self.scale is not None:
            jittered_data = jittered_data / self.scale
        return jittered_data, target


class RotatePointCloud:
    """ Randomly rotate the point clouds to augument the dataset
                rotation is per shape based along up direction
                Input:
                  Nx3 array, original batch of point clouds
                Return:
                  Nx3 array, rotated batch of point clouds
            """

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(inp, rotation_matrix)
        return rotated_data, target


