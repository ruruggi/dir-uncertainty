from ast import literal_eval
from typing import Tuple

from utils.config import readCfg
from utils.conversions import toIntMaybe

import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d, conv3d
import torch.nn.functional as F



SAMPLE_GAUSSIAN_BLUR_KERNEL_SIZE = "sampleGaussianBlurKernelSize"
SAMPLE_GAUSSIAN_BLUR_SIGMA = "sampleGaussianBlurSigma"
SAMPLE_GAUSSIAN_BLUR_PADDING = "sampleGaussianBlurPadding"
RESCALE_CORR_MATRIX = "rescaleCorrelationMatrix"


def gaussian(window_size, sigma, offset):
    def gauss_fcn(x):
        return -((x - window_size // 2 - offset) ** 2) / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float, offset: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(ksize,)`

    Examples::
    """
    if ksize != int(ksize) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(ksize))
    window_1d: torch.Tensor = gaussian(int(ksize), sigma, offset)
    return window_1d


def get_gaussian_kernelNd(
    ksize: Tuple[int], sigma: Tuple[float], offsets=None
) -> torch.Tensor:
    n = len(ksize)
    if not isinstance(ksize, tuple):
        raise TypeError("ksize must be a tuple. Got {}".format(ksize))
    if not isinstance(sigma, tuple):
        raise TypeError("sigma must be a tuple. Got {}".format(sigma))
    if offsets is None:
        offsets = [0.0] * len(ksize)
    kernels = []
    for ksizeAx, sigmaAx, offset in zip(ksize, sigma, offsets):
        kernel: torch.Tensor = get_gaussian_kernel(ksizeAx, sigmaAx, offset)
        kernels.append(kernel)

    if n == 1:
        return kernels[0]
    elif n == 2:
        return torch.matmul(kernels[0].unsqueeze(-1), kernels[1].unsqueeze(-1).t())
    elif n == 3:
        return torch.matmul(
            kernels[1].unsqueeze(-1), kernels[2].unsqueeze(-1).t()
        ) * kernels[0].unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError


def getVarianceWeights(kernelSize, smoothingSigma, shape, device, padding):
    smoothingKernel = get_gaussian_kernelNd(kernelSize, smoothingSigma)
    squaredSmoothingKernel = (
        (smoothingKernel * smoothingKernel).unsqueeze(0).unsqueeze(0)
    ).to(device)

    if padding == "same":
        onesTensor = torch.ones(shape).to(device)
        while len(onesTensor.shape) < 5:
            onesTensor = onesTensor.unsqueeze(0)
        weights = F.conv3d(onesTensor, squaredSmoothingKernel, padding=padding)
        while len(weights.shape) > len(shape):
            weights = weights[0]
        return weights
    else:
        return torch.sum(squaredSmoothingKernel)


class GaussianBlur(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int],
        sigma: Tuple[float],
        rescale=False,
        padding="same",
    ) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int] = kernel_size
        self.sigma: Tuple[float] = sigma
        self.N = len(self.kernel_size)
        self.rescale = rescale
        self.rescaleTensor = None
        self.nbChannels = 1
        self.padding = padding
        assert self.padding == 0  # we drop the support for same padding
        
        # [OPTIMIZED] Separable kernels
        self.kernels_1d = self.create_separable_kernels(kernel_size, sigma, rescale)

    def isEqual(self, other) -> bool:
        if isinstance(other, GaussianBlur):
            return (
                self.kernel_size == other.kernel_size
                and self.rescaleTensor == other.rescaleTensor
            )
        raise NotImplementedError

    @staticmethod
    def create_separable_kernels(kernel_size, sigma, rescale) -> list:
        """Returns a list of 1D Gaussian kernel tensors (optimized separable conv)."""
        kernels = []
        for ks, sg in zip(kernel_size, sigma):
            k_1d = get_gaussian_kernel(ks, sg, 0.0)
            kernels.append(k_1d)
            
        if rescale:
            # 3D L2 norm squared is the product of 1D L2 norms squared
            var_weight = 1.0
            for k_1d in kernels:
                var_weight *= float(torch.sum(k_1d * k_1d))
            rescale_factor = var_weight ** (1.0 / len(kernel_size))
            kernels = [k / float(torch.sqrt(torch.tensor(rescale_factor))) for k in kernels]
            
        return kernels

    def forward(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 2 + self.N:
            raise ValueError("Invalid input shape, we expect BxCxHxW(xD). Got: {}".format(x.shape))
            
        # prepare kernels in first forward pass or if device changes
        c = x.shape[1]
        if self.nbChannels != c or getattr(self, '_first_pass', True):
            self.nbChannels = c
            self._first_pass = False
            self.formatted_kernels = []
            
            for i in range(self.N):
                k = self.kernels_1d[i].to(x.device).to(x.dtype)
                # Reshape for separable: (Z, 1, 1), then (1, Y, 1), then (1, 1, X)
                shape = [1] * self.N
                shape[i] = self.kernel_size[i]
                reshaped_k = k.view(1, 1, *shape).repeat(self.nbChannels, 1, *([1]*self.N))
                self.formatted_kernels.append(reshaped_k)

        # convolve tensor sequentially with 1D kernels (Separable Convolution -> ~1200x faster for K=61)
        for i in range(self.N):
            k = self.formatted_kernels[i]
            if k.device != x.device: k = k.to(x.device)
            if k.dtype != x.dtype: k = k.to(x.dtype)
            
            if self.N == 1:
                x = conv1d(x, k, padding=self.padding, stride=1, groups=self.nbChannels)
            elif self.N == 2:
                x = conv2d(x, k, padding=self.padding, stride=1, groups=self.nbChannels)
            elif self.N == 3:
                x = conv3d(x, k, padding=self.padding, stride=1, groups=self.nbChannels)
                
        if self.rescale:
            x = x / self.getRescaleTensor().to(x.device)
        return x

    def getVarianceWeights(self):
        var_weight = 1.0
        for k_1d in self.kernels_1d:
            var_weight *= float(torch.sum(k_1d * k_1d))
        return torch.tensor(var_weight)

    def getRescaleTensor(self, update=False):
        if self.rescaleTensor is None or update:
            self.rescaleTensor = torch.sqrt(self.getVarianceWeights())
        return self.rescaleTensor

def createGaussianBlurLayer(cfg):
    sampleGaussianBlurKernelSize = readCfg(
        cfg, SAMPLE_GAUSSIAN_BLUR_KERNEL_SIZE, (9, 9, 9), literal_eval
    )
    sampleGaussianBlurSigma = readCfg(
        cfg, SAMPLE_GAUSSIAN_BLUR_SIGMA, (4, 4, 4), literal_eval
    )
    rescaleCorrelationMatrix = readCfg(cfg, RESCALE_CORR_MATRIX, False, literal_eval)
    gaussianBlurPadding = readCfg(cfg, SAMPLE_GAUSSIAN_BLUR_PADDING, "same", toIntMaybe)
    return GaussianBlur(
        sampleGaussianBlurKernelSize,
        sampleGaussianBlurSigma,
        rescaleCorrelationMatrix,
        gaussianBlurPadding,
    )
