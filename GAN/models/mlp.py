from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class MLP_G(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Generator.

    Args:
        isize (int): Size of the generated image (height/width).
        nz (int): Dimension of the latent noise vector.
        nc (int): Number of output channels (e.g., 3 for RGB images).
        ngf (int): Number of generator feature maps.
        ngpu (int): Number of GPUs to use for parallel computation.
    """
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        """
        Forward pass of the generator.

        Args:
            input (Tensor): Latent noise vector of shape (batch_size, nz).

        Returns:
            Tensor: Generated image of shape (batch_size, nc, isize, isize).
        """
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Discriminator.

    Args:
        isize (int): Size of the input image.
        nz (int): Dimension of the latent noise vector (not used in the discriminator).
        nc (int): Number of input channels (e.g., 3 for RGB images).
        ndf (int): Number of discriminator feature maps.
        ngpu (int): Number of GPUs to use for parallel computation.
    """
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        """
        Forward pass of the discriminator.

        Args:
            input (Tensor): Image tensor of shape (batch_size, nc, isize, isize).

        Returns:
            Tensor: Discriminator output score (real or fake).
        """
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)
