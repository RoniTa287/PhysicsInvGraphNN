import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import ShallowUNet, UNet


class ConvEncoder(nn.Module):

    def __init__(self, n_objs, conv_ch, input_shape, D=2):
        """
        :param input_shape: Shape of the input images (H, W).
        :param n_objs: Number of objects to detect.
        :param conv_ch: Number of input channels.
        :param D: Dimensionality of the coordinates for each object.
        """
        super(ConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.n_objs = n_objs
        self.conv_ch = conv_ch
        self.D = D  # Dimensionality of coordinates
        self.saved_masks = None

        # Defining the U-Net architecture
        # self.encoder_net = ShallowUNet(input_channels=conv_ch, base_channels=8 if input_shape[0] < 40 else 16,
        #                                out_channels=n_objs * 2, upsamp=True)
        if input_shape[0] < 40:
            self.encoder_net = ShallowUNet(input_channels=conv_ch, base_channels=8, out_channels=n_objs * 2,
                                           upsamp=True)
        else:
            self.encoder_net = UNet(input_channels=conv_ch, base_channels=16, out_channels=n_objs * 2,
                                    upsamp=True)

        # Assuming the U-Net reduces the spatial dimensions by a factor
        # This factor depends on the U-Net architecture and input dimensions
        # Let's say the final spatial dimensions are reduced by a factor of 4x4
        # and you flatten this output to a single vector.
        reduced_size_factor = (input_shape[0] // 4) * (input_shape[1] // 4)
        flattened_size = reduced_size_factor * n_objs * 2 * 4  # Adjust based on your U-Net's output

        # Dense layers to map U-Net output to object coordinates
        self.dense_layers = nn.Sequential(
            nn.Linear(flattened_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, n_objs * D)  # Mapping to D-dimensional coordinates
        )

    def forward(self, x):
        # Processing through U-Net
        x = self.encoder_net(x)
        # Save masks for later use
        normalized_masks = F.softmax(x, dim=1)
        self.saved_masks = normalized_masks

        # Flatten the U-Net output
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # Passing the flattened output through dense layers to get coordinates
        x = self.dense_layers(x)

        # Reshaping output to have it in the shape [batch_size, n_objs, D]
        x = x.view(batch_size, self.n_objs, self.D)

        return x

    def create_coord_grid(self, size):
        # Assuming size is a tuple (H, W)
        H, W = size
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape: [2, H, W]
        return grid

    def get_saved_masks(self):
        return self.saved_masks


class VelEncoder(nn.Module):
    def __init__(self, n_objs, input_steps, coord_units, device='cpu'):
        """
        Initializes the velocity estimator module.

        :param n_objs: Number of objects to estimate velocities for.
        :param input_steps: Number of input steps to consider for velocity estimation.
        :param coord_units: Total number of coordinate units (e.g., for 2D positions, this is 2 * n_objs).
        :param device: The device to use for computations.
        """
        super(VelEncoder, self).__init__()
        self.n_objs = n_objs
        self.input_steps = input_steps
        self.coord_units = coord_units
        self.device = device

        # Define MLP layers
        self.fc1 = nn.Linear(input_steps * coord_units // n_objs, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, coord_units // n_objs)

    def forward(self, x):
        """
        Forward pass through the velocity estimator.

        :param x: The input tensor containing encoded positions for all objects across all input steps.
                  Expected shape: [batch_size, n_objs, input_steps, coord_units // n_objs]
        """
        batch_size = x.shape[0]

        # Reshape input to combine batch and object dimensions
        x = x.view(batch_size * self.n_objs, self.input_steps * self.coord_units // self.n_objs).to(self.device)

        # Forward pass through MLP
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        # Reshape output to separate batch and object dimensions
        x = x.view(batch_size, self.n_objs, -1)

        return x