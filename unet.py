import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowUNet(nn.Module):
    def __init__(self, input_channels, base_channels, out_channels, upsamp=True):
        """
        Initialize the shallow U-Net.

        Parameters:
        - input_channels: Number of input channels (e.g., 3 for RGB images).
        - base_channels: Number of channels in the first and last layers of the U-Net.
        - out_channels: Number of output channels, which can include object masks plus any additional per-object features.
        - upsamp: Boolean indicating whether to use bilinear upsampling (True) or transposed convolutions (False).
        """
        super(ShallowUNet, self).__init__()
        self.upsamp = upsamp

        # Initial convolution layers
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Downsample
        self.conv3 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        # Bottleneck
        self.conv5 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        # Upsample and Concatenate
        self.upconv1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)  # Adjusted for concatenated channels

        # Final output layer adjusted for out_channels
        self.final_conv = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        p1 = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = F.relu(self.conv3(p1))
        c4 = F.relu(self.conv4(c3))
        p2 = F.max_pool2d(c4, kernel_size=2, stride=2)

        # Bottleneck
        c5 = F.relu(self.conv5(p2))
        c6 = F.relu(self.conv6(c5))

        # Decoding path
        if self.upsamp:
            up1 = F.interpolate(c6, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            up1 = F.conv_transpose2d(c6, self.base_channels * 2, kernel_size=3, stride=2, padding=1)
        up1 = F.relu(self.upconv1(up1))

        # Ensure spatial dimensions match for concatenation
        if up1.size()[2:] != c4.size()[2:]:
            up1 = F.interpolate(up1, size=c4.size()[2:], mode='bilinear', align_corners=True)

        concat1 = torch.cat((up1, c4), dim=1)
        c7 = F.relu(self.conv7(concat1))

        # Final output layer for masks
        masks = self.final_conv(c7)

        # Softmax applied across the channels to generate probabilistic masks
        masks = F.softmax(masks, dim=1)

        # Handling masked objects (omitted for brevity) and coordinate extraction would follow here
        return masks


    # def __init__(self, input_channels, base_channels, out_channels, upsamp=True):
    #     super(ShallowUNet, self).__init__()
    #     self.upsamp = upsamp
    #
    #     # Initial convolution layers
    #     self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
    #
    #     # Downsample
    #     self.conv3 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
    #
    #     # Upsample
    #     self.upconv1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
    #     self.conv5 = nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3,
    #                            padding=1)  # Adjusted for concatenated channels
    #
    #     # Final output layer
    #     self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    #
    # def forward(self, x):
    #     print(f"Input x shape: {x.shape}")
    #     # Encoding path
    #     c1 = F.relu(self.conv1(x))
    #     print(f"c1 shape: {c1.shape}")
    #     c2 = F.relu(self.conv2(c1))
    #     print(f"c2 shape: {c2.shape}")
    #     p1 = F.max_pool2d(c2, kernel_size=2, stride=2)
    #     print(f"p1 (after pooling c2) shape: {p1.shape}")
    #
    #     c3 = F.relu(self.conv3(p1))
    #     print(f"c3 shape: {c3.shape}")
    #     c4 = F.relu(self.conv4(c3))
    #     print(f"c4 shape: {c4.shape}")
    #     p2 = F.max_pool2d(c4, kernel_size=2, stride=2)
    #     print(f"p2 (after pooling c4) shape: {p2.shape}")
    #
    #     # Decoding path
    #     if self.upsamp:
    #         # Using bilinear upsampling
    #         up1 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=True)
    #         print(f"up1 (upsampled p2) shape: {up1.shape}")
    #     # else:
    #     #     # Alternatively, use transpose convolutions for upsampling
    #     #     up1 = F.conv_transpose2d(p2, base_channels * 2, kernel_size=3, stride=2, padding=1)
    #
    #     up1 = self.upconv1(up1)
    #     print(f"up1 (after upconv1) shape: {up1.shape}")
    #
    #     # Ensure up1 and c2 have the same spatial dimensions for concatenation
    #     # If they don't match, adjust the size of up1 using interpolate
    #     if up1.size()[2:] != c2.size()[2:]:
    #         up1 = F.interpolate(up1, size=c2.size()[2:], mode='bilinear', align_corners=True)
    #         print(f"up1 resized for concatenation with c2: {up1.shape}")
    #
    #     # Concatenate upsampled output with the corresponding feature map from the encoding path
    #     concat1 = torch.cat((up1, c2), dim=1)
    #     print(f"concat1 shape: {concat1.shape}")
    #
    #     # Final convolutions after concatenation
    #     c5 = F.relu(self.conv5(concat1))
    #     print(f"c5 shape: {c5.shape}")
    #
    #     # Output layer
    #     out = self.final_conv(c5)
    #     print(f"Output shape: {out.shape}")
    #
    #     return out


class VariableFromNetwork(nn.Module):
    def __init__(self, shape):
        super(VariableFromNetwork, self).__init__()
        self.shape = shape
        self.fc1 = nn.Linear(10, 200)


class UNet(nn.Module):
    def __init__(self, input_channels, base_channels, out_channels, upsamp=True):
        super(UNet, self).__init__()
        self.upsamp = upsamp

        # Initial convolution layers
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Downsample
        self.conv3 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        # Upsample
        self.upconv1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3,
                               padding=1)  # Adjusted for concatenated channels

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        pass
