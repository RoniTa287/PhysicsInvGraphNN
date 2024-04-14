import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# plt.ion()   # interactive mode

############################################################
# LOOK BELOW
############################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(6, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # changed 320 -> to 500
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.h = 6 #input channels image
        # Spatial transformer localization-network
        # By stacking these convolutional layers with pooling and ReLU activations,
        # the localization network learns to extract spatial features from the input
        # image necessary for predicting the transformation parameters in
        # the subsequent regressor part of the network.

        # self.localization = nn.Sequential(
        #     nn.Conv2d(6, 8, kernel_size=3), # changed the n_channels to 14 to match dimensions of joint
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=3), #tweaked
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )  # in brief it extracts features from the input image and locates spatial info, it is CNN
        self.localization = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=3),  # Adjusted input channels to match input tensor
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),  # Adjusted output channels based on the previous layer
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        # nn.sequential allows to stack multiple layers sequentially and it consists of fully onnected linea r layers
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),  # if initial input is 32x32, adjust to 3->4
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()  # access the weight parameter of the linear layer
        # and then sets them to zero .zero()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))  # copies thr bias to that tensor

        # Spatial transformer network forward function

    def stn(self, x):  # starts (3,10,3,3)
        x = x.transpose(1, 3)  # switch 3 and 6
        print(x.shape)
        xs = self.localization(x)  # it return a 3,10,3,3
        xs = xs.view(-1, 10 * 7 * 7)  # it return a 2, 160)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # has dimension (batch size, 2,3)

        ###################### Vale added: align_corners=True ######################
        grid = F.affine_grid(theta, x.size(), align_corners=True)  # creates a grid based on theta theta and x
        x = F.grid_sample(x, grid, align_corners=True)  # applied transformation
        # To get the desired output, the input feature map should be sampled from the
        # parameterized sampling grid. The grid generator outputs the parameterized sampling grid.
        # Then we apply the transformation theta to the grid (grid_sample)

        ###################### Vale added this since we were transposing ######################
        x = x.transpose(1, 3)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        print("size x:",x.shape)
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print("Size of x after relu", x.size())
        # given this we need to adjust the next line 320 -> 500
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x =  F.log_softmax(x, dim=1) # transpose(0,1)?
        print("x after softmax", x.shape)
        # print('Hello this runsssss!!!!')
        return x


class VariableFromNetwork(nn.Module):
    def __init__(self, shape):
        super(VariableFromNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 200)
        self.fc2 = nn.Linear(200, torch.prod(torch.tensor(shape)))
        self.shape = shape

    def forward(self):
        x = torch.ones(1, 10)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        x = x.view(self.shape)
        return x


class ConvSTDecoder(nn.Module):  # input dimension 32x6 ; where 80 is batch and 6 = NxD = 3x2 objects x dimension 2D
    def __init__(self, input_size, n_objs, conv_ch):
        super(ConvSTDecoder, self).__init__()

        self.n_objs = n_objs
        self.conv_ch = conv_ch

        self.input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch]
        self.conv_input_shape = [int(np.sqrt(input_size))] * 2 + [self.conv_ch]

        tmpl_size = self.conv_input_shape[0]    # 2
        # make it nn.Parameter because it is a trainable parameter
        self.logsigma = nn.Parameter(torch.tensor([np.log(2.0)]))    # log 2.0 better
        self.sigma = torch.exp(self.logsigma)

        # Create variables from network
        self.template = VariableFromNetwork([n_objs, tmpl_size, tmpl_size, 1])
        # self.template = template
        self.contents = VariableFromNetwork([n_objs, tmpl_size, tmpl_size, conv_ch])
        # self.contents = contents
        self.background_content = VariableFromNetwork([1, self.input_shape[0],
                                                       self.input_shape[1], self.input_shape[2]])
        # changed 1 to 3 for 3 objects
        self.net = Net()

    def forward(self, inp):
        batch_size = inp.size(0)   # 32
        # [n_objs, tmpl_size, tmpl_size, 1]
        template = self.template.forward()   # torch.Size([3, 36, 36, 1])
        template = template.repeat(1, 1, 1, 3) + 5
        # ToDO: check if we need softmax before stn
        template = F.softmax(template, dim=1)   # torch.Size([3, 36, 36, 3])
        contents = torch.sigmoid(self.contents.forward())
        joint = torch.cat([template, contents], dim=-1)  # concatenate along depth dimension
        out_temp_cont = []
        out_joint = self.net.stn(joint)   # ([3, 36, 36, 6])
        out_temp_cont.append(torch.split(out_joint, 3, -1))  # [template, content]] !!!!

        # Background
        background_content = torch.nn.Sigmoid()(self.background_content())
        background_content = torch.tile(background_content, [1, 1, 1])
        print("background contents shape:", background_content.shape)
        #  batch_size --> 3? added 3 as a dimension

        contents = [p[1] for p in out_temp_cont]
        contents.append(background_content)
        print(f"Shape of background content: {background_content.shape}")
        # contents has second tensor in out_temp_cont and background content

        background_mask = torch.ones_like(out_temp_cont[0][0])  # [3, 36, 36, 3]
        print("background mask shape:", background_mask.shape)
        # stack to apply softmax & unstuck after
        masks = torch.stack([p[0] - 5 for p in out_temp_cont] + [background_mask], dim=-1)
        print(f"masks0 _ : {masks.shape}")
        masks = torch.nn.Softmax(dim=-1)(masks)
        masks = torch.unbind(masks, dim=-1)
        # masks_tensor = torch.stack(masks, dim=-1)  # Convert tuple to tensor
        for i, (mask, content) in enumerate(zip(masks, contents)):
            print(f"Shape of mask {i}: {mask.shape}, Shape of content {i}: {content.shape}")

        # out = torch.stack([m * c for m, c in zip(masks, contents)])
        # print(f"out decoder1: {out.shape}")
        # out = torch.sum(out, dim=0)
        # print(f"out decoder2: {out.shape}")

        multiplied = [mask * content for mask, content in zip(masks, contents)]
        # Assuming multiplied is now a list of 3D tensors resulting from element-wise multiplication.
        # You would then stack these tensors to create a 4D tensor and sum across the first dimension.
        stacked = torch.stack(multiplied, dim=0)
        print(f"out stacked: {stacked.shape}")
        out = torch.sum(stacked, dim=0)
        print(f"out decoder: {out.shape}")

        return out

    # print(contents[2].shape)
    # print(contents[3].shape)
    # masks = [torch.tensor(m) for m in masks]
    # contents = [torch.tensor(c) for c in contents]

    # contents_tensor = torch.stack(contents, dim=-1)  # Convert tuple to tensor
    # print("contents:", contents_tensor.size())  # Print the size of the tensor

    # # Stack the tensors along a new dimension (dim=0) to form a single tensor
    # stacked_tensors = torch.stack([m * c for m, c in zip(masks, contents)])

    # Sum along the new dimension to get the final output
    # out = torch.sum(stacked_tensors, dim=0)