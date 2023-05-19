import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(filters_out)
        self.act = act
        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

        # Initialize weights and biases
        for module in [self.conv1, self.conv2, self.conv3]:
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        conv1_out = self.conv1(inputs)
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out)
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        conv3_out = self.conv3(inputs)
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out
"""
def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    conv1 = nn.Conv3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
    bn1 = nn.BatchNorm3d(filters_in)
    conv2 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
    bn2 = nn.BatchNorm3d(filters_out)
    conv3 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
    bn3 = nn.BatchNorm3d(filters_out)

    if act:
        activation = nn.LeakyReLU(0.2)
    else:
        activation = nn.Identity()

    for layer in [conv1, conv2, conv3]:
        if isinstance(layer, nn.Conv3d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm3d):
            nn.init.normal_(layer, mean=1.0, std=0.02)
            nn.init.zeros_(layer)
            #layer.running_mean.zero_()
            #layer.running_var.fill_(1.0)
            #layer.momentum = 1.0

    conv1_out = conv1(inputs)
    bn1_out = bn1(conv1_out)
    act1_out = activation(bn1_out)

    conv2_out = conv2(act1_out)
    bn2_out = bn2(conv2_out)
    act2_out = activation(bn2_out)

    conv3_out = conv3(inputs)
    bn3_out = bn3(conv3_out)
    act3_out = activation(bn3_out)

    conv_out = act2_out + act3_out

    return conv_out
"""


def resblock_up(inputs, filters_in, filters_out, act=True):
    conv1 = nn.ConvTranspose3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
    bn1 = nn.BatchNorm3d(filters_in)
    conv2 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
    bn2 = nn.BatchNorm3d(filters_out)
    conv3 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
    bn3 = nn.BatchNorm3d(filters_out)

    if act:
        activation = nn.LeakyReLU(0.2)
    else:
        activation = nn.Identity()

    for layer in [conv1, conv2, conv3]:
        if isinstance(layer, nn.ConvTranspose3d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm3d):
            nn.init.normal_(layer.weight, mean=1.0, std=0.02)
            nn.init.zeros_(layer.bias)
            #layer.running_mean.zero_()
            #layer.running_var.fill_(1.0)
            #layer.momentum = 1.0

    conv1_out = conv1(inputs)
    bn1_out = bn1(conv1_out)
    act1_out = activation(bn1_out)

    conv2_out = conv2(act1_out)
    bn2_out = bn2(conv2_out)
    act2_out = activation(bn2_out)

    conv3_out = conv3(inputs)
    bn3_out = bn3(conv3_out)
    act3_out = activation(bn3_out)

    conv_out = act2_out + act3_out

    return conv_out
class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()

        self.conv1 = nn.ConvTranspose3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(filters_out)

        if act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

        # Initialize weights and biases
        for module in [self.conv1, self.conv2, self.conv3]:
            if isinstance(module, nn.ConvTranspose3d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        conv1_out = self.conv1(inputs)
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out)
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        conv3_out = self.conv3(inputs)
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out
    

class VAE():
    pass