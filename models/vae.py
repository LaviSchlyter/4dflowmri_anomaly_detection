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

"""
def ResBlockUp(filters_in, filters_out, act=True):
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

"""
class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()

        self.conv1 = nn.ConvTranspose3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, output_padding=1)
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
    
class Encoder(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8) -> None:
        super(Encoder, self).__init__()
        self.gf_dim = gf_dim
        # Initialization
        w_init = torch.nn.init.normal_
        b_init = torch.nn.init.constant_
        gamma_init = torch.nn.init.normal_

        # 1st Conv block
        self.conv1 = nn.Conv3d(in_channels, gf_dim, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        # Res-Blocks
        self.res1 = ResBlockDown(gf_dim, gf_dim)
        self.res2 = ResBlockDown(gf_dim, gf_dim * 2)
        self.res3 = ResBlockDown(gf_dim * 2, gf_dim * 4)
        self.res4 = ResBlockDown(gf_dim * 4, gf_dim * 8)

        # Latent Convolution layers
        self.conv_latent = nn.Conv3d(gf_dim * 8, gf_dim * 32, kernel_size=(1, 1, 1), padding=0)
        self.conv_latent_std = nn.Conv3d(gf_dim * 8, gf_dim * 32, kernel_size=(1, 1, 1), padding=0)

        # Convolution block
        self.conv2 = nn.Conv3d(gf_dim, gf_dim, kernel_size=(3, 3, 3), dilation=2, padding=2)
        self.bn2 = nn.BatchNorm3d(gf_dim)
        self.relu2 = nn.LeakyReLU(0.2)

        # Convolution block
        self.conv3 = nn.Conv3d(gf_dim, gf_dim * 2, kernel_size=(3, 3, 3), dilation=2, padding=2)
        self.bn3 = nn.BatchNorm3d(gf_dim * 2)
        self.relu3 = nn.LeakyReLU(0.2)

        # Convolution block
        self.conv4 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=(3, 3, 3), dilation=2, padding=2)
        self.bn4 = nn.BatchNorm3d(gf_dim)
        self.relu4 = nn.LeakyReLU(0.2)

        # Convolution block
        self.conv5 = nn.Conv3d(gf_dim, 1, kernel_size=(3, 3, 3), dilation=2, padding=2)

         # Initialize the parameters TODO: Check if this is correct
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                w_init(m.weight, mean=0.0, std=0.01)
                b_init(m.bias, val=0.0)
            elif isinstance(m, nn.BatchNorm3d):
                gamma_init(m.weight, mean=0.5, std=0.01)    

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        res1 = self.res1(conv1)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)

        conv_latent = self.conv_latent(res4)
        conv_latent_std = self.conv_latent_std(res4)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.relu3(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.relu4(conv4)

        conv5 = self.conv5(conv4)
        # ====================================
        # Print shapes at various layers in the encoder network
        # ====================================
        """
        print('===================== ENCODER NETWORK ====================== ')
        print('Shape of input: ' + str(x.shape))
        print('Shape after 1st convolution block: ' + str(conv1.shape))
        print('Shape after 1st res block: ' + str(res1.shape))
        print('Shape after 2nd res block: ' + str(res2.shape))
        print('Shape after 3rd res block: ' + str(res3.shape))
        print('Shape after 4th res block: ' + str(res4.shape))
        print('-------------------------------------------------')
        print('Shape of latent_Mu: ' + str(conv_latent.shape))
        print('Shape of latent_stddev: ' + str(conv_latent_std.shape))
        print('-------------------------------------------------')
        print('=========================================================== ')
        """

        return conv_latent, conv_latent_std, conv5
"""
class Decoder(nn.Module):
    def __init__(self, gf_dim):
        super(Decoder, self).__init__()

        # Dimension of gen filters in first conv layer. [64]
        self.gf_dim = gf_dim

        # Initialization
        w_init = nn.init.normal_
        b_init = nn.init.constant_
        gamma_init = nn.init.ones_

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp(32, gf_dim * 16)
        self.res0 = ResBlockUp(16, gf_dim * 8)
        self.res1 = ResBlockUp(8, gf_dim * 4)
        self.res2 = ResBlockUp(4, gf_dim * 2)

        # 1st convolution block: convolution, followed by batch normalization and activation
        self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(gf_dim)

        # 2nd convolution block: convolution
        self.conv2 = nn.Conv3d(gf_dim, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Initialization
        w_init(self.conv1.weight)
        b_init(self.conv1.bias)
        gamma_init(self.bn1.weight)

    def forward(self, z):
        resp1 = self.resp1(z)
        res0 = self.res0(resp1)
        res1 = self.res1(res0)
        res2 = self.res2(res1)

        conv1 = self.conv1(res2)
        conv1 = self.bn1(conv1)
        conv1 = nn.functional.leaky_relu(conv1, negative_slope=0.2)

        conv2 = self.conv2(conv1)

        return conv2
"""

class Decoder(nn.Module):
    def __init__(self, gf_dim=8):
        super(Decoder, self).__init__()
        
        self.gf_dim = gf_dim

        # Initialization
        w_init = nn.init.trunc_normal_
        b_init = nn.init.constant_
        gamma_init = nn.init.ones_

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp(gf_dim * 32, gf_dim * 16)
        self.res0 = ResBlockUp(gf_dim * 16, gf_dim * 8)
        self.res1 = ResBlockUp(gf_dim * 8, gf_dim * 4)
        self.res2 = ResBlockUp(gf_dim * 4, gf_dim * 2)

        # 1st convolution block: convolution, followed by batch normalization and activation
        self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.conv1.apply(self._initialize_weights)

        # 2nd convolution block: convolution
        self.conv2 = nn.Conv3d(gf_dim, 4, kernel_size=3, stride=1, padding=1)
        self.conv2.apply(self._initialize_weights)



    def forward(self, x):
        #print(' Input to decoder has the following shape:' + str(x.shape))
        # Res-Blocks (for effective deep architecture)
        resp1 = self.resp1(x)
        res0 = self.res0(resp1)
        res1 = self.res1(res0)
        res2 = self.res2(res1)

        # 1st convolution block: convolution, followed by batch normalization and activation
        conv1 = self.conv1(res2)
        conv1 = self.bn1(conv1)
        conv1 = F.leaky_relu(conv1, 0.2)

        # 2nd convolution block: convolution
        conv2 = self.conv2(conv1)
        """
        print('===================== DECODER NETWORK ====================== ')
            
        print('Shape of input: ' + str(x.shape))
                    
        #print('Shape after 1st convolution block: ' + str(resp1.shape))
                    
        print('Shape after 1st res block: ' + str(resp1.shape))
                    
        print('Shape after 2nd res block: ' + str(res0.shape))
                    
        print('Shape after 3rd res block: ' + str(res1.shape))
                    
        print('Shape after 4th res block: ' + str(res2.shape))
                    
        #print('Shape after 5th res block: ' + str(res4.shape))
                    
        print('-------------------------------------------------')
                    
        print('Shape after 1st convolution ' + str(conv1.shape))
                    
        print('Shape of output of decoder' + str(conv2.shape))
                    
        print('-------------------------------------------------')
                    
        print('=========================================================== ')
        """

        

        return conv2

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class VAE(nn.Module):
    def __init__(self, z_dim:int, in_channels:int =4, gf_dim:int =8):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels, gf_dim)
        self.decoder = Decoder()

        self.z_dim = z_dim

    def forward(self, x):
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)
        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decode
        decoder_output = self.decoder(self.guessed_z)

        return decoder_output, z_mean, z_std, res
