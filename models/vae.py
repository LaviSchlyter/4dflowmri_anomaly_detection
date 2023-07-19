import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, stride = (2,2,2), apply_initialization = False):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=stride, padding=1)
        self.bn3 = nn.BatchNorm3d(filters_out)
        self.act = act
        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

        if apply_initialization:
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
    
class ResBlockDown_linear(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, stride = (2,2,2), apply_initialization = False):
        super(ResBlockDown_linear, self).__init__()
        self.conv1 = nn.Conv3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=stride, padding=1)
        self.bn3 = nn.BatchNorm3d(filters_out)
        self.act = act
        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()
        if apply_initialization:
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

class ResBlockUp_interpolate(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, scale_factor =(2,2,2)):
        super(ResBlockUp_interpolate, self).__init__()

        self.conv1 = nn.Conv3d(filters_in, filters_in, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.Conv3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(filters_out)

        if act:
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
        self.scale_factor = scale_factor

    def forward(self, inputs):
        # 256,2,2,3
        # Interpolate
        inputs_ = F.interpolate(inputs, scale_factor=self.scale_factor, mode='nearest')
        conv1_out = self.conv1(inputs_)
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out)
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        inputs_ = F.interpolate(inputs, scale_factor=self.scale_factor, mode='nearest')
        conv3_out = self.conv3(inputs_)
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out

class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, stride =(2,2,2), output_padding = (1,1,1), apply_initialization = False):
        super(ResBlockUp, self).__init__()

        self.conv1 = nn.ConvTranspose3d(filters_in, filters_in, kernel_size=(3, 3, 3), stride=stride, padding=1, output_padding=output_padding)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = nn.ConvTranspose3d(filters_in, filters_out, kernel_size=(3, 3, 3), stride=stride, padding=1, output_padding=output_padding)
        self.bn3 = nn.BatchNorm3d(filters_out)

        if act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

        if apply_initialization:
            # Initialize weights and biases
            for module in [self.conv1, self.conv2, self.conv3]:
                if isinstance(module, nn.ConvTranspose3d):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm3d):
                    nn.init.normal_(module.weight, mean=1.0, std=0.02)
                    nn.init.zeros_(module.bias)

    def forward(self, inputs):
        # inputs: [b, 256, 2,2,3]
        conv1_out = self.conv1(inputs)
        # conv1_out: [b, 256, 4,4,3]
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out)
        # conv1_out: [b, 256, 4,4,3]
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        conv3_out = self.conv3(inputs)
        # 256, 4,4,3
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out
    

class Encoder(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8, z_dim:int = 3072, apply_initialization = False) -> None:
        super(Encoder, self).__init__()
        self.gf_dim = gf_dim
        

        # 1st Conv block
        self.conv1 = nn.Conv3d(in_channels, gf_dim, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        # Res-Blocks
        self.res1 = ResBlockDown(gf_dim, gf_dim, stride=(2,2,2))
        self.res2 = ResBlockDown(gf_dim, gf_dim * 2, stride=(2,2,2))
        self.res3 = ResBlockDown(gf_dim * 2, gf_dim * 4, stride=(2,2,2))
        self.res4 = ResBlockDown(gf_dim * 4, gf_dim * 8, stride=(2,2,1))

        # Latent Convolution layers
        self.conv_latent = nn.Conv3d(gf_dim * 8, gf_dim * 32, kernel_size=(1, 1, 1), padding=0)
        self.conv_latent_std = nn.Conv3d(gf_dim * 8, gf_dim * 32, kernel_size=(1, 1, 1), padding=0)

        # Last Linear layer
        # The image size (x,y,t) is 2x2x3 and channels depends on gf_dim 
        self.linear_latent = nn.Linear(gf_dim * 32 * 2*2*3, z_dim)

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

        if apply_initialization: 
            # Initialization
            w_init = torch.nn.init.normal_
            b_init = torch.nn.init.constant_
            gamma_init = torch.nn.init.normal_
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

        # TODO: Add linear layer
        #linear_latent = self.linear_latent(res4.view(res4.size(0), -1))
        #linear_latent_std = self.linear_latent(res4.view(res4.size(0), -1))

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
    
class Encoder_linear(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8, z_dim:int = 3072, apply_initialization = False) -> None:
        super(Encoder_linear, self).__init__()
        self.gf_dim = gf_dim
        

        # 1st Conv block
        self.conv1 = nn.Conv3d(in_channels, gf_dim, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        # Res-Blocks
        self.res1 = ResBlockDown_linear(gf_dim, gf_dim, stride=(2,2,2))
        self.res2 = ResBlockDown_linear(gf_dim, gf_dim * 2, stride=(2,2,2))
        self.res3 = ResBlockDown_linear(gf_dim * 2, gf_dim * 4, stride=(2,2,2))
        self.res4 = ResBlockDown_linear(gf_dim * 4, gf_dim * 8, stride=(2,2,1))
        
        # Last Linear layer
        # The image size (x,y,t) is 2x2x3 and channels depends on gf_dim 
        self.linear_latent = nn.Linear(gf_dim * 8 * 2*2*3, z_dim)
        
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

        if apply_initialization: 
            # Initialization
            w_init = torch.nn.init.normal_
            b_init = torch.nn.init.constant_
            gamma_init = torch.nn.init.normal_
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

        #conv_latent = self.conv_latent(res4)
        #conv_latent_std = self.conv_latent_std(res4)

        # TODO: Add linear layer
        res4 = res4.reshape(res4.size(0), -1)
        linear_latent = self.linear_latent(res4)
        linear_latent_std = self.linear_latent(res4)

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
        return linear_latent, linear_latent_std, conv5
class Decoder_interpolate(nn.Module):
    def __init__(self, gf_dim = 8, out_channels = 4, apply_initialization = False):
        super(Decoder_interpolate, self).__init__()
        
        self.gf_dim = gf_dim
        print(gf_dim)

        # Initialization
        #w_init = nn.init.trunc_normal_
        #b_init = nn.init.constant_
        #gamma_init = nn.init.ones_

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp_interpolate(gf_dim * 32, gf_dim * 16, scale_factor=(2,2,1))
        self.res0 = ResBlockUp_interpolate(gf_dim * 16, gf_dim * 8, scale_factor=(2,2,2))
        self.res1 = ResBlockUp_interpolate(gf_dim * 8, gf_dim * 4, scale_factor=(2,2,2))
        self.res2 = ResBlockUp_interpolate(gf_dim * 4, gf_dim * 2, scale_factor=(2,2,2))
        #self.resp1 = ResBlockUp_interpolate(gf_dim * 8, gf_dim * 4, scale_factor=(2,2,1))
        #self.res0 = ResBlockUp_interpolate(gf_dim * 4, gf_dim * 2, scale_factor=(2,2,2))
        #self.res1 = ResBlockUp_interpolate(gf_dim * 2, gf_dim, scale_factor=(2,2,2))
        #self.res2 = ResBlockUp_interpolate(gf_dim, gf_dim, scale_factor=(2,2,2))

        # 1st convolution block: convolution, followed by batch normalization and activation
        self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        if apply_initialization:
            self.conv1.apply(self._initialize_weights)

        # 2nd convolution block: convolution
        self.conv2 = nn.Conv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        if apply_initialization:
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

class Decoder_linear(nn.Module):
    def __init__(self, gf_dim = 8, out_channels = 4, apply_initialization = False, z_dim:int = 3072, interpolate = False):
        super(Decoder_linear, self).__init__()
        
        self.gf_dim = gf_dim
        print(gf_dim)

        # Linear to enable reshape
        self.linear_latent = nn.Linear(z_dim, gf_dim * 8 * 2*2*3)

        if interpolate:
            # Res-Blocks (for effective deep architecture)
            self.resp1 = ResBlockUp_interpolate(gf_dim * 8 , gf_dim * 16, scale_factor=(2,2,1))
            self.res0 = ResBlockUp_interpolate(gf_dim * 16, gf_dim * 8, scale_factor=(2,2,2))
            self.res1 = ResBlockUp_interpolate(gf_dim * 8, gf_dim * 4, scale_factor=(2,2,2))
            self.res2 = ResBlockUp_interpolate(gf_dim * 4, gf_dim * 2, scale_factor=(2,2,2))
        else:
            # Res-Blocks (for effective deep architecture)
            self.resp1 = ResBlockUp(gf_dim * 8 , gf_dim * 16, stride=(2,2,1), output_padding = (1,1,0))
            self.res0 = ResBlockUp(gf_dim * 16, gf_dim * 8, stride=(2,2,2), output_padding = (1,1,1))
            self.res1 = ResBlockUp(gf_dim * 8, gf_dim * 4, stride=(2,2,2), output_padding = (1,1,1))
            self.res2 = ResBlockUp(gf_dim * 4, gf_dim * 2, stride=(2,2,2), output_padding = (1,1,1))
 
        # 1st convolution block: convolution, followed by batch normalization and activation
        self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        if apply_initialization:
            self.conv1.apply(self._initialize_weights)

        # 2nd convolution block: convolution
        self.conv2 = nn.Conv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        if apply_initialization:
            self.conv2.apply(self._initialize_weights)



    def forward(self, x):
        #print(' Input to decoder has the following shape:' + str(x.shape))
        # Give to linear layer to match the shape
        x = self.linear_latent(x)
        # Reshape the input following the linear layer
        x = x.reshape((x.size(0), -1, 2,2,3))
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

class Decoder(nn.Module):
    def __init__(self, gf_dim = 8, out_channels = 4, apply_initialization = False):
        super(Decoder, self).__init__()
        
        self.gf_dim = gf_dim
        #print(gf_dim)

        # Initialization
        #w_init = nn.init.trunc_normal_
        #b_init = nn.init.constant_
        #gamma_init = nn.init.ones_

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp(gf_dim * 32, gf_dim * 16, stride=(2,2,1), output_padding = (1,1,0))
        self.res0 = ResBlockUp(gf_dim * 16, gf_dim * 8, stride=(2,2,2), output_padding = (1,1,1))
        self.res1 = ResBlockUp(gf_dim * 8, gf_dim * 4, stride=(2,2,2), output_padding = (1,1,1))
        self.res2 = ResBlockUp(gf_dim * 4, gf_dim * 2, stride=(2,2,2), output_padding = (1,1,1))

        # 1st convolution block: convolution, followed by batch normalization and activation
        self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        if apply_initialization:
            self.conv1.apply(self._initialize_weights)

        # 2nd convolution block: convolution
        self.conv2 = nn.Conv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        if apply_initialization:
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


class VAE_convT(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False):
        super(VAE_convT, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization)
        self.decoder = Decoder(gf_dim = gf_dim, out_channels = out_channels, apply_initialization=apply_initialization)

    def forward(self, x):
        x = x.get('input_images')
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decoder
        decoder_output = self.decoder(self.guessed_z)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        return dict

class VAE(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, ):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim)
        self.decoder = Decoder_interpolate(gf_dim = gf_dim, out_channels = out_channels)

    def forward(self, x):
        x = x.get('input_images')
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decoder
        decoder_output = self.decoder(self.guessed_z)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        return dict
    
class VAE_linear(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, z_dim:int = 3072, apply_initialization:bool =False, interpolate:bool = False):
        super(VAE_linear, self).__init__()

        self.encoder = Encoder_linear(in_channels = in_channels, gf_dim = gf_dim, z_dim=z_dim, apply_initialization=apply_initialization)
        self.decoder = Decoder_linear(gf_dim = gf_dim, out_channels = out_channels, z_dim=z_dim, apply_initialization=apply_initialization, interpolate= interpolate)

    def forward(self, x):
        x = x.get('input_images')
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decoder
        decoder_output = self.decoder(self.guessed_z)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        return dict
