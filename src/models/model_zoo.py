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
    def __init__(self, gf_dim = 8, aux_output_dim = 0, out_channels = 4, apply_initialization = False):
        super(Decoder, self).__init__()
        
        self.gf_dim = gf_dim
        #print(gf_dim)

        # Initialization
        #w_init = nn.init.trunc_normal_
        #b_init = nn.init.constant_
        #gamma_init = nn.init.ones_

        self.resp1_input_dim = gf_dim * 32 + aux_output_dim  # Adjusted input dimension

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp(self.resp1_input_dim, gf_dim * 16, stride=(2,2,1), output_padding = (1,1,0))
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



## Auxillary network for the rotation matrix 
class DenseAuxNet(nn.Module):
    """
    Encoder-decoder network for analyzing 3x3 rotation matrices, predicting the trace and Euler angles.
    
    Inputs:
        - 3x3 rotation matrix, flattened to a 9-element vector.
    
    Outputs:
        - 4-element vector: 1 element for the trace of the matrix and 3 elements for the Euler angles.
    
    The encoder compresses the input to a lower-dimensional space, and the decoder reconstructs the target outputs.
    """
    def __init__(self, input_size:int = 9, output_size:int = 3) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        # Flatten 3x3 rotation matrix to a 9-element vector
        x = x.view(x.size(0), -1)  # Assuming x is [batch_size, 3, 3]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        return output
    
## Auxillary network for the rotation matrix 
class DenseDeepAuxNet(nn.Module):
    """
    Deep Encoder-decoder network for analyzing 3x3 rotation matrices, predicting the trace and Euler angles.
    
    Inputs:
        - 3x3 rotation matrix, flattened to a 9-element vector.
    
    Outputs:
        - 4-element vector: 1 element for the trace of the matrix and 3 elements for the Euler angles.
    
    The encoder compresses the input to a lower-dimensional space, and the decoder reconstructs the target outputs.
    """
    def __init__(self, input_size:int = 9, output_size:int = 3) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, output_size)

    def forward(self, x):
        # Flatten 3x3 rotation matrix to a 9-element vector
        x = x.view(x.size(0), -1)  # Assuming x is [batch_size, 3, 3]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        output = self.fc7(x)
        
        return output


class EncDecDeepAuxNet(nn.Module):
    """
    This class defines an encoder-decoder auxiliary network tailored for analyzing 3x3 rotation matrices.
    The encoder compresses the rotation matrix into a compact representation, which the decoder then uses
    to predict the trace and Euler angles of the matrix.

    Input: 3x3 rotation matrix
    Outputs: Trace of the matrix (1-element) and Euler angles (3-element vector)
    """
    def __init__(self, input_size:int=9, encoded_size:int=6) -> None:
        super().__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_size, 32)
        self.encoder_fc2 = nn.Linear(32, 64)
        self.encoder_fc3 = nn.Linear(64, 128)
        self.encoder_fc4 = nn.Linear(128, encoded_size)
        
        # Decoder for Trace
        self.decoder_trace_fc1 = nn.Linear(encoded_size, 32)
        self.decoder_trace_fc2 = nn.Linear(32, 16)
        self.decoder_trace_fc3 = nn.Linear(16, 8)
        self.decoder_trace_fc4 = nn.Linear(8, 1)  
        
        # Decoder for Euler Angles
        self.decoder_euler_fc1 = nn.Linear(encoded_size, 64)
        self.decoder_euler_fc2 = nn.Linear(64, 32)
        self.decoder_euler_fc3 = nn.Linear(32, 16)
        self.decoder_euler_fc4 = nn.Linear(16, 3)  

    def forward(self, x):
        # Flatten the input 3x3 matrix to a 9-element vector
        x = x.view(x.size(0), -1)
        
        # Encoder pathway
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = F.relu(self.encoder_fc3(x))
        encoded = F.relu(self.encoder_fc4(x))
        
        # Decoder pathway for Trace
        trace = F.relu(self.decoder_trace_fc1(encoded))
        trace = F.relu(self.decoder_trace_fc2(trace))
        trace = F.relu(self.decoder_trace_fc3(trace))
        trace = self.decoder_trace_fc4(trace)
        
        # Decoder pathway for Euler Angles
        euler_angles = F.relu(self.decoder_euler_fc1(encoded))
        euler_angles = F.relu(self.decoder_euler_fc2(euler_angles))
        euler_angles = F.relu(self.decoder_euler_fc3(euler_angles))
        euler_angles = self.decoder_euler_fc4(euler_angles)
        
        return trace, euler_angles, encoded

class EncDecDeepBNAuxNet(nn.Module):
    """
    This class defines an encoder-decoder auxiliary network tailored for analyzing 3x3 rotation matrices.
    The encoder compresses the rotation matrix into a compact representation, which the decoder then uses
    to predict the trace and Euler angles of the matrix.

    Input: 3x3 rotation matrix
    Outputs: Trace of the matrix (1-element) and Euler angles (3-element vector)
    """
    def __init__(self, input_size: int = 9, encoded_size: int = 6) -> None:
        super().__init__()

        # Encoder
        self.encoder_fc1 = nn.Linear(input_size, 32)
        self.encoder_fc2 = nn.Linear(32, 64)
        self.encoder_fc3 = nn.Linear(64, 128)
        self.encoder_fc4 = nn.Linear(128, encoded_size)

        # Batch normalization for encoder
        self.bn_encoder_fc1 = nn.BatchNorm1d(32)
        self.bn_encoder_fc2 = nn.BatchNorm1d(64)
        self.bn_encoder_fc3 = nn.BatchNorm1d(128)

        # Decoder for Trace
        self.decoder_trace_fc1 = nn.Linear(encoded_size, 32)
        self.decoder_trace_fc2 = nn.Linear(32, 16)
        self.decoder_trace_fc3 = nn.Linear(16, 8)
        self.decoder_trace_fc4 = nn.Linear(8, 1)

        # Decoder for Euler Angles
        self.decoder_euler_fc1 = nn.Linear(encoded_size, 64)
        self.decoder_euler_fc2 = nn.Linear(64, 32)
        self.decoder_euler_fc3 = nn.Linear(32, 16)
        self.decoder_euler_fc4 = nn.Linear(16, 3)

        # Batch normalization for decoder
        self.bn_decoder_trace_fc1 = nn.BatchNorm1d(32)
        self.bn_decoder_trace_fc2 = nn.BatchNorm1d(16)
        self.bn_decoder_trace_fc3 = nn.BatchNorm1d(8)

        self.bn_decoder_euler_fc1 = nn.BatchNorm1d(64)
        self.bn_decoder_euler_fc2 = nn.BatchNorm1d(32)
        self.bn_decoder_euler_fc3 = nn.BatchNorm1d(16)

    def forward(self, x):
        # Flatten the input 3x3 matrix to a 9-element vector
        x = x.view(x.size(0), -1)

        # Encoder pathway
        x = F.relu(self.bn_encoder_fc1(self.encoder_fc1(x)))
        x = F.relu(self.bn_encoder_fc2(self.encoder_fc2(x)))
        x = F.relu(self.bn_encoder_fc3(self.encoder_fc3(x)))
        encoded = F.relu(self.encoder_fc4(x))

        # Decoder pathway for Trace
        trace = F.relu(self.bn_decoder_trace_fc1(self.decoder_trace_fc1(encoded)))
        trace = F.relu(self.bn_decoder_trace_fc2(self.decoder_trace_fc2(trace)))
        trace = F.relu(self.bn_decoder_trace_fc3(self.decoder_trace_fc3(trace)))
        trace = self.decoder_trace_fc4(trace)

        # Decoder pathway for Euler Angles
        euler_angles = F.relu(self.bn_decoder_euler_fc1(self.decoder_euler_fc1(encoded)))
        euler_angles = F.relu(self.bn_decoder_euler_fc2(self.decoder_euler_fc2(euler_angles)))
        euler_angles = F.relu(self.bn_decoder_euler_fc3(self.decoder_euler_fc3(euler_angles)))
        euler_angles = self.decoder_euler_fc4(euler_angles)

        return trace, euler_angles, encoded

class EncDecAuxNet(nn.Module):
    """
    This class defines an encoder-decoder auxiliary network tailored for analyzing 3x3 rotation matrices.
    The encoder compresses the rotation matrix into a compact representation, which the decoder then uses
    to predict the trace and Euler angles of the matrix.

    Input: 3x3 rotation matrix
    Outputs: Trace of the matrix (1-element) and Euler angles (3-element vector)
    """
    def __init__(self, input_size:int=9, encoded_size:int = 6) -> None:
        super().__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_size, 12)
        self.encoder_fc2 = nn.Linear(12, encoded_size)
        
        # Decoder for Trace
        self.decoder_trace_fc1 = nn.Linear(encoded_size, 4)
        self.decoder_trace_fc2 = nn.Linear(4, 1)  
        
        # Decoder for Euler Angles
        self.decoder_euler_fc1 = nn.Linear(encoded_size, 8)
        self.decoder_euler_fc2 = nn.Linear(8, 3)  

    def forward(self, x):
        # Flatten the input 3x3 matrix to a 9-element vector
        x = x.view(x.size(0), -1)
        
        # Encoder pathway
        x = F.relu(self.encoder_fc1(x))
        encoded = F.relu(self.encoder_fc2(x))
        
        # Decoder pathway for Trace
        trace = F.relu(self.decoder_trace_fc1(encoded))
        trace = self.decoder_trace_fc2(trace)
        
        # Decoder pathway for Euler Angles
        euler_angles = F.relu(self.decoder_euler_fc1(encoded))
        euler_angles = self.decoder_euler_fc2(euler_angles)
        
        #return_dict = {'trace': trace, 'euler_angles': euler_angles, 'latent_space': encoded}
        return trace, euler_angles, encoded


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False):
        super(SimpleConvNet, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= 3072)
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

# Confusing... It is not a VAE, but a CNN. 

class VAE_convT(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False):
        super(VAE_convT, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= 3072)
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
    
class ConvWithAux(nn.Module):
    """
    This class is a combination of the encoder, decoder and the auxillary network
    """
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 3
        self.auxillary_network = DenseAuxNet(input_size=9, output_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)

        

    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples
        

        # Process rotation matrix through the auxiliary network
        aux_output = self.auxillary_network(rotation_matrix)
        #expanded_aux_output = aux_output.reshape(1, 3, 1, 1, 1).expand(-1, 3, 2, 2, 3)
        # Step 1: Expand the auxiliary output
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, 3, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, 3, 2, 2, 3]

        # Step 2: Concatenate along the channel dimension
        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 3, 2, 2, 3]

        #self.guessed_z = torch.cat([self.guessed_z, expanded_aux_output], dim=1)

        # Decoder
        decoder_output = self.decoder(self.guessed_z)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        return dict
    

class ConvWithDeepAux(nn.Module):
    """
    This class is a combination of the encoder, decoder and the auxillary network
    """
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithDeepAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 3
        self.auxillary_network = DenseDeepAuxNet(input_size=9, output_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)

        

    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')
        # Run encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples
        

        # Process rotation matrix through the auxiliary network
        aux_output = self.auxillary_network(rotation_matrix)
        #expanded_aux_output = aux_output.reshape(1, 3, 1, 1, 1).expand(-1, 3, 2, 2, 3)
        # Step 1: Expand the auxiliary output
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, 3, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, 3, 2, 2, 3]

        # Step 2: Concatenate along the channel dimension
        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 3, 2, 2, 3]

        #self.guessed_z = torch.cat([self.guessed_z, expanded_aux_output], dim=1)

        # Decoder
        decoder_output = self.decoder(self.guessed_z)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        return dict




class ConvWithEncDecAux(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithEncDecAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 6 # Here this is the size of the latent space 
        
        self.auxillary_network = EncDecAuxNet(input_size=9, encoded_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)
    
    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')

        # Rim encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Process rotation matrix through the auxiliary network
        #dict_aux_output = self.auxillary_network(rotation_matrix)
        trace, euler_angles, aux_output = self.auxillary_network(rotation_matrix)
        # Add the auxillary output to the dictionary
        dict_aux_output = {'trace': trace, 'euler_angles': euler_angles, 'latent_space': aux_output}
        aux_output = dict_aux_output.get('latent_space')
        
        
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, output_size_aux, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, output_size_aux, 2, 2, 3]

        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 6, 2, 2, 3]

        # Decoder
        decoder_output = self.decoder(self.guessed_z)

        # Add the auxillary output to the dictionary
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        dict.update(dict_aux_output)
        return dict
    

class ConvWithDeepEncDecAux(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithDeepEncDecAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 6 # Here this is the size of the latent space 
        
        self.auxillary_network = EncDecDeepAuxNet(input_size=9, encoded_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)
    
    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')

        # Rim encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Process rotation matrix through the auxiliary network
        #dict_aux_output = self.auxillary_network(rotation_matrix)
        trace, euler_angles, aux_output = self.auxillary_network(rotation_matrix)
        # Add the auxillary output to the dictionary
        dict_aux_output = {'trace': trace, 'euler_angles': euler_angles, 'latent_space': aux_output}
        aux_output = dict_aux_output.get('latent_space')
        
        
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, output_size_aux, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, output_size_aux, 2, 2, 3]

        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 6, 2, 2, 3]

        # Decoder
        decoder_output = self.decoder(self.guessed_z)

        # Add the auxillary output to the dictionary
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        dict.update(dict_aux_output)
        return dict
    
class ConvWithDeeperEncDecAux(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithDeeperEncDecAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 64 # Here this is the size of the latent space 
        
        self.auxillary_network = EncDecDeepAuxNet(input_size=9, encoded_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)
    
    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')

        # Rim encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Process rotation matrix through the auxiliary network
        #dict_aux_output = self.auxillary_network(rotation_matrix)
        trace, euler_angles, aux_output = self.auxillary_network(rotation_matrix)
        # Add the auxillary output to the dictionary
        dict_aux_output = {'trace': trace, 'euler_angles': euler_angles, 'latent_space': aux_output}
        aux_output = dict_aux_output.get('latent_space')
        
        
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, output_size_aux, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, output_size_aux, 2, 2, 3]

        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 6, 2, 2, 3]

        # Decoder
        decoder_output = self.decoder(self.guessed_z)

        # Add the auxillary output to the dictionary
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        dict.update(dict_aux_output)
        return dict

    
class ConvWithDeeperBNEncDecAux(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, apply_initialization:bool =False, z_dim:int = 3072):
        super(ConvWithDeeperBNEncDecAux, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, gf_dim = gf_dim, apply_initialization=apply_initialization, z_dim= z_dim)

        # 9 because rotation matrix is 3x3
        output_size_aux = 64 # Here this is the size of the latent space 
        
        self.auxillary_network = EncDecDeepBNAuxNet(input_size=9, encoded_size=output_size_aux)

        # The decoder has to change a bit, because we concatenate the latent space with the output of the auxillary network

        self.decoder = Decoder(gf_dim = gf_dim, aux_output_dim = output_size_aux, out_channels = out_channels, apply_initialization=apply_initialization)
    
    def forward(self, x):
        rotation_matrix = x.get('rotation_matrix')
        
        x = x.get('input_images')

        # Rim encoder network and get the latent space distribution
        z_mean, z_std, res = self.encoder(x)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Process rotation matrix through the auxiliary network
        #dict_aux_output = self.auxillary_network(rotation_matrix)
        trace, euler_angles, aux_output = self.auxillary_network(rotation_matrix)
        # Add the auxillary output to the dictionary
        dict_aux_output = {'trace': trace, 'euler_angles': euler_angles, 'latent_space': aux_output}
        aux_output = dict_aux_output.get('latent_space')
        
        
        aux_expanded = aux_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now: [batch_size, output_size_aux, 1, 1, 1]
        aux_expanded = aux_expanded.expand(-1, -1, 2, 2, 3)  # Now: [batch_size, output_size_aux, 2, 2, 3]

        self.guessed_z = torch.cat((self.guessed_z, aux_expanded), dim=1)  # Now: [batch_size, 256 + 6, 2, 2, 3]

        # Decoder
        decoder_output = self.decoder(self.guessed_z)

        # Add the auxillary output to the dictionary
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std, 'res': res}
        dict.update(dict_aux_output)
        return dict

    


class SimpleConvNetInterpolate(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, ):
        super(SimpleConvNet, self).__init__()

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
    
class SimpleConvNet_linear(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, z_dim:int = 3072, apply_initialization:bool =False, interpolate:bool = False):
        super(SimpleConvNet_linear, self).__init__()

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
