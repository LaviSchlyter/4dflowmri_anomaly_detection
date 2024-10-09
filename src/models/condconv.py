#https://github.com/d-li14/condconv.pytorch/blob/master/condconv.py

# Adapt to 3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RouteFunc(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        in_channels (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
        num_cond (int): Number of convultions to condition on. Default: 6+
    """

    def __init__(self, in_channels, num_experts, num_cond=29):
        super(RouteFunc, self).__init__()
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        #self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(in_channels, num_experts*num_cond, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.num_experts = num_experts

    def forward(self, x):
        # The input is the slice number of each image in the batch 
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        # Reshape to [b, num_cond, num_experts]
        x = x.view(x.size(0), -1, self.num_experts)
        return x
        
        
        

class RouteFunc2(nn.Module):
    def __init__(self, in_channels, num_experts, num_cond=29):
        super(RouteFunc2, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels*2)
        self.fc2 = nn.Linear(in_channels*2, in_channels*4)
        self.fc3 = nn.Linear(in_channels*4, num_experts*num_cond)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.num_experts = num_experts

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = x.view(x.size(0), -1, self.num_experts)
        return x
    

class ConvRouteFunc(nn.Module):
    def __init__(self, in_channels, num_experts, num_cond = 29):
        super(ConvRouteFunc, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)


        self.fc1 = nn.Linear(128*4*4*12, in_channels*2)
        self.fc2 = nn.Linear(in_channels*2, in_channels*4)
        self.fc3 = nn.Linear(in_channels*4, num_experts*num_cond)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.num_experts = num_experts
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # 8,64,8,8,6
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        x = x.view(x.size(0), -1, self.num_experts)
        
        return x
        



        


class CondConv3d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=3):
        
        super(CondConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size,kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, in_channels, h, w, d = x.size()
        k, c_out, in_channels, kh, kw, kd = self.weight.size()
        x = x.reshape(1, -1, h, w, d)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, in_channels, kh, kw, kd)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-3), output.size(-2), output.size(-1))
        return output
    
class CondConvTranspose3d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=3, output_padding = 0):
        
        super(CondConvTranspose3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.output_padding = output_padding

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, in_channels // groups,out_channels, kernel_size,kernel_size,kernel_size)
            )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, in_channels, h, w, d = x.size()
        k, in_channels, c_out, kh, kw, kd = self.weight.size()
        # 3, 256,256,3,3,3
        x = x.reshape(1, -1, h, w, d)
        # 1st: 1, 2048, 2,2,3
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_out, kh, kw, kd)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv_transpose3d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b, output_padding = self.output_padding)
        else:
            output = F.conv_transpose3d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b, output_padding=self.output_padding)

        output = output.view(b, c_out, output.size(-3), output.size(-2), output.size(-1))
        return output
    


class CondConvResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride = (2,2,2), act=True, num_experts=3):
        super(CondConvResBlockDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #self.padding = padding
        #self.dilation = dilation
        #self.groups = groups
        #self.num_experts = num_experts
        self.act = act

        self.conv1 = CondConv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, num_experts=num_experts)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = CondConv3d(in_channels, out_channels, kernel_size=3, padding=1, num_experts=num_experts)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = CondConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, num_experts=num_experts)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()
    
    def forward(self, inputs, routing_weight1, routing_weight2, routing_weight3):
        conv1_out = self.conv1(inputs, routing_weight1)
        # [8,8,8,8,6]
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out, routing_weight2)
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        conv3_out = self.conv3(inputs, routing_weight3)
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out



class CondConvResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True, stride =(2,2,2), output_padding = (1,1,1), num_experts=3):
        super(CondConvResBlockUp, self).__init__()

        self.conv1 = CondConvTranspose3d(filters_in, filters_in, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, num_experts=num_experts)
        self.bn1 = nn.BatchNorm3d(filters_in)
        self.conv2 = CondConvTranspose3d(filters_in, filters_out, kernel_size=3, padding=1, num_experts=num_experts)
        self.bn2 = nn.BatchNorm3d(filters_out)
        self.conv3 = CondConvTranspose3d(filters_in, filters_out, kernel_size=3, stride=stride, padding=1, output_padding=output_padding, num_experts=num_experts)
        self.bn3 = nn.BatchNorm3d(filters_out)

        if act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()



    def forward(self, inputs, routing_weight1, routing_weight2, routing_weight3):
        # inputs: [b, 256, 2,2,3]
        conv1_out = self.conv1(inputs, routing_weight1)
        # conv1_out: [b, 256, 4,4,3]
        bn1_out = self.bn1(conv1_out)
        act1_out = self.activation(bn1_out)

        conv2_out = self.conv2(act1_out, routing_weight2)
        # conv1_out: [b, 256, 4,4,3]
        bn2_out = self.bn2(conv2_out)
        act2_out = self.activation(bn2_out)

        conv3_out = self.conv3(inputs, routing_weight3)
        # 256, 4,4,3
        bn3_out = self.bn3(conv3_out)
        act3_out = self.activation(bn3_out)

        conv_out = act2_out + act3_out

        return conv_out
    


class CondConvEncoder(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8, num_experts:int = 3) -> None:
        super(CondConvEncoder, self).__init__()
        
        # 1st conv block
        self.conv1 = CondConv3d(in_channels, gf_dim, kernel_size=3, padding=1, num_experts=num_experts)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        # CondConv Res-Blocks
        self.res1 = CondConvResBlockDown(gf_dim, gf_dim, stride=(2,2,2) , num_experts=num_experts)
        self.res2 = CondConvResBlockDown(gf_dim, gf_dim*2, stride=(2,2,2) , num_experts=num_experts)
        self.res3 = CondConvResBlockDown(gf_dim*2, gf_dim*4, stride=(2,2,2) , num_experts=num_experts)
        self.res4 = CondConvResBlockDown(gf_dim*4, gf_dim*8, stride=(2,2,1) , num_experts=num_experts)

        # Latent convolutional layers
        self.latent = CondConv3d(gf_dim*8, gf_dim*32, kernel_size=1, padding=0, num_experts=num_experts)
        self.latent_std = CondConv3d(gf_dim*8, gf_dim*32, kernel_size=1, padding=0, num_experts=num_experts)

    def forward(self, x, routing_weights):
        # Routing weights have size (b, #conv, #experts)
        conv1 = self.conv1(x, routing_weights[:,0,:])
        # [8,8,32,32,24]
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        # CondConv Res-Blocks
        res1 = self.res1(conv1, routing_weights[:, 1,:], routing_weights[:, 2,:], routing_weights[:, 3,:])
        # [8,8,16,16,12]
        res2 = self.res2(res1, routing_weights[:, 4,:], routing_weights[:, 5,:], routing_weights[:, 6,:])
        # [8,16,8,8,6]
        res3 = self.res3(res2, routing_weights[:, 7,:], routing_weights[:, 8,:], routing_weights[:, 9,:])
        # [8,32,4,4,3]
        res4 = self.res4(res3, routing_weights[:,10,:], routing_weights[:,11,:], routing_weights[:,12,:])
        # [8,64,2,2,3]

        # Latent convolutional layers
        latent = self.latent(res4, routing_weights[:,13,:])
        latent_std = self.latent_std(res4, routing_weights[:,14,:])

        return latent, latent_std


class CondConvDecoder(nn.Module):
    def __init__(self, gf_dim = 8, out_channels = 4, num_experts = 3):
        super(CondConvDecoder, self).__init__()
        
        self.gf_dim = gf_dim

        # Res-Blocks (for effective deep architecture)
        self.resp1 = CondConvResBlockUp(gf_dim * 32, gf_dim * 16, stride=(2,2,1), output_padding = (1,1,0), num_experts=num_experts)
        self.res0 = CondConvResBlockUp(gf_dim * 16, gf_dim * 8, stride=(2,2,2), output_padding = (1,1,1), num_experts=num_experts)
        self.res1 = CondConvResBlockUp(gf_dim * 8, gf_dim * 4, stride=(2,2,2), output_padding = (1,1,1), num_experts=num_experts)
        self.res2 = CondConvResBlockUp(gf_dim * 4, gf_dim * 2, stride=(2,2,2), output_padding = (1,1,1), num_experts=num_experts)

        # 1st convolution block: convolution, followed by batch normalization and activation
        #self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = CondConv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1, num_experts=num_experts)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        
        # 2nd convolution block: convolution
        #self.conv2 = nn.Conv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CondConv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1, num_experts=num_experts)
        

    def forward(self, x, routing_weights):
        #print(' Input to decoder has the following shape:' + str(x.shape))
        # Res-Blocks (for effective deep architecture)
        resp1 = self.resp1(x, routing_weights[:,15,:], routing_weights[:,16,:], routing_weights[:,17,:])
        res0 = self.res0(resp1, routing_weights[:,18,:], routing_weights[:,19,:], routing_weights[:,20,:])
        res1 = self.res1(res0, routing_weights[:,21,:], routing_weights[:,22,:], routing_weights[:,23,:])
        res2 = self.res2(res1, routing_weights[:,24,:], routing_weights[:,25,:], routing_weights[:,26,:])

        # 1st convolution block: convolution, followed by batch normalization and activation
        conv1 = self.conv1(res2, routing_weights[:,27,:])
        conv1 = self.bn1(conv1)
        conv1 = F.leaky_relu(conv1, 0.2)

        # 2nd convolution block: convolution
        conv2 = self.conv2(conv1, routing_weights[:,28,:])

        return conv2
        




class CondVAE(nn.Module):
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, num_experts:int = 3):

        super(CondVAE, self).__init__()
        
        self.encoder = CondConvEncoder(in_channels, gf_dim, num_experts=num_experts)
        self.decoder = CondConvDecoder(gf_dim, out_channels, num_experts=num_experts)

        #self.route_func_encoder = RouteFunc(in_channels = 1, num_experts=num_experts, num_cond= 29)
        self.route_func_encoder = RouteFunc2(in_channels = 1, num_experts=num_experts, num_cond= 29)
    
    def forward(self, input_dict):
        x = input_dict.get('input_images')
        z_slices = input_dict.get('batch_z_slice')
        # z_slices are going to be used for the routing function 

        # Routing weights 
        routing_weights = self.route_func_encoder(z_slices)
        # Size of routing weights: [b, num_cond, num_experts]

        # Encoder
        z_mean, z_std = self.encoder(x, routing_weights)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decoder
        decoder_output = self.decoder(self.guessed_z, routing_weights)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std}
        return dict
    
class CondConv(nn.Module):
    # In this version, the routing funciton will be taking the adjacent slices to the current inputed slices in order to tune the weights of the network
    def __init__(self, in_channels:int =4, gf_dim:int =8, out_channels:int =4, num_experts:int = 3):

        super(CondConv, self).__init__()
        
        self.encoder = CondConvEncoder(in_channels, gf_dim, num_experts=num_experts)
        self.decoder = CondConvDecoder(gf_dim, out_channels, num_experts=num_experts)
        # 4 channels times 3 slices entering the routing function
        self.route_func_encoder = ConvRouteFunc(in_channels=12, num_experts=num_experts, num_cond=29)
    
    def forward(self, input_dict):
        x = input_dict.get('input_images')
        adjacent_slices = input_dict.get('adjacent_batch_slices') 


        # Routing weights 
        routing_weights = self.route_func_encoder(adjacent_slices)
        # Size of routing weights: [b, num_cond, num_experts]

        # Encoder
        z_mean, z_std = self.encoder(x, routing_weights)

        # Sample the latent space using a normal distribution (samples)
        samples = torch.randn_like(z_mean)
        self.guessed_z = z_mean + z_std * samples

        # Decoder
        decoder_output = self.decoder(self.guessed_z, routing_weights)
        dict = {'decoder_output': decoder_output, 'mu': z_mean, 'z_std': z_std}
        return dict



