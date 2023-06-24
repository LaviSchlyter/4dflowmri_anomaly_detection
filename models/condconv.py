#https://github.com/d-li14/condconv.pytorch/blob/master/condconv.py

# Adapt to 3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class route_func(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf

    Args:
        in_channels (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, num_experts):
        super(route_func, self).__init__()
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return self.sigmoid(x)


#class CondConv2d(nn.Module):
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
                 num_experts=1):
        #super(CondConv2d, self).__init__()
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
        x = x.view(1, -1, h, w, d)
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
    


class CondConvResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride = (2,2,2), act=True):
        super(CondConvResBlockDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        #self.padding = padding
        #self.dilation = dilation
        #self.groups = groups
        #self.num_experts = num_experts
        self.act = act

        self.conv1 = CondConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = CondConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = CondConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()
    
    def forward(self, inputs, routing_weight1, routing_weight2, routing_weight3):
        conv1_out = self.conv1(inputs, routing_weight1)
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

# I don't think we need this for convtranpose
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



class CondConvEncoder(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8) -> None:
        super(CondConvEncoder, self).__init__()
        
        # 1st conv block
        self.conv1 = CondConv3d(in_channels, gf_dim, kernel_size=3, stride=(2,2,2), padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu1 = nn.LeakyReLU(0.2)

        # CondConv Res-Blocks
        self.res1 = CondConvResBlockDown(gf_dim, gf_dim, stride=(2,2,2))
        self.res2 = CondConvResBlockDown(gf_dim, gf_dim*2, stride=(2,2,2))
        self.res3 = CondConvResBlockDown(gf_dim*2, gf_dim*4, stride=(2,2,2))
        self.res4 = CondConvResBlockDown(gf_dim*4, gf_dim*8, stride=(2,2,1))

        # Latent convolutional layers
        self.latent = CondConv3d(gf_dim*8, gf_dim*32, kernel_size=1, padding=0)
        self.latent_std = CondConv3d(gf_dim*8, gf_dim*32, kernel_size=1, padding=0)

    def forward(self, x, routing_weight1, routing_weight2, routing_weight3, routing_weight4, routing_weight5, routing_weight6):
        conv1 = self.conv1(x, routing_weight1)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        # CondConv Res-Blocks
        res1 = self.res1(conv1, routing_weight2, routing_weight3, routing_weight4)
        res2 = self.res2(res1, routing_weight5, routing_weight6, routing_weight7)
        res3 = self.res3(res2, routing_weight8, routing_weight9, routing_weight10)
        res4 = self.res4(res3, routing_weight11, routing_weight12, routing_weight13)

        # Latent convolutional layers
        latent = self.latent(res4, routing_weight14)
        latent_std = self.latent_std(res4, routing_weight15)

        return latent, latent_std

class Decoder(nn.Module):
    def __init__(self, gf_dim = 8, out_channels = 4):
        super(Decoder, self).__init__()
        
        self.gf_dim = gf_dim

        # Res-Blocks (for effective deep architecture)
        self.resp1 = ResBlockUp(gf_dim * 32, gf_dim * 16, stride=(2,2,1), output_padding = (1,1,0))
        self.res0 = ResBlockUp(gf_dim * 16, gf_dim * 8, stride=(2,2,2), output_padding = (1,1,1))
        self.res1 = ResBlockUp(gf_dim * 8, gf_dim * 4, stride=(2,2,2), output_padding = (1,1,1))
        self.res2 = ResBlockUp(gf_dim * 4, gf_dim * 2, stride=(2,2,2), output_padding = (1,1,1))

        # 1st convolution block: convolution, followed by batch normalization and activation
        #self.conv1 = nn.Conv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = CondConv3d(gf_dim * 2, gf_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(gf_dim)
        
        # 2nd convolution block: convolution
        #self.conv2 = nn.Conv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CondConv3d(gf_dim, out_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x, routing_weight1, routing_weight2):
        #print(' Input to decoder has the following shape:' + str(x.shape))
        # Res-Blocks (for effective deep architecture)
        resp1 = self.resp1(x)
        res0 = self.res0(resp1)
        res1 = self.res1(res0)
        res2 = self.res2(res1)

        # 1st convolution block: convolution, followed by batch normalization and activation
        conv1 = self.conv1(res2, routing_weight1)
        conv1 = self.bn1(conv1)
        conv1 = F.leaky_relu(conv1, 0.2)

        # 2nd convolution block: convolution
        conv2 = self.conv2(conv1, routing_weight2)


        return conv2
        



#####################################################################################3
# That VERSION IS PROBABLY NOT WORKING

















    
class CondConvResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(2,2,2), padding=1, dilation=1, groups=1, bias=True,
                 num_experts=1, act=True):
        #super(CondConv2d, self).__init__()
        super(CondConvResBlockDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts
        self.act = act
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

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
        x = x.view(1, -1, h, w, d)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, in_channels, kh, kw, kd)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            conv1_out = F.conv3d(x, weight=combined_weight, bias= combined_bias, stride=self.stride, padding=self.padding)
            bn1_out = self.bn1(conv1_out)
            act1_out = self.activation(bn1_out)

            conv2_out = F.conv3d(act1_out, weight=combined_weight, bias= combined_bias, padding=self.padding)
            bn2_out = self.bn2(conv2_out)
            act2_out = self.activation(bn2_out)

            conv3_out = F.conv3d(x, weight=combined_weight, bias= combined_bias, stride=self.stride, padding=self.padding)
            bn3_out = self.bn3(conv3_out)
            act3_out = self.activation(bn3_out)
            conv_out = act2_out + act3_out
            # TODO: Fix size
            output = conv_out.view(b, c_out, conv_out.size(-3), conv_out.size(-2), conv_out.size(-1))
        else:
            conv1_out = F.conv3d(x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding)
            bn1_out = self.bn1(conv1_out)
            act1_out = self.activation(bn1_out)

            conv2_out = F.conv3d(act1_out, weight=combined_weight, bias=None, padding=self.padding)
            bn2_out = self.bn2(conv2_out)
            act2_out = self.activation(bn2_out)

            conv3_out = F.conv3d(x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding)
            bn3_out = self.bn3(conv3_out)
            act3_out = self.activation(bn3_out)
            conv_out = act2_out + act3_out

            output = conv_out.view(b, c_out, conv_out.size(-3), conv_out.size(-2), conv_out.size(-1))
        return output



class CondConvResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, act=True, stride =(2,2,2), output_padding = (1,1,1),num_experts=1, kernel_size=3,groups=1, bias=True):
        super(CondConvResBlockUp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.output_padding = output_padding
        self.act = act
        self.kernel_size = kernel_size
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

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

        b,c,h,w,d = x.size()
        k, c_out, in_channels, kh, kw, kd = self.weight.size()
        x = x.view(1, -1, h, w, d)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw, kd)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            conv1_out = F.conv_transpose3d(x, weight=combined_weight, bias = combined_bias, stride=self.stride, padding=1, output_padding=self.output_padding)
            bn1_out = self.bn1(conv1_out)
            act1_out = self.activation(bn1_out)

            conv2_out = F.conv_transpose3d(act1_out, weight=combined_weight, bias = combined_bias, padding=1)
            bn2_out = self.bn2(conv2_out)
            act2_out = self.activation(bn2_out)

            conv3_out = F.conv_transpose3d(x, weight=combined_weight, bias = combined_bias, stride=self.stride, padding=1, output_padding=self.output_padding)
            bn3_out = self.bn3(conv3_out)   
            act3_out = self.activation(bn3_out)

            conv_out = act2_out + act3_out

            output = conv_out.view(b, c_out, conv_out.size(-3), conv_out.size(-2), conv_out.size(-1))

        else:
            conv1_out = F.conv_transpose3d(x, weight=combined_weight, bias = None, stride=self.stride, padding=1, output_padding=self.output_padding)
            bn1_out = self.bn1(conv1_out)
            act1_out = self.activation(bn1_out)

            conv2_out = F.conv_transpose3d(act1_out, weight=combined_weight, bias = None, padding=1)
            bn2_out = self.bn2(conv2_out)
            act2_out = self.activation(bn2_out)

            conv3_out = F.conv_transpose3d(x, weight=combined_weight, bias = None, stride=self.stride, padding=1, output_padding=self.output_padding)
            bn3_out = self.bn3(conv3_out)
            act3_out = self.activation(bn3_out)

            conv_out = act2_out + act3_out

            output = conv_out.view(b, c_out, conv_out.size(-3), conv_out.size(-2), conv_out.size(-1))

        return output

            


class CondConvEncoder(nn.Module):
    def __init__(self, in_channels:int, gf_dim:int = 8, num_experts=1, kernel_size=3, groups=1, bias=True) -> None:
        super(CondConvEncoder, self).__init__()

        self.in_channels = in_channels
        self.gf_dim = gf_dim
        self.num_experts = num_experts
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, gf_dim, in_channels // groups, kernel_size, kernel_size,kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, gf_dim))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.bn1 = nn.BatchNorm3d(gf_dim)
        self.relu = nn.LeakyReLU(0.2)

        # Res-Blocks
        self.res1 = CondConvResBlockDown(gf_dim, gf_dim, kernel_size=kernel_size, stride=(2,2,2), num_experts=num_experts, groups=groups, bias=bias)
        self.res2 = CondConvResBlockDown(gf_dim, gf_dim*2, kernel_size=kernel_size, stride=(2,2,2), num_experts=num_experts, groups=groups, bias=bias)
        self.res3 = CondConvResBlockDown(gf_dim*2, gf_dim*4, kernel_size=kernel_size, stride=(2,2,2), num_experts=num_experts, groups=groups, bias=bias)
        self.res4 = CondConvResBlockDown(gf_dim*4, gf_dim*8, kernel_size=kernel_size, stride=(2,2,1), num_experts=num_experts, groups=groups, bias=bias)


    def forward(self, x, routing_weight):
        b, c_in, h, w, d = x.size()
        k, c_out, c_in, kh, kw, kd = self.weight.size()
        x = x.view(1, -1, h, w, d)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw, kd)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            conv1 = F.conv3d(x, weight=combined_weight, bias=combined_bias, padding=1)
            conv1 = self.bn1(conv1)
            conv1 = self.relu(conv1)

            # Res-Blocks
            res1 = self.res1(conv1, routing_weight)
            res2 = self.res2(res1, routing_weight)
            res3 = self.res3(res2, routing_weight)
            res4 = self.res4(res3, routing_weight)




        
