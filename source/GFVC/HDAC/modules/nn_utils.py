import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputLayer(nn.Module):
    def __init__(self, in_features, out_features=3, kernel_size=(7,7), padding=(3,3), activation='sigmoid') -> None:
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

class Mask(nn.Module):
    def __init__(self, in_features, out_features=3, kernel_size=(7,7), padding=(3,3)) -> None:
        super(Mask, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.softmax(self.conv(x))
        return out

class KP_Output(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(7, 7), padding=(3,3)) -> None:
        super(KP_Output, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size=(3,3), padding=(1,1)):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        return out

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features,scale_factor=2, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=self.scale_factor,mode='bilinear', align_corners=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()        
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs
    
class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256,norm='batch',qp=False):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))



'''
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

This is an implementation of ECA-Net(CVPR2020,paper), created by Banggu Wu.

'''

def conv3x3(in_planes, out_planes,kernel_size=(3,3), stride=(1,1), padding=(1,1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,groups=1, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, kernel_size=5):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECABlock2d(nn.Module):
    def __init__(self, in_planes, kernel_size=(3,3),stride=(1,1), padding=(1,1)):
        super(ECABlock2d, self).__init__()
        self.norm_1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, in_planes,kernel_size, stride, padding)
        self.norm_2 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_planes, in_planes, kernel_size, stride, padding)
        self.norm_3 = nn.BatchNorm2d(in_planes)
        self.eca = ECA()

    def forward(self, x):
        out = self.norm_1(x)
        out = self.norm_2(self.conv1(out))
        out = self.relu(out)
        out = self.norm_3(self.conv2(out))
        out = self.eca(out)
        out += x
        return self.relu(out)

    def __init__(self, in_planes, out_planes, kernel_size=3,stride=1):
        super(ECA_ATT, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, 1)
        self.eca = ECA(kernel_size)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu_1(self.conv1(x))
        out = self.eca(self.conv2(out))        
        out += residual
        return self.relu_2(out)
    

    def __init__(self, in_channel=3, out_channel=3, block_expansion=64, group_norm=False, group_norm_channel=-1):
        super(UNet, self).__init__()
        if group_norm_channel==-1:
            group_norm_channel = block_expansion
        self.dconv_down1 = double_conv(in_channel, block_expansion, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down2 = double_conv(block_expansion, block_expansion*2, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down3 = double_conv(block_expansion*2, block_expansion*2, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down4 = double_conv(block_expansion*2, block_expansion*4, group_norm = group_norm, group_norm_channel=group_norm_channel)        

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   

        self.dconv_up3 = double_conv(block_expansion*2 + block_expansion*4, block_expansion*2, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_up2 = double_conv(block_expansion*2 + block_expansion*2, block_expansion, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_up1 = double_conv(block_expansion + block_expansion, block_expansion, group_norm = group_norm, group_norm_channel=group_norm_channel)

        self.conv_last = torch.nn.Conv2d(block_expansion, out_channel, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)    
  
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)  
          
        x = torch.cat([x, conv2], dim=1)    

        x = self.dconv_up2(x)
        x = self.upsample(x)  
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)
        
        x = self.conv_last(x)

        return torch.sigmoid(x)

    def __init__(self,in_features,num_layers=2, kernel_size=3,padding=1):
        super(AffineLayer, self).__init__()
        self.layers = nn.Sequential()
        for idx in range(num_layers):
            self.layers.add_module(f'conv_{idx}',nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding))
            self.layers.add_module(f'relu_{idx}', nn.ReLU())
    def forward(self,x):
        return self.layers(x)