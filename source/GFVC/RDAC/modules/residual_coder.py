import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.layers import conv3x3
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class ConvBlock(nn.Module):
    def __init__(self, in_ft,out_ft, kernel_size=5,stride=2, act=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ft,out_ft,kernel_size=kernel_size,stride=stride,padding=kernel_size // 2,)
        self.act = act
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor)->torch.Tensor:
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x
        
class DeconvBlock(nn.Module):
    def __init__(self,in_ft, out_ft,kernel_size=5, stride=2, act=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
                                        in_ft,out_ft,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=stride - 1,
                                        padding=kernel_size // 2,
                                    )
        self.act = act
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.deconv(x)
        if self.act:
            x = self.relu(x)
        return x

class ResidualCoder(CompressionModel):
    '''An extension of the Scale-Space Hyperprior model adapted for 
        Low bitrate residual coding
    '''
    def __init__(self,in_ft,out_ft, N, M, scale_factor=1,**kwargs):
        super(ResidualCoder, self).__init__()
        num_int_layers = kwargs['num_intermediate_layers']
        self.g_a = nn.Sequential()
        self.g_a.add_module("inp", ConvBlock(in_ft, N))
        for idx in range(num_int_layers):
            self.g_a.add_module(f"conv_{idx}",ConvBlock(N,N))
        self.g_a.add_module("out",ConvBlock(N, M, act=False))
        
        self.g_s = nn.Sequential()
        self.g_s.add_module('inp', DeconvBlock(M,N))
        for idx in range(num_int_layers):
            self.g_s.add_module(f"deconv_{idx}", DeconvBlock(N,N))
        self.g_s.add_module('out',DeconvBlock(N, out_ft, act=False))
        
        
        self.h_a = nn.Sequential(
            ConvBlock(M, N, stride=1, kernel_size=3,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N, act=False))
        
        self.h_s = nn.Sequential(
            DeconvBlock(N, M, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M, M * 3 // 2, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M * 3 // 2, M * 2, stride=1, kernel_size=3, act=False),)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = N
        self.M = M

        #some inputs may be subsambled before compression
        self.scale_factor = scale_factor

        self.variable_bitrate = kwargs['variable_bitrate']
        if self.variable_bitrate:
            self.levels = kwargs['levels']
            self.gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            self.inverse_gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            self.hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
            self.inverse_hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    def resize(self, frame, scale_factor=1):
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)

    def estimate_bitrate(self, likelihood):
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 
    

    def compute_gain(self, x: torch.Tensor, rate_idx: int,hyper=False)-> torch.Tensor:
        if hyper:
            x =  x * torch.abs(self.hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x

    def compute_inverse_gain(self, x: torch.Tensor, rate_idx: int,hyper=False)->torch.Tensor:
        if hyper:
            x =  x * torch.abs(self.inverse_hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.inverse_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_interpolated_gain(self, x:torch.Tensor, rate_idx, q_value, hyper=False)->torch.Tensor:
        if hyper:
            gain = torch.abs(self.hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.hyper_gain[rate_idx + 1]) * q_value
        else:
            gain = torch.abs(self.gain[rate_idx]) * (1 - q_value) + torch.abs(self.gain[rate_idx + 1]) * q_value
        x = x * gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_inverse_interpolated_gain(self, x:torch.Tensor, rate_idx:int, q_value:float, hyper=False)-> torch.Tensor:
        if hyper:
            inv_gain = torch.abs(self.inverse_hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_hyper_gain[rate_idx + 1]) * (q_value)
        else:
            inv_gain = torch.abs(self.inverse_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_gain[rate_idx + 1]) * (q_value)
        x = x * inv_gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x

    def forward(self, x,rate_idx=0):
        if self.scale_factor != 1:
            x  = self.resize(x, self.scale_factor)
        B,H,W,_ = x.shape
        y = self.g_a(x)
        #apply gain on latent (y)
        if self.variable_bitrate:
            y = self.compute_gain(y, rate_idx)

        z = self.h_a(y)
        #apply gain in hyperprior (z)
        if self.variable_bitrate:
            z = self.compute_gain(z, rate_idx, hyper=True)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_bpp = self.estimate_bitrate(z_likelihoods)/(B*H*W)
        #apply inverse gain on hyperprior (z_hat)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        y_bpp = self.estimate_bitrate(y_likelihoods)/(B*H*W)
        #apply inverse gain on latent (y_hat)
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)

        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat  = self.resize(x_hat, 1//self.scale_factor)
        total_bpp = y_bpp+z_bpp
        return x_hat.clamp(-1,1), total_bpp, y_likelihoods

    def similarity(self, prev, cur)->torch.Tensor:
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(prev, cur)
        return output

    def rans_compress(self, residual,  prev_latent, rate_idx=0,q_value=1.0,use_skip=False, skip_thresh=0.95, scale_factor=1.0):
        enc_start = time.time()
        # self.scale_factor = scale_factor
        B,C,H,W = residual.shape
        if self.scale_factor != 1:
            residual = self.downsample(residual,self.scale_factor)

        y = self.g_a(residual)
        if prev_latent != None and use_skip:
            sim = torch.mean(self.similarity(prev_latent, y)).item()
        else:
            sim = 0
        if sim > skip_thresh:
            #skip this residual
            return None, True
        else:
            if self.variable_bitrate:
                y = self.compute_interpolated_gain(y, rate_idx, q_value)
        
            z = self.h_a(y)
            if self.variable_bitrate:
                z = self.compute_interpolated_gain(z, rate_idx, q_value, hyper=True)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            if self.variable_bitrate:
                z_hat = self.compute_inverse_interpolated_gain(z_hat, rate_idx, q_value, hyper=True)

            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            # scale_h, mean_h = self.get_averages(scales_hat, means_hat, H,W)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
            bts = (len(y_strings[0])+len(z_strings[0])) * 8
            enc_time = time.time() - enc_start
            dec_start = time.time()
            res_hat = self.rans_decompress([y_strings, z_strings], z.size()[-2:],rate_idx=rate_idx, q_value=q_value)
            dec_time = time.time() - dec_start
            #update bitstream info
            out = {'time':{'enc_time': enc_time,'dec_time': dec_time},
                    'bitstring_size':bts}
            out.update({'res_hat':res_hat,'prev_latent':y})
            return out, False

    def rans_decompress(self, strings, shape, rate_idx=0, q_value=1.0):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_interpolated_gain(z_hat, rate_idx, q_value, hyper=True)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat)
        
        if self.variable_bitrate:
            y_hat = self.compute_inverse_interpolated_gain(y_hat, rate_idx, q_value)

        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat = self.upsample(x_hat, 1//self.scale_factor)
        return x_hat
   
    def ae_compress(self, residual,  prev_latent, rate_idx=0,q_value=0.0,use_skip=False, skip_thresh=0.95, scale_factor=1.0,**kwargs):
        B,C,H,W = residual.shape
        if self.scale_factor != 1:
            residual = self.downsample(residual,self.scale_factor)

        y = self.g_a(residual)
        if prev_latent != None and use_skip:
            sim = torch.mean(self.similarity(prev_latent, y)).item()
        else:
            sim = 0
        if sim > skip_thresh:
            #skip this residual
            return None, True
        else:
            if self.variable_bitrate:
                y = torch.round(self.compute_interpolated_gain(y, rate_idx, q_value))
            return y, False
        
    def ae_decompress(self, y_hat, rate_idx=0, q_value=0.0, **kwargs):       
        if self.variable_bitrate:
            y_hat = self.compute_inverse_interpolated_gain(y_hat, rate_idx, q_value)

        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat = self.upsample(x_hat, 1//self.scale_factor)
        return x_hat
   
