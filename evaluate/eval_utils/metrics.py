import os
import torch 
import inspect
import numpy as np 
from .utils import *
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from skimage import  img_as_float32
from scipy.signal import convolve2d
from typing import List,Dict, Any



class MetricParent:
    def __init__(self, bits=8, max_val=255, mvn=1, name=''):
        self.__name = name
        self.bits = bits
        self.max_val = max_val
        self.__metric_val_number = mvn
        self.metric_name = ''

    def set_bd_n_maxval(self, bitdepth=None, max_val=None):
        if bitdepth is not None:
            self.bits = bitdepth
        if max_val is not None:
            self.max_val = max_val

    def name(self):
        return self.__name

    def metric_val_number(self):
        return self.__metric_val_number

    def calc(self, orig, rec):
        raise NotImplementedError

class PSNR(MetricParent):
    '''Wrapper class computing PSNR-YUV
        Note: Currently returning PSNR-Y only
    '''
    def __init__(self, *args, **kwards):
        super().__init__(*args,
                         **kwards,
                         mvn=3,
                         name=['PSNR_Y', 'PSNR_U', 'PSNR_V'])

    def calc(self, org, dec, weight=None, _lambda=1.0):
        ans = []
        for plane in org:
            a = org[plane].mul((1 << self.bits) - 1)
            b = dec[plane].mul((1 << self.bits) - 1)
            sq_diff = (a-b)**2
            if weight is not None: #useful for computing ROI-weighted PSNR
                sq_diff = sq_diff*weight[:,:,0]
            mse = torch.mean(sq_diff)
            if mse == 0.0:
                ans.append(100)
            else:
                ans.append(20 * np.log10(self.max_val) - 10 * np.log10(mse))
        return float(ans[0])

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
  
class SSIM(MetricParent):
    '''A wrapper class for computing ssim'''
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (PyTorch)')
    
    def calc(self,org, dec, k1=0.01, k2=0.03, win_size=11, L=255):
        #compute only on the R channel
        im1 = org[0]
        im2 = dec[0]
        M, N = im1.shape
        C1 = (k1*L)**2
        C2 = (k2*L)**2
        window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
        window = window/np.sum(np.sum(window))
    
        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)
    
        mu1 = filter2(im1, window, 'valid')
        mu2 = filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
        sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    
        ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    
        return np.mean(np.mean(ssim_map))

class MS_SSIM(MetricParent):
    '''A wrapper class for computing msssim_pytorch'''
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (PyTorch)')
        from pytorch_msssim import ms_ssim
        self.metric = ms_ssim

    def calc(self, org, dec):
        ans = 0.0
        
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        plane = 'Y'
        
        a = org[plane].mul((1 << self.bits) - 1)
        b = dec[plane].mul((1 << self.bits) - 1)
        a.unsqueeze_(0).unsqueeze_(0)
        b.unsqueeze_(0).unsqueeze_(0)
        ans = self.metric(a, b, data_range=self.max_val).item()
        return ans
    
class FSIM_IQA(MetricParent):
    'Wrapper class for FSIM_IQA'
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='FSIM')
        from piq import fsim
        self.metric = fsim

    def calc(self, org: np.array, dec: np.array):  
        ans = 0.0
        # print(org.shape, dec.shape)
        org = torch.tensor(org).unsqueeze(0)
        dec = torch.tensor(dec).unsqueeze(0)
        ans = self.metric(org, dec).item()
        return ans
    
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    '''
    Refer to https://github.com/dingkeyan93/DISTS
    '''
    def __init__(self, channels=3, load_weights=True, device='cpu'):
        assert channels == 3
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(DISTS),'..','weights/DISTS.pt')), map_location=device)
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, as_loss=True, resize = True):
        assert x.shape == y.shape
        if resize:
            x, y = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if as_loss:
            return score.mean()
        else:
            return score

class LPIPSvgg(torch.nn.Module):
    def __init__(self, channels=3, device='cpu'):
        # Refer to https://github.com/richzhang/PerceptualSimilarity

        assert channels == 3
        super(LPIPSvgg, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]
        self.weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(LPIPSvgg),'..','weights/LPIPSvgg.pt')), map_location=device)
        self.weights = list(self.weights.items())
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        score = 0 
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1]*(feats0[k]-feats1[k])**2).mean([2,3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score

class LPIPS_IQA(MetricParent):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs, name='LPIPS')
        self.lpips = LPIPSvgg(device=device).to(device)
        self.device = device

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        org = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        ans = self.lpips(org, dec).item()  
        return ans
    
class DISTS_IQA(MetricParent):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs, name='LPIPS')
        self.dists = DISTS(device=device).to(device)
        self.device = device

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        org = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        ans = self.dists(org, dec).item()  
        return ans


class VMAF_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='VMAF')
        import platform
        if platform.system() == 'Linux':
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.linux')
        else:
            # TODO: check that
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf.exe'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.exe')

    def download(self, url, output_path):
        import requests
        r = requests.get(url, stream=True)  # , verify=False)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)

    def check(self):
        if not os.path.exists(self.OUTPUT_NAME):
            import stat
            self.download(self.URL, self.OUTPUT_NAME)
            os.chmod(self.OUTPUT_NAME, stat.S_IEXEC)

    def calc(self, org: np.ndarray, dec: np.ndarray) -> float:

        import subprocess
        import tempfile
        fp_o = tempfile.NamedTemporaryFile(delete=False)
        fp_r = tempfile.NamedTemporaryFile(delete=False)

        write_yuv(org, fp_o, self.bits)
        write_yuv(dec, fp_r, self.bits)

        out_f = tempfile.NamedTemporaryFile(delete=False)
        out_f.close()

        self.check()

        args = [
            self.OUTPUT_NAME, '-r', fp_o.name, '-d', fp_r.name, '-w',
            str(org['Y'].shape[1]), '-h',
            str(org['Y'].shape[0]), '-p', '420', '-b',
            str(self.bits), '-o', out_f.name, '--json'
        ]
        subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        import json
        with open(out_f.name, 'r') as f:
            tmp = json.load(f)
        ans = tmp['frames'][0]['metrics']['vmaf']

        os.unlink(fp_o.name)
        os.unlink(fp_r.name)
        os.unlink(out_f.name)

        return ans

import torch.nn.functional as F
from torchvision import models


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
  

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class MSVGG_IQA(MetricParent):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs, name='msVGG')
        self.loss_weights = [10, 10, 10, 10, 10]
        self.scales  = [1, 0.5, 0.25,0.125]
        self.wm_scales = [1, 0.5, 0.25, 0.125, 0.0625]
        self.device = device
        self.vgg = Vgg19().to(self.device)
        self.pyramid = ImagePyramide(self.scales, 3).to(self.device)

 
    def calc(self, org: np.array, dec: np.array)->float: 	
        org = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        
        pyramide_real = self.pyramid(org)
        pyramide_generated = self.pyramid(dec)
        value_total = 0.0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

            for i, _ in enumerate(self.loss_weights):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += value*self.loss_weights[i]
        return value_total.detach().cpu().item()


class MultiMetric:
    def __init__(self,metrics: List[str] = ['psnr','fsim','lpips','dists','ms_ssim','ssim'], device='cpu') -> None:
        
        self.metrics = metrics
        self.monitor = {}
        if 'psnr' in self.metrics:
            self.monitor.update({'psnr':PSNR()})
        
        if 'ssim' in self.metrics:
            self.monitor.update({'ssim':SSIM()})
        
        if 'ms_ssim' in self.metrics:
            self.monitor.update({'ms_ssim':MS_SSIM()})

        if 'fsim' in self.metrics:
            self.monitor.update({'fsim':FSIM_IQA()})

        if 'lpips' in self.metrics:
            self.monitor.update({'lpips':LPIPS_IQA(device=device)})

        if 'dists' in self.metrics:
            self.monitor.update({'dists':DISTS_IQA(device=device)})

        if 'msVGG' in self.metrics:
            self.monitor.update({'msVGG':MSVGG_IQA(device=device)})

        if 'vmaf' in self.metrics:
            self.monitor.update({'vmaf':VMAF_IQA()})


    def compute_metrics(self,org: np.ndarray, dec: np.ndarray)-> Dict[str,float]:
        output = {}

        #According to JPEG_AI, we should compute IQA metrics in YUV format
        org_yuv = load_image_array(org)
        dec_yuv = load_image_array(dec)

        # Perceptual metrics still require RGB images in float32 format (Also the current implementation of SSIM requires RGB input)
        org_rgb_float = img_as_float32(org)
        dec_rgb_float = img_as_float32(dec)

        for metric in self.monitor:
            if metric =='ssim':
                #use uint8 RGB input :: TO-DO (MAYBE THIS SHOULD BE MOVED TO YUV[0-1])
                val = self.monitor[metric].calc(org, dec) 
            elif metric in ['lpips','dists','fsim','msVGG']:
                #use RGB input as float 32
                val = self.monitor[metric].calc(org_rgb_float, dec_rgb_float) 
            else:
                #use YUV images as input
                val = self.monitor[metric].calc(org_yuv, dec_yuv) 
            output[metric] = val
        return output


