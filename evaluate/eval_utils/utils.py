import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List

def abs(x):
    return torch.sqrt(x[:,:,:,:,0]**2+x[:,:,:,:,1]**2+1e-12)

def real(x):
    return x[:,:,:,:,0]

def imag(x):
    return x[:,:,:,:,1]

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def preprocess_lab(lab):
		L_chan, a_chan, b_chan =torch.unbind(lab,dim=2)
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]

def deprocess_lab(L_chan, a_chan, b_chan):
		#TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
			   # ( we process individual images but deprocess batches)
		#return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
		return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)

def rgb_to_lab(srgb):
    srgb = srgb/255
    srgb_pixels = torch.reshape(srgb, [-1, 3])
    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
    rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor).to(device)
	
    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).to(device))

    epsilon = 6.0/29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ]).type(torch.FloatTensor).to(device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device)
    #return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)

def lab_to_rgb(lab, device='cpu'):
		lab_pixels = torch.reshape(lab, [-1, 3])
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor).to(device)
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device))


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor).to(device)

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		return torch.reshape(srgb_pixels, lab.shape)

def spatial_normalize(x):
    min_v = torch.min(x.view(x.shape[0],1,-1),dim=2)[0]
    range_v = torch.max(x.view(x.shape[0],1,-1),dim=2)[0] - min_v
    return (x - min_v.unsqueeze(2).unsqueeze(3)) / (range_v.unsqueeze(2).unsqueeze(3)+1e-12)

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)



def downsample(img1, img2, maxSize = 256):
    _,channels,H,W = img1.shape
    #channels,H,W = img1.shape
    f = int(max(1,np.round(min(H,W)/maxSize)))
    if f>1:
        aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    return img1, img2

def extract_patches_2d(img, patch_shape=[64, 64], step=[27,27], batch_first=True, keep_last_patch=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0) and keep_last_patch:
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0) and keep_last_patch:
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches.reshape(-1,3,patch_H,patch_W)

def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



data_range = [0, 1]

def load_image_array(rgb_data,color_conv='709',def_bits=8, device='cpu'):
    rgb_data = torch.tensor(rgb_data,dtype=torch.float,device=device)
    rgb_data = convert_and_round_plane(rgb_data, [0, 255], data_range,def_bits).unsqueeze(0)
    yuv_t = rgb_to_yuv(rgb_data, color_conv).clamp(min(data_range),max(data_range))
    yuv_t = round_plane(yuv_t, def_bits)
    yuv_data = {
                'Y': yuv_t[0, 0],
                'U': yuv_t[0, 1],
                'V': yuv_t[0, 2]
                }
    return yuv_data

def convert_and_round_plane(plane, cur_range, new_range, bits):
        return round_plane(convert_range(plane, cur_range, new_range), bits)

def convertup_and_round_plane(plane, cur_range, new_range, bits):
    return convert_range(plane, cur_range, new_range).mul((1 << bits) - 1).round()

def round_plane(plane, bits):
        return plane.mul((1 << bits) - 1).round().div((1 << bits) - 1)

def convert_range(plane, cur_range, new_range=[0, 1]):
    if cur_range[0] == new_range[0] and cur_range[1] == new_range[1]:
        return plane
    return (plane + cur_range[0]) * (new_range[1] - new_range[0]) / (
        cur_range[1] - cur_range[0]) - new_range[0]

def rgb_to_yuv(image: torch.Tensor, color_conv='709') -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            'Input size must have a shape of (*, 3, H, W). Got {}'.format(
                image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    a1, b1, c1, d1, e1 = color_conv_matrix(color_conv)

    y: torch.Tensor = a1 * r + b1 * g + c1 * b
    u: torch.Tensor = (b - y) / d1 + 0.5
    v: torch.Tensor = (r - y) / e1 + 0.5

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out

def color_conv_matrix(color_conv='709'):
    if color_conv == '601':
        # BT.601
        a = 0.299
        b = 0.587
        c = 0.114
        d = 1.772
        e = 1.402
    elif color_conv == '709':
        # BT.709
        a = 0.2126
        b = 0.7152
        c = 0.0722
        d = 1.8556
        e = 1.5748
    elif color_conv == '2020':
        # BT.2020
        a = 0.2627
        b = 0.6780
        c = 0.0593
        d = 1.8814
        e = 1.4747
    else:
        raise NotImplementedError

    return a, b, c, d, e


def write_yuv(yuv: Dict[str, torch.Tensor], f: str, bits: int=8):
    """
    dump a yuv file to the provided path
    @path: path to dump yuv to (file must exist)
    @bits: bitdepth
    @frame_idx: at which idx to write the frame (replace), -1 to append
    """
    nr_bytes = np.ceil(bits / 8)
    if nr_bytes == 1:
        data_type = np.uint8
    elif nr_bytes == 2:
        data_type = np.uint16
    elif nr_bytes <= 4:
        data_type = np.uint32
    else:
        raise NotImplementedError(
            'Writing more than 16-bits is currently not supported!')

    # rescale to range of bits
    for plane in yuv:
        yuv[plane] = convertup_and_round_plane(yuv[plane], data_range, data_range,bits).cpu().numpy()

    # dump to file
    lst = []
    for plane in ['Y', 'U', 'V']:
        if plane in yuv.keys():
            lst = lst + yuv[plane].ravel().tolist()

    raw = np.array(lst)

    raw.astype(data_type).tofile(f)



import yaml
def read_config_file(config_path):
    '''Simply reads the configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
