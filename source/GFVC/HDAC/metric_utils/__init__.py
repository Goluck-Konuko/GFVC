from .metrics import *

eval_metrics = {
    'psnr': PSNR_IQA,
    'lpips': LPIPS_IQA,
    'dists': DISTS_IQA,
    'ms_ssim': MS_SSIM_IQA,
    'ssim': SSIM_IQA,
    'fsim': FSIM_IQA
}