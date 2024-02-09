import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path

from GFVC.CFTE.sync_batchnorm import DataParallelWithCallback
from GFVC.CFTE.modules.util import *
from GFVC.CFTE.modules.generator import OcclusionAwareGenerator ###
from GFVC.CFTE.modules.keypoint_detector import KPDetector ###

def load_cfte_checkpoints(config_path, checkpoint_path, device='cpu'):
    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True) ####

    if device =='cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator




