import yaml
import numpy as np
import torch
from GFVC.FOMM.sync_batchnorm import DataParallelWithCallback
from GFVC.FOMM.modules.generator import OcclusionAwareGenerator ###
from GFVC.FOMM.modules.keypoint_detector import KPDetector


def load_fomm_checkpoints(config_path, checkpoint_path, device='cpu'):

    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict = True) ####

    if device=='cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator








