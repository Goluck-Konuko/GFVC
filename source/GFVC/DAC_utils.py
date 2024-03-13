import yaml
import numpy as np
import torch



from GFVC.DAC.sync_batchnorm import DataParallelWithCallback
from GFVC.DAC.modules.generator import GeneratorDAC ###
from GFVC.DAC.modules.keypoint_detector import KPD
from GFVC.DAC.animate import normalize_kp



def load_dac_checkpoints(config_path, checkpoint_path,num_kp=10, device='cpu'):
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config['model_params']['common_params']['num_kp'] = num_kp

    generator = GeneratorDAC(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPD(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True) ####

    generator.eval()
    kp_detector.eval()
    return kp_detector,generator







