import matplotlib
matplotlib.use('Agg')
import yaml
import torch
import numpy as np
from GFVC.RDAC.animate import normalize_kp
from GFVC.RDAC.modules.generator import GeneratorRDAC ###
from GFVC.RDAC.modules.keypoint_detector import KPDetector




def load_RDAC_checkpoints(config_path, checkpoint_path, cpu=False):
    if cpu:
        device = 'cpu'
    else:
        device = 'cuda'

    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = GeneratorRDAC(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=False)
    #Update the pdf tables for SDC and TDC
    generator.sdc.update(force=True)
    generator.tdc.update(force=True)

    kp_detector.load_state_dict(checkpoint['kp_detector']) ####

    if device=='cuda':
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()
    
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator


def make_RDAC_prediction(reference_frame, kp_reference, kp_current, generator,relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    #Generate animation and compress the spatial and temporal residual
    anim_frame = generator(reference_frame,kp_reference, kp_norm)
    prediction=np.transpose(anim_frame['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]
    return prediction







