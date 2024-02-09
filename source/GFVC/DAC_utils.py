import yaml
import numpy as np
import torch



from GFVC.DAC.sync_batchnorm import DataParallelWithCallback
from GFVC.DAC.modules.generator import GeneratorDAC ###
from GFVC.DAC.modules.keypoint_detector import KPD
from GFVC.DAC.animate import normalize_kp



def load_dac_checkpoints(config_path, checkpoint_path, device='cpu'):
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = GeneratorDAC(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPD(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True) ####

    # if device=='cuda':
    #     generator = DataParallelWithCallback(generator)
    #     kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator


# def make_DAC_prediction(reference_frame, kp_reference, kp_current, generator,relative=False, adapt_movement_scale=False, cpu=False):
        
#     kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
#                            kp_driving_initial=kp_reference, use_relative_movement=relative,
#                            use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
#     out = generator(reference_frame,kp_reference, kp_norm)
    
#     prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]
#     return prediction







