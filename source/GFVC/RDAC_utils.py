import matplotlib
matplotlib.use('Agg')
import os
import re
import yaml
import torch
import numpy as np
from typing import Dict, Any
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




def write_bitstring(enc_info:Dict[str,Any], dec_dir:str, frame_idx:int)->float:
    strings = enc_info['strings']
    shape = f"{enc_info['shape'][0]}_{enc_info['shape'][1]}_"
    out_path = dec_dir+'/'+shape+str(frame_idx)
    y_string = strings[0][0]
    z_string = strings[1][0]

    #write both strings to binary file
    with open(f"{out_path}_y.bin", 'wb') as y:
        y.write(y_string)
    bits = os.path.getsize(f"{out_path}_y.bin")*8
    with open(f"{out_path}_z.bin", 'wb') as z:
        z.write(z_string)
    bits += os.path.getsize(f"{out_path}_z.bin")*8
    return bits


def read_bitstring(dec_dir:str, frame_idx:int):
    #locate the correct files in the dec_dir
    #The process is clumsy but should work for now
    bin_files = [x for x in os.listdir(dec_dir) if x.endswith('.bin')]
    y_pattern = re.compile(r'_{}_y\.bin'.format(frame_idx))
    z_pattern = re.compile(r'_{}_z\.bin'.format(frame_idx))
    y_file = [file for file in bin_files if y_pattern.search(file)][-1]
    z_file = [file for file in bin_files if z_pattern.search(file)][-1]

    rec_shape = y_file.split("_")
    shape = [int(rec_shape[0]), int(rec_shape[0])]
    with open(f"{dec_dir}/{y_file}", 'rb') as y_out:
        y_string = y_out.read()
    bits = os.path.getsize(f"{dec_dir}/{y_file}")*8

    with open(f"{dec_dir}/{z_file}", 'rb') as z_out:
        z_string = z_out.read()
    bits += os.path.getsize(f"{dec_dir}/{z_file}")*8
    dec_info = {'strings': [[y_string],[z_string]], 'shape':shape}
    return dec_info, bits



