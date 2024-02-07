import matplotlib
matplotlib.use('Agg')
import os
import re
import yaml
import torch
import numpy as np
from typing import Dict, Any, List
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
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
    prediction = generator.animate(reference_frame, kp_norm,kp_reference) #reference_frame, kp_target, kp_reference
    prediction=np.transpose(prediction.data.cpu().numpy(), [0, 1, 2, 3])[0]
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


class KPCoder:
    def __init__(self, kp_output_dir:str, q_step:int=64) -> None:
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step

        #coding info
        self.rec_sem=[]

    def encode_kp(self, current_frame_kp: List[float], frame_idx: int)->None:
        ### residual
        kp_difference=(np.array(current_frame_kp)-np.array(self.rec_sem[-1])).tolist()
        ## quantization

        kp_difference=[i*self.q_step for i in kp_difference]
        kp_difference= list(map(round, kp_difference[:]))

        frame_idx = str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+str(frame_idx)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8
        # sum_bits += bits          

        #### decoding for residual
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        res_difference_dec=[i/self.q_step for i in res_difference_dec]

        rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits

    def encode_metadata(self, metadata: List[int])->None:
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(metadata,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits

    def compress_motion_keypoints(self,kp_frame: Dict[str, torch.Tensor], frame_idx_str, kp_output_dir)->str:
        #Extract the keypoints
        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        with open(self.kp_output_dir+'/frame'+frame_idx_str+'.txt','w')as f:
            f.write(kp_value_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame


class KPDecoder:
    def __init__(self, kp_output_dir:str, q_step:int=64, device='cpu') -> None:
        self.device= device
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step

        #coding info
        self.rec_sem=[]

    def decode_kp(self, frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_save=self.kp_output_dir+'/frame'+frame_index+'.bin'            
        kp_dec = final_decoder_expgolomb(bin_save)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quantization
        kp_difference_dec=[i/self.q_step for i in kp_difference]
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer,kp_value= listformat_kp_DAC(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_value=json.loads(kp_value)
        kp_target_value=torch.Tensor(kp_value).to(self.device)          
        kp_target_decoded = {'value': kp_target_value.reshape((1,10,2))  }
        return kp_target_decoded

    def load_metadata(self)->None:
        bin_file=self.kp_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        reference_frames = [int(i) for i in metadata]
        return reference_frames, os.path.getsize(bin_file)*8

