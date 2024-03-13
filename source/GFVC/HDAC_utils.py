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
from GFVC.HDAC.modules.generator import GeneratorHDAC
from GFVC.HDAC.modules.keypoint_detector import KPD
from copy import copy



def load_hdac_checkpoints(config_path, checkpoint_path, device='cpu'):
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = GeneratorHDAC(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPD(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'],strict=True)
        kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True) ####
    
    generator.to(device).eval()
    kp_detector.to(device).eval()
    return kp_detector,generator

class KPEncoder:
    def __init__(self,kp_output_dir:str, q_step:int=64,num_kp=10, device='cpu'):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.device = device
        self.num_kp = num_kp
        self.rec_sem = []
        self.ref_frame_idx = []

    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:
        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
            f.write(kp_value_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame
    
    def encode_kp(self,kp_list:List[str], frame_idx:int):
        ### residual
        kp_difference=(np.array(kp_list)-np.array(self.rec_sem[-1])).tolist()
        ## quantization

        kp_difference=[i*self.q_step for i in kp_difference]
        kp_difference= list(map(round, kp_difference[:]))

        bin_file=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        kp_dec = final_decoder_expgolomb(bin_file)
        kp_difference_dec = data_convert_inverse_expgolomb(kp_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        kp_difference_dec=[i/self.q_step for i in kp_difference_dec]

        kp_integer,kp_value=listformat_kp_DAC(self.rec_sem[-1], kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).reshape((1,self.num_kp,2)).to(self.device)          
        kp_inter_frame['value']=kp_current_value  
        #reconstruct the KPs 
        return kp_inter_frame, bits
    
    def encode_metadata(self)->None:
        '''this can be optimized to use run-length encoding which would be more efficient'''
        data = copy(self.ref_frame_idx)
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits
  
class KPDecoder:
    def __init__(self, kp_output_dir:str, q_step:int=64,num_kp=10, device='cpu') -> None:
        self.device= device
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.num_kp = num_kp

        #coding info
        self.rec_sem=[]
        self.ref_frame_idx = []

    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:
        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
            f.write(kp_value_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame

    
    def decode_kp(self,frame_idx:int):
        frame_idx=str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+frame_idx+'.bin'
        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        kp_dec = final_decoder_expgolomb(bin_file)
        kp_difference_dec = data_convert_inverse_expgolomb(kp_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        kp_difference_dec=[i/self.q_step for i in kp_difference_dec]

        kp_integer,kp_value=listformat_kp_DAC(self.rec_sem[-1], kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).reshape((1,self.num_kp,2)).to(self.device)          
        kp_inter_frame['value']=kp_current_value  
        #reconstruct the KPs 
        return kp_inter_frame, bits
    

    def load_metadata(self)->None:
        bin_file=self.kp_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        self.ref_frame_idx = [int(i) for i in metadata]
        return os.path.getsize(bin_file)*8

