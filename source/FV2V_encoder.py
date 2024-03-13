import matplotlib
matplotlib.use('Agg')
import os
import cv2
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from GFVC.utils import *
from GFVC.FV2V_utils import *
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict, List 
from copy import copy


class FV2VKPEncoder:
    def __init__(self,kp_output_dir:str, q_step:int=64):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.rec_sem = []
        self.ref_frame_idx = []

    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:

        yaw=kp_frame['yaw']
        pitch=kp_frame['pitch']
        roll=kp_frame['roll']

        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        rot_mat = get_rotation_matrix(yaw, pitch, roll)            
        rot_mat_list=rot_mat.tolist()
        rot_mat_list=str(rot_mat_list)
        rot_mat_list="".join(rot_mat_list.split())               

        t=kp_frame['t']
        t_list=t.tolist()
        t_list=str(t_list)
        t_list="".join(t_list.split())

        exp=kp_frame['exp']
        exp_list=exp.tolist()
        exp_list=str(exp_list)
        exp_list="".join(exp_list.split())   


        with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
            f.write(rot_mat_list)
            f.write('\n'+t_list)  
            f.write('\n'+exp_list)

        rot_frame=json.loads(rot_mat_list)###torch.Size([1, 3, 3])
        rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))

        t_frame=json.loads(t_list)  ###torch.Size([1, 3])
        t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))

        exp_frame=json.loads(exp_list)  ###torch.Size([1, 45])
        exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))

        kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
        return kp_integer  
    
    def encode_kp(self,kp_list:List[str], frame_idx:int):
        ### residual
        kp_difference=(np.array(kp_list)-np.array(self.rec_sem[frame_idx-1])).tolist()
        ## quantization

        kp_difference=[i*self.q_step for i in kp_difference]
        kp_difference= list(map(round, kp_difference[:]))

        bin_file=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        res_difference_dec=[i/self.q_step for i in res_difference_dec]

        rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[frame_idx-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits
    
    def encode_metadata(self)->None:
        '''this can be optimized to use run-length encoding which would be more efficient'''
        data = copy(self.ref_frame_idx)
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits
  

class FV2V:
    '''Wrapper for models and forward methods'''
    def __init__(self, model_config_path:str, model_checkpoint_path:str, device:str='cpu') -> None:
        fv2v_analysis_model_detector, fv2v_analysis_model_estimator, fv2v_synthesis_model = load_fv2v_checkpoints(model_config_path, model_checkpoint_path, device=device)
        self.detector_model = fv2v_analysis_model_detector
        self.estimator_model = fv2v_analysis_model_estimator
        self.synthesis_model = fv2v_synthesis_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=256, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--ref_codec", default='vtm', type=str,help="Reference frame codec [vtm | lic]")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    q_step=opt.quantization_factor
    qp=opt.iframe_qp
    iframe_format=opt.iframe_format    
    original_seq=opt.original_seq
    gop_size = opt.gop_size
    device = opt.device
    if not torch.cuda.is_available():
        device = 'cpu'
        
    
    
    ## FV2V
    model_name = 'FV2V'
    model_config_path=f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    model_dirname='../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   
            
    ##########################
    start=time.time()

    f_org=open(original_seq,'rb')

    listR,listG,listB=raw_reader_planar(original_seq,width, height,frames)

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    #Output paths
    enc_output_path =model_dirname+'/enc/'+'/'+seq+'_qp'+str(qp)+'/'
    os.makedirs(enc_output_path,exist_ok=True)     # the frames to be compressed by vtm      

    kp_output_path=model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'
    os.makedirs(kp_output_path,exist_ok=True)     # the frames to be compressed by vtm   

    start=time.time() 

    sum_bits = 0            
    seq_kp_integer = []

    #Initialize models and entropy coders
    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name=opt.ref_codec)

    # motion entropy coder
    kp_coder = FV2VKPEncoder(kp_output_path,q_step=q_step)
    #analysis and synthesis models
    enc_main = FV2V(model_config_path, model_checkpoint_path, device=device)

    for frame_idx in tqdm(range(0, frames)):         
        current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]   
        if frame_idx%gop_size == 0:      # I-frame      
            reference, ref_bits = ref_coder.compress(current_frame, frame_idx)  
            sum_bits+=ref_bits          
            if isinstance(reference, np.ndarray):
                reference = frame2tensor(np.transpose(reference,[2,0,1]))
            reference = reference.to(device)

            head_pose_info = enc_main.estimator_model(reference)
            kp_list_frame = kp_coder.get_kp_list(head_pose_info, frame_idx)
            kp_coder.rec_sem.append(kp_list_frame)
            kp_coder.ref_frame_idx.append(frame_idx) #metadata for reference frame indices
        
        else:

            inter_frame = cv2.merge(current_frame)       
            inter_frame = resize(inter_frame, (width, height))[..., :3]

            inter_frame = torch.tensor(inter_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            inter_frame = inter_frame.to(device)    # require GPU      


            head_pose_info = enc_main.estimator_model(inter_frame)
            #convert pose information to list and compress relative to the reference pose information
            kp_list_frame = kp_coder.get_kp_list(head_pose_info, frame_idx)
            rec_head_pose_info, kp_bits = kp_coder.encode_kp(kp_list_frame, frame_idx) 
            sum_bits += kp_bits
    
    sum_bits+= kp_coder.encode_metadata()       
    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   




