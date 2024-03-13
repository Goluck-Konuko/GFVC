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
from GFVC.CFTE_utils import *
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from GFVC.CFTE.animate import normalize_kp
from typing import Dict
from copy import copy

class CFTEncoder:
    '''Compact feature encoder'''
    def __init__(self,kp_output_dir:str, q_step:int=64, device='cpu'):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.device = device
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
        kp_difference=(np.array(kp_list)-np.array(self.rec_sem[frame_idx-1])).tolist()
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

        kp_integer,kp_value=listformat_kp_DAC(self.rec_sem[frame_idx-1], kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).to(self.device)          
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
  

class CFTE:
    def __init__(self, model_config_path:str, model_checkpoint_path:str, device:str='cpu') -> None:
        cfte_analysis_Model, cfte_synthesis_model = load_cfte_checkpoints(model_config_path, model_checkpoint_path, device=device)
        self.analysis_model = cfte_analysis_Model
        self.synthesis_model = cfte_synthesis_model
    
    
if __name__ == "__main__":
   
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--ref_codec", default='vtm', type=str,help="Reference frame codec [vtm | lic]")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
        
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width = opt.seq_width #SourceWidth
    height = opt.seq_width #SourceHeight
    q_step = opt.quantization_factor #Keypoint quantization factor
    qp = opt.iframe_qp #Reference frame QP
    iframe_format=opt.iframe_format #Reference frame color format
    original_seq=opt.original_seq #path to video input video sequences
    gop_size = opt.gop_size
    device = opt.device
    if not torch.cuda.is_available():
        device='cpu'
    
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    
    model_name = 'CFTE' 
    model_dirname=f'../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   
    
    
    #############################
    listR,listG,listB=raw_reader_planar(original_seq, width, height,frames)

    kp_output_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'    
    os.makedirs(kp_output_path,exist_ok=True)     # the frames to be compressed by vtm                 

    enc_output_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'/'
    os.makedirs(enc_output_path,exist_ok=True)     # the frames to be compressed by vtm                 

    f_org=open(original_seq,'rb')


    #Initialize the codec with models
    model_config_path = f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path = f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar' 

    enc_main = CFTE(model_config_path, model_checkpoint_path, device=device)
    
    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name=opt.ref_codec, device=device)

    #Motion keypoints coding
    kp_coder = CFTEncoder(kp_output_path,q_step, device=device) #Compact feature encoder | similar to KP Encoder for DAC and FOMM


    seq_kp_integer=[]
    start=time.time() 

    sum_bits = 0
    out_video = []
    with torch.no_grad():
        for frame_idx in tqdm(range(0, frames)):       
            current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
            cur_fr = np.transpose(np.array(current_frame),[1,2,0])
            if frame_idx%gop_size == 0:      # I-frame      
                reference, ref_bits = ref_coder.compress(current_frame, frame_idx)  
                sum_bits+=ref_bits          
                if isinstance(reference, np.ndarray):
                    reference = frame2tensor(np.transpose(reference,[2,0,1]))
                reference = reference.to(device)

                #Extract motion representation vectors [Keypoints | Compact features etc]
                kp_reference =  enc_main.analysis_model(reference) ################ 
                kp_value_frame = kp_coder.get_kp_list(kp_reference, frame_idx)
                #append to list for use in predictively coding the next frame KPs
                kp_coder.rec_sem.append(kp_value_frame)
                kp_coder.ref_frame_idx.append(frame_idx) #metadata for reference frame indices
                
                #update enc main with reference frame info
                enc_main.reference_frame = reference
                enc_main.kp_reference = kp_reference
            else:
                inter_frame = cv2.merge(current_frame)
                inter_frame = resize(inter_frame, (width, height))[..., :3]

                inter_frame = torch.tensor(inter_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                inter_frame = inter_frame.to(device)    # require GPU | Use the available device      

                ###extracting motion representation
                kp_inter_frame = enc_main.analysis_model(inter_frame) ################
                kp_frame = kp_coder.get_kp_list(kp_inter_frame, frame_idx)
                #Encode to binary string
                rec_kp_frame, kp_bits = kp_coder.encode_kp(kp_frame, frame_idx)
                sum_bits += kp_bits  
    sum_bits+= kp_coder.encode_metadata()
    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   

