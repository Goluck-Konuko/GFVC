import os
import time
import torch
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.FOMM_utils import *
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List


class FOMMKPEncoder:
    def __init__(self,kp_output_dir:str, q_step:int=64):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.rec_sem = []
        self.ref_frame_idx = []

    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:

        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        kp_jacobian=kp_frame['jacobian'] 
        kp_jacobian_list=kp_jacobian.tolist()
        kp_jacobian_list=str(kp_jacobian_list)
        kp_jacobian_list="".join(kp_jacobian_list.split()) 
    
        with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
            f.write(kp_value_list)  
            f.write('\n'+kp_jacobian_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
        kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
        #List of floating point values representing the keypoints and jacobian matrices
        kp_list=kp_value_frame+kp_jacobian_frame ###20+40
        return kp_list
    
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
  
class FOMM:
    '''FOMM Encoder'''
    def __init__(self, fomm_config_path:str, fomm_checkpoint_path:str, device='cpu') -> None:
        self.device = device
        fomm_analysis_model, fomm_synthesis_model = load_fomm_checkpoints(fomm_config_path, fomm_checkpoint_path, device=device)
        self.analysis_model = fomm_analysis_model.to(device)
        self.synthesis_model = fomm_synthesis_model.to(device)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    q_step= opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    gop_size = opt.gop_size
    device = opt.device
    
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    
    model_name = 'FOMM' 
    model_dirname=f'../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   
    
    

    ###################
    start=time.time()
    kp_output_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'    #OutPut directory for motion keypoints
    os.makedirs(kp_output_path,exist_ok=True)                   

    enc_output_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'/' ## OutPut path for encoded reference frame     
    os.makedirs(enc_output_path,exist_ok=True)                      

    listR,listG,listB=raw_reader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    seq_kp_integer = []
    sum_bits = 0

    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name='vtm')
    #Motion keypoints coding
    kp_coder = FOMMKPEncoder(kp_output_path,q_step)


    #Main encoder models wrapper
    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    enc_main = FOMM(model_config_path, model_checkpoint_path,device=device)

    
    with torch.no_grad():
        for frame_idx in tqdm(range(0, frames)):            
            current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
            if frame_idx % gop_size == 0:      # I-frame      

                reference, ref_bits = ref_coder.compress(current_frame, frame_idx)   
                sum_bits+=ref_bits          
                if isinstance(reference, np.ndarray):
                    #convert to tensor
                    reference = torch.tensor(reference[np.newaxis].astype(np.float32))
                reference = reference.to(device) 

                #Extract motion representation vectors [Keypoints | Compact features etc]
                kp_reference =  enc_main.analysis_model(reference) ################ 
                kp_value_frame = kp_coder.get_kp_list(kp_reference, frame_idx)
                #append to list for use in predictively coding the next frame KPs
                kp_coder.rec_sem.append(kp_value_frame)
                kp_coder.ref_frame_idx.append(frame_idx) #metadata for reference frame indices
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






