import matplotlib
matplotlib.use('Agg')
import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from GFVC.utils import *
from GFVC.CFTE_utils import *
from GFVC.CFTE.animate import normalize_kp
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict
from copy import copy

class CFTDecoder:
    '''Compact feature decoder'''
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

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        #List of floating point values representing the keypoints and jacobian matrices
        return kp_value_frame
        
    def decode_kp(self, frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+frame_index+'.bin' 
        bits=os.path.getsize(bin_file)*8              
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quantization
        kp_difference_dec=[i/self.q_step for i in kp_difference]
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer = listformat_adaptive_cfte(kp_previous, kp_difference_dec, 1, 4)
        self.rec_sem.append(kp_integer)
  
        cf_inter_frame={}
        cf_value=json.loads(kp_integer)
        cf_current_value=torch.Tensor(cf_value).to(device)          
        cf_inter_frame['value']=cf_current_value  
        return cf_inter_frame, bits
    
    def load_metadata(self)->None:
        bin_file=self.kp_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        self.ref_frame_idx = [int(i) for i in metadata]
        return os.path.getsize(bin_file)*8


class CFTE:
    def __init__(self, model_config_path:str, model_checkpoint_path:str, device:str='cpu') -> None:
        cfte_analysis_Model, cfte_synthesis_model = load_cfte_checkpoints(model_config_path, model_checkpoint_path, device=device)
        self.analysis_model = cfte_analysis_Model
        self.synthesis_model = cfte_synthesis_model

        self.reference_frame = None
        self.kp_reference = None
    
    def predict_inter_frame(self, kp_inter_frame: Dict[str,torch.Tensor], relative:bool =False,adapt_movement_scale:bool =False)->torch.Tensor:
        kp_norm = normalize_kp(kp_source=self.kp_reference, kp_driving=kp_inter_frame,
                            kp_driving_initial=self.kp_reference, use_relative_movement=relative,
                            use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        
        out = self.synthesis_model(self.reference_frame, self.kp_reference, kp_norm)
        # summary(generator, reference_frame, kp_reference, kp_norm)
        return out['prediction']


    
if __name__ == "__main__":
            
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
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
    gop_size = opt.gop_size
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    device = opt.device
    if not torch.cuda.is_available():
        device = 'cpu'
    ## CFTE
    model_name = 'CFTE' 
    model_config_path='./GFVC/CFTE/checkpoint/CFTE-256.yaml'
    model_checkpoint_path='./GFVC/CFTE/checkpoint/CFTE-checkpoint.pth.tar'         

    model_dirname='../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   
       
    ##################################################################
    ## Input paths [compact features and reference frame bitstreams]
    cf_input_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'  
    ref_input_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'/' 

    # Output paths for decoded sequences
    dec_output_path=model_dirname+'/dec/'
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    output_video_rgb =dec_output_path+seq+'_qp'+str(qp)+'.rgb'
    output_video_mp4 =dec_output_path+seq+'_qp'+str(qp)+'.mp4'

    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)         

    f_dec=open(output_video_rgb,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,qp,dec_name=opt.ref_codec)

    # Keypoint Decoder
    kp_decoder = CFTDecoder(cf_input_path,q_step)
    sum_bits += kp_decoder.load_metadata()
    #Reconstruction models
    dec_main = CFTE(model_config_path, model_checkpoint_path,device=device)
    
    out_video = []
    with torch.no_grad():
        for frame_idx in tqdm(range(0, frames)):            
            if frame_idx in kp_decoder.ref_frame_idx:      # I-frame                      
                reference, ref_bits = ref_decoder.decompress(frame_idx)  
                
                sum_bits+=ref_bits          
                if isinstance(reference, np.ndarray):
                    #convert to tensor
                    out_fr = reference
                    img_rec_out = np.transpose(reference,[2,0,1])
                    #convert the HxWx3 (uint8)-> 1x3xHxW (float32) 
                    reference = frame2tensor(img_rec_out)
                else:
                    #When using LIC for reference compression we get back a tensor 1x3xHXW
                    img_rec_out = tensor2frame(reference)
                    out_fr = np.transpose(img_rec_out,[1,2,0])
            
                out_video.append(out_fr)
                
                reference = reference.to(device)

                img_rec_out.tofile(f_dec)
                reference = frame2tensor(img_rec_out).to(device) #resize(img_rec, (3, height, width))    # normalize to 0-1  
                
                kp_reference = dec_main.analysis_model(reference) 

                #update decoder with the reference frame info
                dec_main.reference_frame = reference
                dec_main.kp_reference = kp_reference

                #append to list for use in predictively coding the next frame KPs
                #KPs used for relative decoding of inter_Frame KPs
                kp_value_frame = kp_decoder.get_kp_list(kp_reference, frame_idx)
                kp_decoder.rec_sem.append(kp_value_frame)
            else:
                # Decoding motion features
                dec_kp_inter_frame, kp_bits = kp_decoder.decode_kp(frame_idx)
                sum_bits+= kp_bits
                
                # Inter_frame reconstruction through animation
                gene_start = time.time()
                predicted_frame  = dec_main.predict_inter_frame(dec_kp_inter_frame)

                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(predicted_frame)
                #Save to output file
                pred.tofile(f_dec)             
                #HxWx3 format
                out_video.append(np.transpose(pred,[1,2,0]))  

    f_dec.close()     
    end=time.time()

    imageio.mimsave(output_video_mp4,out_video, fps=25.0)
    print(seq+'_qp'+str(qp)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_qp'+str(qp)+'.txt', totalResult, fmt = '%.5f')            


