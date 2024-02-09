import os
import time
import json
import torch
import imageio
import numpy as np
from GFVC.utils import *
from GFVC.DAC_utils import *
from GFVC.DAC.animate import normalize_kp
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict, List
from copy import copy


class DACKPDecoder:
    def __init__(self,kp_output_dir:str, q_step:int=64):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.rec_sem = []

    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:

        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
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

        kp_integer,kp_value=listformat_kp_DAC(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).reshape((1,10,2)).to(device)          
        kp_inter_frame['value']=kp_current_value  
        return kp_inter_frame, bits
    
    def decode_metadata(self, metadata: List[int])->None:
        '''this can be optimized to use run-length encoding which would be more efficient'''
        data = copy(metadata)
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits
  
class DAC:
    '''DAC Decoder models'''
    def __init__(self, dac_config_path:str, dac_checkpoint_path:str, device='cpu') -> None:
        self.device = device
        dac_analysis_model, dac_synthesis_model = load_dac_checkpoints(dac_config_path, dac_checkpoint_path, device=device)
        self.analysis_model = dac_analysis_model.to(device)
        self.synthesis_model = dac_synthesis_model.to(device)

        self.reference_frame = None
        self.kp_reference = None

    def generate_frame(self,kp_inter_frame:Dict[str, torch.Tensor], relative=False,adapt_movement_scale=False)->torch.Tensor:
        prediction = self.synthesis_model.animate(self.reference_frame,self.kp_reference, kp_inter_frame)
        return prediction



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--adaptive_metric", default='PSNR', type=str,help="RD adaptation metric (for selecting reference frames to keep in buffer)")
    parser.add_argument("--adaptive_thresh", default=30, type=float,help="Reference selection threshold")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    q_step=opt.quantization_factor
    qp = opt.iframe_qp
    iframe_format=opt.iframe_format
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    device = opt.device
    gop_size = opt.gop_size
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    
    ## FOM
    model_name = "DAC"

    model_config_path=f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'          
    model_dirname=f'../experiment/{model_name}/Iframe_{iframe_format}'  
    
###################################################

    kp_input_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'   

    ref_input_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'/'
    os.makedirs(ref_input_path,exist_ok=True)     # the frames to be compressed by vtm    

    dec_output_path = model_dirname+'/dec/'
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    dec_sequence_path_rgb  =dec_output_path+seq+'_qp'+str(qp)+'.rgb'
    dec_sequence_path_mp4 =dec_output_path+seq+'_qp'+str(qp)+'.mp4'


    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)        


    #Initialize the output file 
    f_dec=open(dec_sequence_path_rgb,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,qp)

    # Keypoint Decoder
    kp_decoder = DACKPDecoder(kp_input_path,q_step)

    #Reconstruction models
    dec_main = DAC(model_config_path, model_checkpoint_path,device=device)
    
    output_video = [] #Output an mp4 video for sanity check
    with torch.no_grad():
        for frame_idx in tqdm(range(0, frames)):            
            if frame_idx%gop_size == 0:      # I-frame                      
                img_rec, ref_bits = ref_decoder.decompress(frame_idx) 
                sum_bits+= ref_bits
                #convert and save the decoded reference frame
                output_video.append(img_rec[:,:,::-1])
                img_rec_out = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
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
                predicted_frame  = dec_main.generate_frame(dec_kp_inter_frame)

                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(predicted_frame)
                #Save to output file
                pred.tofile(f_dec)             
                #HxWx3 format
                output_video.append(np.transpose(pred,[1,2,0]))                 

    f_dec.close()     
    end=time.time()
    imageio.mimsave(dec_sequence_path_mp4,output_video, fps=25.0)
    print(seq+'_qp'+str(qp)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_qp'+str(qp)+'.txt', totalResult, fmt = '%.5f')            


