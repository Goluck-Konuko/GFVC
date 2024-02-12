import os
import time
import json
import torch
import imageio
import numpy as np
from GFVC.utils import *
from GFVC.RDAC_utils import *
from argparse import ArgumentParser
from GFVC.RDAC.entropy_coders.residual_entropy_coder import ResEntropyDecoder
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict, List
from copy import copy


# class KPDecoder:
#     def __init__(self,kp_output_dir:str, q_step:int=64):
#         self.kp_output_dir = kp_output_dir
#         self.q_step = q_step
#         self.rec_sem = []
#         self.ref_frame_idx = []

#     def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int)->List[str]:
#         kp_value = kp_frame['value']
#         kp_value_list = kp_value.tolist()
#         kp_value_list = str(kp_value_list)
#         kp_value_list = "".join(kp_value_list.split())

#         kp_value_frame=json.loads(kp_value_list)###20
#         kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
#         return kp_value_frame
        
#     def decode_kp(self, frame_idx: int)->None:
#         frame_index=str(frame_idx).zfill(4)
#         bin_file=self.kp_output_dir+'/frame'+frame_index+'.bin' 
#         bits=os.path.getsize(bin_file)*8              
#         kp_dec = final_decoder_expgolomb(bin_file)

#         ## decoding residual
#         kp_difference = data_convert_inverse_expgolomb(kp_dec)
#         ## inverse quantization
#         kp_difference_dec=[i/self.q_step for i in kp_difference]
#         kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
#         kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

#         kp_integer,kp_value=listformat_kp_DAC(kp_previous, kp_difference_dec) #######
#         self.rec_sem.append(kp_integer)
  
#         kp_inter_frame={}
#         kp_value=json.loads(kp_value)
#         kp_current_value=torch.Tensor(kp_value).reshape((1,10,2)).to(device)          
#         kp_inter_frame['value']=kp_current_value  
#         return kp_inter_frame, bits
    
#     def decode_metadata(self, metadata: List[int])->None:
#         '''this can be optimized to use run-length encoding which would be more efficient'''
#         data = copy(metadata)
#         bin_file=self.kp_output_dir+'/metadata.bin'
#         final_encoder_expgolomb(data,bin_file)     

#         bits=os.path.getsize(bin_file)*8
#         return bits
  
class RDAC:
    '''DAC Decoder models'''
    def __init__(self, rdac_config_path:str, rdac_checkpoint_path:str,res_input_path:str,
                 res_coding_params:Dict[str,Any], device='cpu') -> None:
        self.device = device
        rdac_analysis_model, rdac_synthesis_model = load_rdac_checkpoints(rdac_config_path, rdac_checkpoint_path, device=device)
        self.analysis_model = rdac_analysis_model.to(device)
        self.synthesis_model = rdac_synthesis_model.to(device)

        #Residual coder
        self.res_inp_path = res_input_path
        self.res_decoder = ResEntropyDecoder(input_path=self.res_inp_path)
        self.res_coding_params = res_coding_params

        #Coding buffer
        self.reference_frame = None
        self.kp_reference = None
        self.prev_animation = None
        self.prev_res_hat = None
        self.prev_latent = None
        self.prev_kp_target = None

        #Metadata - List of flags indicating if the frame residual is coded or not
        #For now it's just a list of 0s (frame residual skipper) and 1s (frame residual coded)
        self.res_coding_flag = []
        self.ref_frame_idx = []

    def predict_inter_frame(self,kp_inter_frame:Dict[str, torch.Tensor], relative=False,adapt_movement_scale=False)->torch.Tensor:
        anim_prediction = self.synthesis_model.animate(self.reference_frame,self.kp_reference, kp_inter_frame)
        res_bits = 0
        if frame_idx-1 in self.ref_frame_idx:
            #Decode without temporal residual enhancement
            dec_info, res_bits = read_bitstring(self.res_inp_path, frame_idx)
            res_hat,_ = self.synthesis_model.sdc.rans_decompress(dec_info['strings'],dec_info['shape'], **self.res_coding_params)
            self.prev_res_hat = res_hat
        else:
            deform_params = {'prev_rec': self.prev_animation,
                             'kp_prev': self.prev_kp_target,
                             'kp_cur': kp_inter_frame,
                             'res_hat_prev': self.prev_res_hat}
            def_prev_res_hat = self.synthesis_model.deform_residual(deform_params)

            if self.res_coding_flag[frame_idx-1]==1:
                #Decode and enhance with previously decoded residual
                dec_info, res_bits = read_bitstring(self.res_inp_path, frame_idx)
                res_hat_temp,_ = self.synthesis_model.tdc.rans_decompress(dec_info['strings'],dec_info['shape'], **self.res_coding_params)
                res_hat = def_prev_res_hat+res_hat_temp*2.0
                self.prev_res_hat = res_hat
            else:
                res_hat = def_prev_res_hat
        prediction = (anim_prediction+res_hat).clamp(0,1)
        self.prev_animation = prediction
        self.prev_kp_target = kp_inter_frame
        # prediction = np.transpose(anim_prediction.data.cpu().numpy(), [0, 1, 2, 3])[0]
        return prediction,anim_prediction, res_bits



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
    parser.add_argument("--rate_idx", default=0, type=int,help="The RD point for coding the frame residuals [1-5]")
    parser.add_argument("--int_value", default=0.0, type=float,help="Interpolation value between RD points for residual coding [0.0 - 1.0]")
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
    rate_idx = opt.rate_idx
    int_value = opt.int_value
    if not torch.cuda.is_available():
        device = 'cpu'
    
    ## FOM
    model_name = "RDAC"

    model_config_path=f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'          
    model_dirname=f'../experiment/{model_name}/Iframe_{iframe_format}'  
    
    ###################################################
    #Input directories created by the encoder
    kp_input_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/'   
    res_input_path =model_dirname+'/res/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/' 
    ref_input_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/'

    #Output directory
    dec_output_path = model_dirname+'/dec/'
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    dec_sequence_path_rgb  =dec_output_path+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'.rgb'
    dec_sequence_path_mp4 =dec_output_path+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'.mp4'


    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)        


    #Initialize the output file 
    f_dec=open(dec_sequence_path_rgb,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,qp, device=device)

    # Keypoint Decoder
    kp_decoder = KPDecoder(kp_input_path,q_step,device=device)
    sum_bits += kp_decoder.load_metadata()


    #Reconstruction models
    res_coding_params = {'rate_idx':rate_idx,'q_value':int_value}
    dec_main = RDAC(model_config_path, model_checkpoint_path,res_input_path,res_coding_params,device=device)
    res_flags, res_mt_bits = dec_main.res_decoder.load_metadata()
    dec_main.res_coding_flag = res_flags
    dec_main.ref_frame_idx = kp_decoder.ref_frame_idx
    sum_bits += res_mt_bits

    output_video = [] #Output an mp4 video for sanity check
    with torch.no_grad():
        for frame_idx in tqdm(range(0, frames)):            
            if frame_idx%gop_size == 0:      # I-frame                      
                img_rec, ref_bits = ref_decoder.decompress(frame_idx) 
                sum_bits+= ref_bits
                #convert and save the decoded reference frame
                out_frame = np.concatenate((img_rec[:,:,::-1],img_rec[:,:,::-1]), axis=1)
                output_video.append(out_frame)

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
                pred, anim_pred,res_bits  = dec_main.predict_inter_frame(dec_kp_inter_frame)
                sum_bits += res_bits
                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(pred)
                anim_pred = tensor2frame(anim_pred)
                #Save to output file
                pred.tofile(f_dec)             
                #HxWx3 format
                out_frame = np.concatenate((np.transpose(anim_pred,[1,2,0]),np.transpose(pred,[1,2,0])),axis=1)
                output_video.append(out_frame)                 

    f_dec.close()     
    end=time.time()
    imageio.mimsave(dec_sequence_path_mp4,output_video, fps=25.0)
    print(seq+'_qp'+str(qp)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'.txt', totalResult, fmt = '%.5f')            


