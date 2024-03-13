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


class FV2VKPDecoder:
    def __init__(self,kp_output_dir:str, q_step:int=64, device='cpu'):
        self.kp_output_dir = kp_output_dir
        self.device = device
        self.q_step = q_step
        self.rec_sem = []

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
    
    def decode_kp(self,kp_canonical, frame_idx: int)->None:
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

        kp_integer,kp_mat_value,kp_t_value,kp_exp_value=listformat_kp_mat_exp_fv2v(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer)


        rec_kp_current={}                  
        kp_mat_value=json.loads(kp_mat_value)
        kp_current_mat=torch.Tensor(kp_mat_value).to(self.device)          
        rec_kp_current['rot_mat']=kp_current_mat  

        kp_t_value=json.loads(kp_t_value)
        kp_current_t=torch.Tensor(kp_t_value).to(self.device)          
        rec_kp_current['t']=kp_current_t  

        kp_exp_value=json.loads(kp_exp_value)
        kp_current_exp=torch.Tensor(kp_exp_value).to(self.device)          
        rec_kp_current['exp']=kp_current_exp  

        kp_current = keypoint_transformation(kp_canonical, rec_kp_current, estimate_jacobian=False, 
                                                free_view=False, yaw=0, pitch=0, roll=0)    

        return kp_current, bits
    
    def load_metadata(self)->None:
        bin_file=self.kp_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        self.ref_frame_idx = [int(i) for i in metadata]
        return os.path.getsize(bin_file)*8
 
class FV2V:
    '''Wrapper for models and forward methods'''
    def __init__(self, model_config_path:str, model_checkpoint_path:str, device:str='cpu') -> None:
        self.device = device
        fv2v_analysis_model_detector, fv2v_analysis_model_estimator, fv2v_synthesis_model = load_fv2v_checkpoints(model_config_path, model_checkpoint_path, device=device)
        self.detector_model = fv2v_analysis_model_detector.to(self.device)
        self.estimator_model = fv2v_analysis_model_estimator.to(self.device)
        self.synthesis_model = fv2v_synthesis_model.to(self.device)

        self.reference_frame = None
        self.kp_canonical = None
        self.kp_reference = None

    def predict_inter_frame(self, kp_inter_frame:Dict[str, torch.Tensor], relative=False,
                            adapt_movement_scale=False, estimate_jacobian=False, cpu=False, free_view=False, 
                            yaw=0, pitch=0, roll=0)->torch.Tensor:
        kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_inter_frame,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=(estimate_jacobian & relative),
                           adapt_movement_scale=adapt_movement_scale) 
    
        out = self.synthesis_model(self.reference_frame,kp_source=self.kp_reference,kp_driving= kp_norm)
        return out['prediction']
  
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
    model_dirname='../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   

      
    ############################################                   
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0] 
    #Inputs   
    kp_input_path = model_dirname+'/kp/'+seq+'_qp'+str(qp)+'/'
    ref_input_path = model_dirname+'/enc/'+seq+'_qp'+str(qp)+'/'

    #Outputs
    dec_output_path=model_dirname+'/dec/'
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    dec_video_path_rgb = dec_output_path+seq+'_qp'+str(qp)+'.rgb'
    dec_video_path_mp4 = dec_output_path+seq+'_qp'+str(qp)+'.mp4'
    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)         
    
    #Open file to write output frames
    f_dec=open(dec_video_path_rgb,'w') 

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    #Initialize entropy decoders and models
    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,qp, dec_name=opt.ref_codec)

    #KP decoder
    kp_decoder = FV2VKPDecoder(kp_input_path, q_step=q_step, device=device)
    sum_bits += kp_decoder.load_metadata()
    model_config_path=f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    dec_main = FV2V(model_config_path, model_checkpoint_path, device=device)

    out_video = []
    with torch.no_grad():
        for frame_idx in tqdm(range(frames)):            
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
                
                kp_canonical = dec_main.detector_model(reference)  ####reference 
                he_source = dec_main.estimator_model(reference)  
                kp_reference = keypoint_transformation_source(kp_canonical, he_source, estimate_jacobian=False,
                                                            free_view=False, yaw=0, pitch=0, roll=0) ###### I frame                         


                rec_head_pose_info = kp_decoder.get_kp_list(he_source, frame_idx) 
                kp_decoder.rec_sem.append(rec_head_pose_info)

                #Update decoder with reference frame information
                dec_main.reference_frame = reference
                dec_main.kp_reference = kp_reference
                dec_main.kp_canonical = kp_canonical

            else:
                dec_kp_inter_frame, kp_bits = kp_decoder.decode_kp(dec_main.kp_canonical, frame_idx)
                sum_bits+= kp_bits
            
                # generated frame
                gene_start = time.time()    

                # prediction = make_FV2V_prediction(reference, kp_reference, kp_current, FV2V_Synthesis_Model) #######################
                predicted_frame = dec_main.predict_inter_frame(dec_kp_inter_frame)
                        
                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(predicted_frame)
                #Save to output file
                pred.tofile(f_dec)             
                #HxWx3 format
                out_video.append(np.transpose(pred,[1,2,0]))                 

    f_dec.close()     
    end=time.time()                    

    imageio.mimsave(dec_video_path_mp4,out_video, fps=25.0)
    print(seq+'_qp'+str(qp)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_qp'+str(qp)+'.txt', totalResult, fmt = '%.5f')            



                    
                    
                    
                    
                    
    
    