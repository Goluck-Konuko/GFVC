import os
import time
import json
import torch
import imageio
import numpy as np
from GFVC.utils import *
from GFVC.HDAC_utils import *
from GFVC.HDAC.bl_codecs import bl_decoders
from argparse import ArgumentParser
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict, List
from copy import copy
from PIL import Image


class HDAC:
    '''HDAC Decoder models'''
    def __init__(self, hdac_config_path:str, hdac_checkpoint_path:str, device='cpu') -> None:
        self.device = device
        hdac_analysis_model, hdac_synthesis_model = load_hdac_checkpoints(hdac_config_path, hdac_checkpoint_path, device=device)
        self.analysis_model = hdac_analysis_model.to(device)
        self.synthesis_model = hdac_synthesis_model.to(device)

        self.reference_frame = None
        self.kp_reference = None
        self.base_layer = None

    def predict_inter_frame(self,kp_inter_frame:Dict[str, torch.Tensor], frame_idx:int)->torch.Tensor:
        if self.base_layer is not None:
            base_layer_frame = frame2tensor(self.base_layer[frame_idx]).to(self.device)
        else:
            base_layer_frame = torch.zeros_like(self.reference_frame).to(self.device)
        prediction = self.synthesis_model.predict(self.reference_frame,self.kp_reference, kp_inter_frame, base_layer_frame)
        return prediction

def resize_frames(frames: np.ndarray, scale_factor=1)->np.ndarray:
    N, H, W, C = frames.shape
    if scale_factor != 1:
        out = []
        for idx in range(N):
            img = Image.fromarray(frames[idx])
            img = img.resize((int(H*scale_factor), int(W*scale_factor)),resample=Image.Resampling.LANCZOS)
            out.append(np.asarray(img))
        return np.array(out)
    else:
        return frames
  


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--ref_codec", default='vtm', type=str,help="Reference frame coder [vtm | lic]")
    parser.add_argument("--adaptive_metric", default='PSNR', type=str,help="RD adaptation metric (for selecting reference frames to keep in buffer)")
    parser.add_argument("--adaptive_thresh", default=30, type=float,help="Reference selection threshold")
    parser.add_argument("--use_base_layer", default='ON', type=str,help="Flag to use hybrid coding framework (OFF sets animation-only reconstruction)")
    parser.add_argument("--base_codec", default='hevc', type=str,help="Base layer codec [hevc | vvc]")
    parser.add_argument("--bl_qp", default=50, type=int,help="QP value for encoding the base layer")
    parser.add_argument("--bl_scale_factor", default=1.0, type=float,help="subsampling factor for base layer frames")
    parser.add_argument("--num_kp", default=10, type=int,help="Number of motion keypoints")
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
    thresh = int(opt.adaptive_thresh)
    num_kp = int(opt.num_kp)

    #base layer params
    use_base_layer = opt.use_base_layer
    bl_codec_name = opt.base_codec
    bl_qp = opt.bl_qp
    bl_scale_factor = opt.bl_scale_factor
    ###

    if not torch.cuda.is_available():
        device = 'cpu'
        
    ##
    model_name = "HDAC"

    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path= f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'  
    model_dirname=f'../experiment/{model_name}_{bl_codec_name.upper()}/Iframe_{iframe_format}'  
    
    ###################################################

    kp_input_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/' 
    bl_input_path =model_dirname+'/bl/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)   

    ref_input_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/'
    os.makedirs(ref_input_path,exist_ok=True)     # the frames to be compressed by vtm    

    dec_output_path = model_dirname+'/dec/'
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    dec_sequence_path_rgb  =dec_output_path+seq+'_qp'+str(qp)+'_th'+str(bl_qp)+'.rgb'
    dec_sequence_path_mp4 =dec_output_path+seq+'_qp'+str(qp)+'_th'+str(bl_qp)+'.mp4'


    dir_bit=model_dirname+f'/resultBit_{bl_qp}/'
    os.makedirs(dir_bit,exist_ok=True)        


    #Initialize the output file 
    f_dec=open(dec_sequence_path_rgb,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,int(qp),dec_name=opt.ref_codec)

    # Keypoint Decoder
    kp_decoder = KPDecoder(kp_input_path,q_step,num_kp=num_kp, device=device)
    sum_bits += kp_decoder.load_metadata()
    #Reconstruction models
    dec_main = HDAC(model_config_path, model_checkpoint_path,device=device)
    
    output_video = [] #Output an mp4 video for sanity check
    with torch.no_grad():
        #decode the base layer if available
        if use_base_layer == 'ON':
            #create base layer spanning the entire sequence
            print('Creating base layer..')
            bl_decoding_params = {
                    'fps':25,
                    'frame_dim': (int(height*bl_scale_factor),int(width*bl_scale_factor)),
                    'bin_path': bl_input_path}
            bl_dec = bl_decoders[bl_codec_name](**bl_decoding_params)
            info_out = bl_dec.run()
            bl_frames = np.transpose(resize_frames(info_out['dec_frames'], 1//bl_scale_factor),[0,3,1,2])
            sum_bits += info_out['bits']
            dec_main.base_layer = bl_frames

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
                
                if use_base_layer == 'ON':
                    bl_fr = np.transpose(dec_main.base_layer[frame_idx],[1,2,0])
                    out_fr = np.concatenate([bl_fr, out_fr], axis=1)
 
                output_video.append(out_fr)
                img_rec_out.tofile(f_dec)
                
                reference = reference.to(device) #resize(img_rec, (3, height, width))    # normalize to 0-1  
                kp_reference = dec_main.analysis_model(reference) 
                #update decoder with the reference frame info
                dec_main.reference_frame = reference
                dec_main.kp_reference = kp_reference

                #append to list for use in predictively coding the next frame KPs
                #KPs used for relative decoding of inter_Frame KPs
                kp_value_frame = kp_decoder.get_kp_list(kp_reference, frame_idx)

                kp_decoder.rec_sem = []
                kp_decoder.rec_sem.append(kp_value_frame)
            else:
                # Decoding motion features
                rec_kp_frame, kp_bits = kp_decoder.decode_kp(frame_idx)
                # print(rec_kp_frame)
                sum_bits+= kp_bits
                # print(dec_kp_inter_frame)
                # Inter_frame reconstruction through animation
                gene_start = time.time()
                predicted_frame  = dec_main.predict_inter_frame(rec_kp_frame, frame_idx)

                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(predicted_frame)
                #Save to output file
                pred.tofile(f_dec)

                if use_base_layer == 'ON':
                    bl_fr = np.transpose(dec_main.base_layer[frame_idx],[1,2,0])
                    pred = np.transpose(pred, [1,2,0])
                    out_fr = np.concatenate([bl_fr, pred], axis=1)
                else:
                    out_fr = np.transpose(pred, [1,2,0])
                                           
                #HxWx3 format
                output_video.append(out_fr)                 

    f_dec.close()     
    end=time.time()
    imageio.mimsave(dec_sequence_path_mp4,output_video, fps=25.0)
    print(seq+'_qp'+str(qp)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'.txt', totalResult, fmt = '%.5f')            


