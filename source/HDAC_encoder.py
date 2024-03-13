import os
import time
import torch
import imageio
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.HDAC_utils import *
from GFVC.HDAC.metric_utils import eval_metrics
from GFVC.HDAC.bl_codecs import bl_encoders
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List
from PIL import Image


class HDAC:
    '''DAC Encoder'''
    def __init__(self, hdac_config_path:str, hdac_checkpoint_path:str,adaptive_metric='lpips',adaptive_thresh=0.5, device='cpu') -> None:
        self.device = device
        dac_analysis_model, dac_synthesis_model = load_hdac_checkpoints(hdac_config_path, hdac_checkpoint_path, device=device)
        self.analysis_model = dac_analysis_model.to(device)
        self.synthesis_model = dac_synthesis_model.to(device)

        self.base_layer = None

        #adaptive refresh tools
        self.adaptive_metric = adaptive_metric.lower()
        if self.adaptive_metric in ['lpips', 'dists']:
            self.metric = eval_metrics[self.adaptive_metric](device=self.device)
        else:
            self.metric = eval_metrics[self.adaptive_metric]()
        self.thresh = adaptive_thresh #Ensure the threshold matches the metric

        self.avg_quality = AverageMeter()

    def predict_inter_frame(self,kp_inter_frame: Dict[str,torch.Tensor],frame_idx:int)->torch.Tensor:
        #the actual prediction call --> similar to what will happend in the GFVC decoder
        if self.base_layer is not None:
            base_layer_frame = frame2tensor(self.base_layer[frame_idx]).to(self.device)
        else:
            base_layer_frame = None

        prediction = self.synthesis_model.predict(self.reference_frame, self.kp_reference,
                                                   kp_inter_frame,base_layer_frame)
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
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
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
    q_step= opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    gop_size = opt.gop_size
    device = opt.device
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
    

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    
    model_name = 'HDAC' 
    model_dirname=f'../experiment/{model_name}_{bl_codec_name.upper()}/Iframe_'+str(iframe_format)   
    
    

    ###################
    start=time.time()
    kp_output_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/'    #OutPut directory for motion keypoints
    os.makedirs(kp_output_path,exist_ok=True)                   

    enc_output_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'/' ## OutPut path for encoded reference frame     
    os.makedirs(enc_output_path,exist_ok=True)     

    bl_output_path =model_dirname+'/bl/'+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp) ## OutPut path for encoded HEVC| VVC base layer  
    os.makedirs(bl_output_path,exist_ok=True)                     

    listR,listG,listB=raw_reader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    seq_kp_integer = []
    sum_bits = 0

    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name=opt.ref_codec, device=device)
    #Motion keypoints coding
    kp_coder = KPEncoder(kp_output_path,q_step,num_kp=num_kp, device=device)

    

    #Main encoder models wrapper
    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path= f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    enc_main = HDAC(model_config_path, model_checkpoint_path,
                   opt.adaptive_metric,opt.adaptive_thresh,device=device)

    
    with torch.no_grad():
        out_video = []
        g_start, g_end = 0,0
        if use_base_layer == 'ON':
            #create base layer spanning the entire sequence
            print(f'Creating {bl_codec_name} base layer..')
            gop = np.transpose(np.array([listR,listG,listB]),[1,2,3,0])
            N, h,w,_ = gop.shape
            bl_coding_params = {
                    'qp':bl_qp,
                    'sequence':resize_frames(gop, bl_scale_factor),
                    'fps':25,
                    'frame_dim': (h,w),
                    'log_path': bl_output_path
                }
            
            bl_enc = bl_encoders[bl_codec_name](**bl_coding_params)
            info_out = bl_enc.run()
            bl_frames = np.transpose(resize_frames(info_out['dec_frames'], 1//bl_scale_factor),[0,3,1,2])
            sum_bits += info_out['bits']
            enc_main.base_layer = bl_frames
    
        print("Extracting animation keypoints..")    
        for frame_idx in tqdm(range(0, frames)):            
            current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
            cur_out = np.transpose(np.array(current_frame),[1,2,0])
            if frame_idx == 0:      # I-frame      
                reference, ref_bits = ref_coder.compress(current_frame, frame_idx)  
                sum_bits+=ref_bits          
                if isinstance(reference, np.ndarray):
                    #convert to tensor
                    out_fr = reference
                    #convert the HxWx3 (uint8)-> 1x3xHxW (float32) 
                    reference = frame2tensor(np.transpose(reference,[2,0,1]))
                else:
                    #When using LIC for reference compression we get back a tensor 1x3xHXW
                    out_fr = np.transpose(tensor2frame(reference),[1,2,0])
                

                if use_base_layer=='ON':
                    bl_fr = np.transpose(enc_main.base_layer[frame_idx], [1,2,0])
                    out_video.append(np.concatenate([cur_out, bl_fr,out_fr], axis=1))
                else:
                    out_video.append(np.concatenate([cur_out,out_fr], axis=1))

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
                # print(rec_kp_frame)
                #Reconstruct the frame and evaluate quality
                pred = enc_main.predict_inter_frame(rec_kp_frame, frame_idx)
                ####
                pred = np.transpose(tensor2frame(pred),[1,2,0])
                if use_base_layer == 'ON':
                    bl_fr = np.transpose(enc_main.base_layer[frame_idx],[1,2,0])
                    out_video.append(np.concatenate([cur_out,bl_fr,pred],axis=1))
                else:
                    out_video.append(np.concatenate([cur_out,pred],axis=1))
                sum_bits += kp_bits
               
    imageio.mimsave(f"{enc_output_path}enc_video.mp4", out_video, fps=25.0)
    sum_bits+= kp_coder.encode_metadata()
    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   






