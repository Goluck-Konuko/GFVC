import os
import time
import torch
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.RDAC_utils import *
from GFVC.RDAC.metric_utils import eval_metrics
from GFVC.RDAC.entropy_coders.residual_entropy_coder import ResEntropyCoder
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List,Any


class RDAC:
    '''Wrapper for Predictive Coding with Animation-Based Models (RDAC)'''
    def __init__(self, rdac_config_path:str, rdac_checkpoint_path:str,res_output_path:str,res_coding_params:Dict[str,Any],adaptive_metric='psnr',adaptive_thresh=20, device='cpu') -> None:
        self.device = device
        rdac_analysis_model, rdac_synthesis_model = load_rdac_checkpoints(rdac_config_path, rdac_checkpoint_path, device=device)
        self.analysis_model = rdac_analysis_model.to(device)
        self.synthesis_model = rdac_synthesis_model.to(device)

        #adaptive refresh tools
        self.adaptive_metric = adaptive_metric.lower()
        if self.adaptive_metric in ['lpips', 'dists']:
            self.metric = eval_metrics[self.adaptive_metric](device=self.device)
        else:
            self.metric = eval_metrics[self.adaptive_metric]()
        self.thresh = adaptive_thresh #Ensure the threshold matches the metric
        self.avg_quality = AverageMeter()

        #Residual coder
        self.res_out_path = res_output_path
        self.res_coder = ResEntropyCoder(out_path=self.res_out_path)
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


    def predict_inter_frame(self, kp_inter_frame: Dict[str,torch.Tensor])->torch.Tensor:
        #the actual prediction call --> similar to what will happend in the GFVC decoder
        prediction = self.synthesis_model.animate(self.reference_frame,self.kp_reference, kp_inter_frame)
        return prediction
    
    def evaluate(self,inter_frame, kp_inter_frame: Dict[str,torch.Tensor], frame_idx:int)->tuple:
        anim_prediction = self.predict_inter_frame(kp_inter_frame)

        #compute the frame residual
        frame_residual = inter_frame - anim_prediction
        res_bits = 0
        if self.prev_res_hat is not None:
            #Compress residual after removing temporal rendundancy
            deform_params = {'prev_rec': self.prev_animation,
                             'kp_prev': self.prev_kp_target,
                             'kp_cur': kp_inter_frame,
                             'res_hat_prev': self.prev_res_hat}
            def_prev_res_hat = self.synthesis_model.deform_residual(deform_params)

            temporal_frame_residual = (frame_residual-def_prev_res_hat)/2.0
            res_info, skip = self.synthesis_model.tdc.rans_compress(temporal_frame_residual,self.prev_latent, **self.res_coding_params)
            if skip:
                res_hat = def_prev_res_hat
                self.res_coding_flag.append(0)
            else:
                #write and decode the temporal residual bitstream
                write_bitstring(res_info['bitstring'], self.res_out_path, frame_idx)
                dec_info, res_bits = read_bitstring(self.res_out_path, frame_idx)
                res_hat_temp , _ = self.synthesis_model.tdc.rans_decompress(dec_info['strings'],dec_info['shape'], **self.res_coding_params)
 
                res_hat = def_prev_res_hat + res_hat_temp*2.0
                self.prev_latent = res_info['res_latent_hat'] 
                self.prev_res_hat = res_hat
                self.res_coding_flag.append(1)
        else:
            #Compress the without temporal redundancy removal
            res_info, _ = self.synthesis_model.sdc.rans_compress(frame_residual,self.prev_latent,**self.res_coding_params)

            #flatten res_latent and compress to binary file
            write_bitstring(res_info['bitstring'], self.res_out_path, frame_idx) #res_info['bits'] #res_info['bits']
            dec_info, res_bits = read_bitstring(self.res_out_path, frame_idx)
            res_hat, _ = self.synthesis_model.sdc.rans_decompress(dec_info['strings'],dec_info['shape'], **self.res_coding_params)
            self.res_coding_flag.append(1)
            #keep the prev latent for skip mode
            self.prev_latent = res_info['res_latent_hat']
            self.prev_res_hat = res_hat
        
        #evaluate the quality of the current prediction 
        prediction = (anim_prediction+res_hat).clamp(0,1)
        #UPdate buffer info for temporal residual prediction
        self.prev_animation = prediction
        self.prev_kp_target = kp_inter_frame

        pred_quality = self.metric.calc(inter_frame,prediction)
        self.avg_quality.update(pred_quality)
        avg_quality = self.avg_quality.avg

        # print(frame_idx, pred_quality)

        # If the quality is above the current threshold then send keypoints 
        # else encode a new reference frame
        if self.adaptive_metric in ['psnr','ssim','ms_ssim','fsim'] and avg_quality >= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, anim_prediction, True, res_bits
        elif self.adaptive_metric in ['lpips', 'dists'] and avg_quality <= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, anim_prediction, True , res_bits
        else:
            return prediction, anim_prediction, False, res_bits


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
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
    q_step= opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    gop_size = opt.gop_size
    rate_idx = opt.rate_idx
    int_value = opt.int_value
    device = opt.device
    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    model_name = 'RDAC' 
    model_dirname=f'../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format)   
    
    ###################
    start=time.time()
    #Output path for compressed keypoints
    kp_output_path =model_dirname+'/kp/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/'    #OutPut directory for motion keypoints
    os.makedirs(kp_output_path,exist_ok=True)                   
    #Output path for compressed frame residualcs
    res_output_path =model_dirname+'/res/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/'    #OutPut directory for motion keypoints
    os.makedirs(res_output_path,exist_ok=True) 
    #Output path for compressed reference frames
    enc_output_path =model_dirname+'/enc/'+seq+'_qp'+str(qp)+'_rqp'+str(rate_idx)+'/' ## OutPut path for encoded reference frame     
    os.makedirs(enc_output_path,exist_ok=True)                      

    listR,listG,listB=raw_reader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    seq_kp_integer = []
    sum_bits = 0

    #Reference image coder [VTM, LIC]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name='vtm', device=device)
    #Motion keypoints coding
    kp_coder = KPEncoder(kp_output_path,q_step, device=device)


    #Main encoder models wrapper
    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    
    res_coding_params = {'rate_idx':rate_idx,'q_value':int_value,
                         'use_skip':True,'skip_thresh':0.75}
    enc_main = RDAC(model_config_path, model_checkpoint_path,res_output_path,res_coding_params,
                   opt.adaptive_metric,opt.adaptive_thresh,device=device)

    
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
                
                #update enc main with reference frame info
                enc_main.reference_frame = reference
                enc_main.kp_reference = kp_reference
                enc_main.res_coding_flag.append(0)
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

                #Reconstruct the frame and evaluate quality
                enh_pred, pred, encode_kp, res_bits = enc_main.evaluate(inter_frame,rec_kp_frame, frame_idx)
                if encode_kp:
                    sum_bits += kp_bits
                    sum_bits += res_bits 
                else:
                    #compress the current frame as a new reference frame
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
                    kp_coder.rec_sem.append(kp_value_frame) #NOTE: The KP reconstructed above before the evaluation was also added to the rec_sem buffer!
                    kp_coder.ref_frame_idx.append(frame_idx) #metadata for reference frame indices
                
                    #update enc main with reference frame info
                    enc_main.reference_frame = reference
                    enc_main.kp_reference = kp_reference
                    enc_main.res_coding_flag.append(0)

    #Encode metadata
    sum_bits+= kp_coder.encode_metadata()
    res_mt_bits = enc_main.res_coder.encode_metadata(enc_main.res_coding_flag)
    # print("Res metadata: ", m_res_bits, 'bits')
    sum_bits += res_mt_bits

    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   






