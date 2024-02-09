import os
import time
import torch
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.DAC_utils import *
from GFVC.DAC.metric_utils import eval_metrics
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List


class DACKPEncoder:
    def __init__(self,kp_output_dir:str, q_step:int=64, device='cpu'):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.device = device
        self.rec_sem = []

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
        kp_current_value=torch.Tensor(kp_value).reshape((1,10,2)).to(self.device)          
        kp_inter_frame['value']=kp_current_value  
        #reconstruct the KPs 
        return kp_inter_frame, bits
    
    def encode_metadata(self, metadata: List[int])->None:
        '''this can be optimized to use run-length encoding which would be more efficient'''
        data = copy(metadata)
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits
  
class DAC:
    '''DAC Encoder'''
    def __init__(self, dac_config_path:str, dac_checkpoint_path:str,adaptive_metric='psnr',adaptive_thresh=20, device='cpu') -> None:
        self.device = device
        dac_analysis_model, dac_synthesis_model = load_dac_checkpoints(dac_config_path, dac_checkpoint_path, device=device)
        self.analysis_model = dac_analysis_model.to(device)
        self.synthesis_model = dac_synthesis_model.to(device)

        #adaptive refresh tools
        self.adaptive_metric = adaptive_metric.lower()
        if self.adaptive_metric in ['lpips', 'dists']:
            self.metric = eval_metrics[self.adaptive_metric](device=self.device)
        else:
            self.metric = eval_metrics[self.adaptive_metric]()
        self.thresh = adaptive_thresh #Ensure the threshold matches the metric
        self.avg_quality = AverageMeter()

    def predict_inter_frame(self, kp_inter_frame: Dict[str,torch.Tensor])->torch.Tensor:
        #the actual prediction call --> similar to what will happend in the GFVC decoder
        prediction = self.synthesis_model.animate(self.reference_frame,self.kp_reference, kp_inter_frame)
        return prediction
    
    def evaluate(self,kp_inter_frame: Dict[str,torch.Tensor], frame_idx:int)->tuple:
        prediction = self.predict_inter_frame(kp_inter_frame)
        #evaluate the quality of the current prediction 
        pred_quality = self.metric.calc(inter_frame,prediction)
        self.avg_quality.update(pred_quality)
        avg_quality = self.avg_quality.avg
        # print(frame_idx,  anim_pred_quality, pred_quality)
        # If the quality is above the current threshold then send keypoints 
        # else encode a new reference frame
        if self.adaptive_metric in ['psnr','ssim','ms_ssim','fsim'] and avg_quality >= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, True
        elif self.adaptive_metric in ['lpips', 'dists'] and avg_quality <= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, True 
        else:
            return prediction, False


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

    
    model_name = 'DAC' 
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
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name='vtm', device=device)
    #Motion keypoints coding
    kp_coder = DACKPEncoder(kp_output_path,q_step, device=device)


    #Main encoder models wrapper
    model_config_path= f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'         
    enc_main = DAC(model_config_path, model_checkpoint_path,
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

                #Reconstruct the frame and evaluate quality
                pred, encode_kp = enc_main.evaluate(rec_kp_frame, frame_idx)
                if encode_kp:
                    sum_bits += kp_bits
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
                    #update enc main with reference frame info
                    enc_main.reference_frame = reference
                    enc_main.kp_reference = kp_reference

    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   





