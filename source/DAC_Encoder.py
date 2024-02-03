import os
import time
import torch
import numpy as np
from GFVC.utils import *
from GFVC.DAC_utils import *
from typing import Dict, List, Any
from GFVC.DAC.metric_utils import eval_metrics
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from skimage.transform import resize
from typing import List
from copy import copy
 

class ReferenceImageCoder:
    def __init__(self,img_output_dir:str,qp:int, frame_format:str, width:int=256, height:int=256):
        self.frame_format = frame_format
        self.qp = qp
        self.width=width
        self.height=height
        self.img_output_dir = img_output_dir


    def vtm_yuv_compress(self, frame_idx:int):
        # wtite ref and cur (rgb444) to file (yuv420)     
        os.system("./image_codecs/vtm/encode.sh "+self.img_output_dir+'frame'+str(frame_idx)+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################

        bin_file=self.img_output_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.img_output_dir+'frame'+str(frame_idx)+'_rec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_yuv[0]
        img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
        img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1  
        return img_rec, bits               
    
    def vtm_rgb_compress(self, frame_idx:int):
        # wtite ref and cur (rgb444) 
        os.system("./image_codecs/vtm/encode_rgb444.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   ########################
        
        bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        sum_bits += bits
        
        f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
        img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                  
    
    def compress(self, reference_frame: List[np.ndarray], frame_idx:int):
        if self.frame_format=='YUV420':
            f_temp=open(self.img_output_dir+'frame'+str(frame_idx)+'_org.yuv','w')
            img_input_rgb = cv2.merge(reference_frame)
            img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
            img_input_yuv.tofile(f_temp)
            f_temp.close()   

            img_rec, bits = self.vtm_yuv_compress(frame_idx)
        elif self.frame_format=='RGB444':
            f_temp=open(self.img_output_dir+'frame'+str(frame_idx)+'_org.rgb','w')
            img_input_rgb = cv2.merge(reference_frame)
            img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
            img_input_rgb.tofile(f_temp)
            f_temp.close()

            img_rec, bits = self.vtm_rgb_compress(frame_idx)
        else:
            raise NotImplementedError(f"Coding in format {self.frame_format} not implemented")
        return img_rec, bits
    
class KPEncoder:
    def __init__(self,kp_output_dir:str, q_step:int=64):
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.rec_sem = []

    def get_kp_list(self, kp_value:torch.Tensor, frame_idx:int)->List[str]:
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
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        res_difference_dec=[i/self.q_step for i in res_difference_dec]

        rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[frame_idx-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits
    
    def encode_metadata(self, metadata: List[int])->None:
        data = copy(metadata)
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits
  
class AdaptiveEncoder:
    def __init__(self,adaptive_metric='psnr',adaptive_thresh=30, device: str='cpu') -> None:
        
        self.cpu = True
        self.device = device #This double check on device selection is annoying
        if device == 'cuda':
            self.cpu = False
        #Initialize the metric computation
        self.adaptive_metric = adaptive_metric.lower()
        if self.adaptive_metric in ['lpips', 'dists']:
            self.metric = eval_metrics[self.adaptive_metric](device=self.device)
        else:
            self.metric = eval_metrics[self.adaptive_metric]()
        self.thresh = adaptive_thresh #Ensure the threshold matches the metric
        self.avg_quality = AverageMeter()

        # Evaluation Buffer
        self.reference_frame_idx = []
        self.current_reference_frame_info: Dict[str, Any] = None

    def get_prediction(self, reference_frame,inter_frame, kp_target, kp_reference):
        prediction = make_DAC_prediction(reference_frame, kp_target, kp_reference, 
                                         DAC_Synthesis_Model, cpu=self.cpu)
        prediction = to_tensor(prediction).to(self.device)
        pred_quality = self.metric.calc(inter_frame,prediction)
        return prediction, pred_quality

    def evaluate(self, inter_frame:torch.Tensor, kp_inter_frame: Dict[str, torch.Tensor])->Dict[str, Any]:
        #Animate with the current reference and evaluate quality
        #If quality is above threshold, then signal to encode the inter_frame KPs
        #Else signal this interframe as the new reference.
        prediction, pred_quality = self.get_prediction(self.current_reference_frame_info['reference'],inter_frame,
                                                       kp_inter_frame, self.current_reference_frame_info['kp_reference'])
        self.avg_quality.update(pred_quality)
        avg_quality = self.avg_quality.avg
        print(frame_idx, avg_quality, pred_quality)
        if self.adaptive_metric in ['psnr', 'ms_ssim', 'fsim'] and avg_quality >= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, True
        elif self.adaptive_metric in ['lpips', 'dists'] and avg_quality <= self.thresh:
            #We have a winner! : encode the keypoints
            return prediction, True
        else:
            return prediction, False


def list2tensor(current_frame:List[np.ndarray], height=256, width=256):
    frame = cv2.merge(current_frame)
    frame = resize(frame, (width, height))[..., :3]
    frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    return frame

if __name__ == "__main__":
   
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--adaptive_metric", default='PSNR', type=str,help="RD adaptation metric (for selecting reference frames to keep in buffer)")
    parser.add_argument("--adaptive_thresh", default=30, type=float,help="Reference selection threshold")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    original_seq=opt.original_seq
    Iframe_format=opt.Iframe_format
    device = opt.device
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    
    ## DAC
    DAC_config_path='./GFVC/DAC/checkpoint/DAC-256.yaml'
    DAC_checkpoint_path='./GFVC/DAC/checkpoint/DAC-checkpoint.pth.tar'         
    DAC_Analysis_Model, DAC_Synthesis_Model = load_DAC_checkpoints(DAC_config_path, DAC_checkpoint_path, cpu=cpu)
    modeldir = 'DAC' 
    model_dirname='../experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
    

###################
    start=time.time()
    kp_output_dir =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'    
    os.makedirs(kp_output_dir,exist_ok=True)     # the frames to be compressed by vtm                 

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                      

    listR,listG,listB=RawReader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    ####
    #Entropy coders
    kp_encoder = KPEncoder(kp_output_dir, Qstep) #Keypoint coder
    ref_coder = ReferenceImageCoder(dir_enc,QP,Iframe_format)
    ####
    #Generator
    dac_encoder = AdaptiveEncoder(adaptive_metric=opt.adaptive_metric,
                                  adaptive_thresh=opt.adaptive_thresh, device=device)

    sum_bits = 0
    with torch.no_grad(): 
        for frame_idx in tqdm(range(0, frames)):            
            frame_idx_str = str(frame_idx).zfill(4)   
            current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
            if frame_idx == 0:      # I-frame      
                img_rec, ref_bits = ref_coder.compress(current_frame, frame_idx)
                sum_bits+=ref_bits
                
                #Extract keypoints
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.to(device)    # require GPU | changed to use the available device

                kp_reference = DAC_Analysis_Model(reference) ################ 

                kp_value = kp_reference['value']
                kp_value_frame = kp_encoder.get_kp_list(kp_value, frame_idx)                  
                kp_encoder.rec_sem.append(kp_value_frame)
                
                #Update the adaptive coder
                ref_info = {'reference': reference,'kp_reference':kp_reference, 'frame_idx':frame_idx}
                dac_encoder.current_reference_frame_info = ref_info
                dac_encoder.reference_frame_idx.append(frame_idx)
            else:
                interframe = list2tensor(current_frame)
                interframe = interframe.to(device)    # require GPU | Use the available device      

                ###extraction
                kp_interframe = DAC_Analysis_Model(interframe) ################
                kp_value_frame = kp_encoder.get_kp_list(kp_interframe['value'], frame_idx)
                rec_kp_value, kp_bits = kp_encoder.encode_kp(kp_value_frame,frame_idx)
                
                #re-format the reconstructed keypoints 
                rec_kp = torch.tensor(rec_kp_value).reshape((1,10,2)).to(device)
                rec_kp_interframe = {'value': rec_kp}
                
                #Generate the final frame and encode kp if reconstruction quality is above threshold
                pred,encode_kp = dac_encoder.evaluate(interframe,rec_kp_interframe)
                if encode_kp:
                    sum_bits+=kp_bits
                else:
                    #We need to encode a new reference frame
                    img_rec, ref_bits = ref_coder.compress(current_frame,frame_idx)
                    sum_bits += ref_bits

                    new_reference = to_tensor(img_rec).to(device)
                    kp_new_reference = DAC_Analysis_Model(new_reference)
                    
                    kp_value_frame = kp_encoder.get_kp_list(kp_new_reference['value'], frame_idx)                 
                    kp_encoder.rec_sem.append(kp_value_frame)
                        
                    ref_info = {'reference': new_reference,'kp_reference':kp_new_reference, 'frame_idx':frame_idx}
                    dac_encoder.current_reference_frame_info = ref_info
                    dac_encoder.reference_frame_idx.append(frame_idx) #update metadata info

    mt_bits = kp_encoder.encode_metadata(dac_encoder.reference_frame_idx)
    sum_bits+= mt_bits
    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   






