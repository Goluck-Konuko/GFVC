import os
import time
import torch
import numpy as np
from GFVC.utils import *
from GFVC.RDAC_utils import *
from typing import List, Dict, Any
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from image_codecs.lic import LICEnc
from GFVC.RDAC.animate import normalize_kp
from GFVC.RDAC.metric_utils import eval_metrics
from GFVC.RDAC.entropy_coders.residual_entropy_coder import ResEntropyCoder


class ReferenceImageCoder:
    def __init__(self,dir_enc:str, qp:int=22, 
                 height: int=256, width: int=256, 
                 format='YUV420',enc_name:str='vtm',model_name='cheng2020attn', device='cpu') -> None:
        self.enc_name= enc_name #[vtm, lic]
        self.dir_enc = dir_enc
        self.qp = qp
        self.device=device
        self.height= height
        self.width = width
        self.format = format
        if self.enc_name == 'lic':
            self.lic_codec = LICEnc(self.dir_enc,model_name=model_name, quality=self.qp, device=self.device)
        else:
            self.lic_codec = None
    def vtm_compress_reference_yuv(self,reference_frame: np.ndarray, frame_idx: int) -> np.ndarray:
        # Compress the reference frame in YUV format and return the decoded image 
        # and number of encoded bits
        # write ref and cur (rgb444) to file (yuv420)
        frame_idx_str = str(frame_idx)
        f_temp=open(self.dir_enc+'frame'+frame_idx_str+'_org.yuv','w')
        img_input_rgb = cv2.merge(reference_frame)
        img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
        img_input_yuv.tofile(f_temp)
        f_temp.close()            

        os.system("./image_codecs/vtm/encode.sh "+self.dir_enc+'frame'+frame_idx_str+" "+str(self.qp)+" "+str(self.width)+" "+str(self.height))   ########################

        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.dir_enc+'frame'+frame_idx_str+'_rec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_yuv[0]
        img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1   
        return img_rec, bits             
                
    def vtm_compress_reference_rgb(self,reference_frame: np.ndarray, frame_idx)->np.ndarray:
        frame_idx_str = str(frame_idx)
        f_temp=open(self.dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
        img_input_rgb = cv2.merge(reference_frame)
        img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
        img_input_rgb.tofile(f_temp)
        f_temp.close()

        os.system("./image_codecs/vtm/encode_rgb444.sh "+self.dir_enc+'frame'+frame_idx_str+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################
        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        f_temp=open(self.dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1   
        return img_rec, bits

    def compress_reference(self, reference_frame:np.ndarray, frame_idx:int)->np.ndarray:
        if self.enc_name == 'vtm':
            if self.format == 'YUV420':
                img_rec, bits = self.vtm_compress_reference_yuv(reference_frame, frame_idx)
            else:
                img_rec, bits = self.vtm_compress_reference_rgb(reference_frame, frame_idx)
        elif self.enc_name == 'lic':
            #Use Cheng2020_attn model to compress
            #be sure to set qp value between 1-6
            ref_fr = to_tensor(np.array(reference_frame)/255)
            # print(ref_fr.shape)
            # reference_frame = resize(cv2.merge(reference_frame), (3, self.height, self.width))
            # reference_frame = to_tensor(reference_frame)
            # print(reference_frame)
            dec_info = self.lic_codec.compress(ref_fr,frame_idx)
            img_rec = dec_info['x_hat']
            bits = dec_info['bits']
        return img_rec, bits

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AdaptiveEncoder:
    def __init__(self,res_out_path, adaptive_metric='psnr',adaptive_thresh=30,res_coding_params:Dict[str, Any] = None,  device: str='cpu') -> None:
        self.cpu = True
        self.res_coding_params = res_coding_params
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
        self.reference_frames_idx = []
        self.current_reference_frame_info: Dict[str, Any] = None

        #Models
        self.generator = RDAC_Synthesis_Model

        #Residual Entropy coder
        # self.res_coder = ResCoder(res_out_path,q_step=res_coding_params['rq_step'])
        self.res_out_path = res_out_path
        self.res_coder = ResEntropyCoder(out_path=res_out_path)
        self.prev_latent, self.res_hat_prev = None, None

    def generate_animation(self, reference_frame: torch.Tensor, kp_target: Dict[str, torch.Tensor], kp_reference: Dict[str, torch.Tensor])->torch.Tensor:
        return self.generator.animate(reference_frame, kp_target, kp_reference)

    def get_prediction(self, reference_frame,inter_frame, kp_target, kp_reference):
        kp_norm = self.normalize(kp_reference, kp_target)
        anim_frame = self.generate_animation(reference_frame, kp_norm, kp_reference)
        return anim_frame
    
    def normalize(self,kp_reference, kp_target, relative=False,adapt_movement_scale=False):
        return normalize_kp(kp_source=kp_reference, kp_driving=kp_target,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    def evaluate(self, target_frame:torch.Tensor, kp_inter_frame: Dict[str, torch.Tensor], frame_idx:int)->Dict[str, Any]:
        #Animate with the current reference and evaluate quality
        #If quality is above threshold, then signal to encode the inter_frame KPs
        #Else signal this interframe as the new reference.
        anim_frame = self.get_prediction(self.current_reference_frame_info['reference'],
                                         inter_frame, kp_inter_frame, 
                                         self.current_reference_frame_info['kp_reference'])

        #compress the residual
        frame_residual = target_frame - anim_frame
        # if self.res_hat_prev is not None:
        #     #We encode a temporal residual
        #     temporal_frame_residual = frame_residual-self.res_hat_prev
        #     res_info, skip = self.generator.tdc.rans_compress(temporal_frame_residual, prev_latent=self.prev_latent, **self.res_coding_params)
        #     #We need to check that the entropy coder doesn't use the latent representation
        #     # of a spatial residual to encode the temporal residual
        #     # if self.res_coder.first_temporal:
        #     #     self.res_coder.previous_res = None
        #     #     #The next temporal residual can use existing samples to predict 
        #     #     # what follows
        #     #     self.res_coder.first_temporal = False 

        # else:
        #We encode a spatial residual
        res_info, skip = self.generator.sdc.rans_compress(frame_residual, prev_latent=self.prev_latent, **self.res_coding_params)
        
        res_bits = 0
        if not skip:
            #flatten res_latent and compress to binary file
            write_bitstring(res_info['bitstring'], self.res_out_path, frame_idx) #res_info['bits'] #res_info['bits']
            dec_info, res_bits = read_bitstring(self.res_out_path, frame_idx)
            res_hat, res_latent_hat = self.generator.sdc.rans_decompress(dec_info['strings'],dec_info['shape'], **self.res_coding_params)
            self.prev_latent = res_latent_hat
            # if self.res_hat_prev is not None:
            #     self.res_hat_prev = res_info['res_hat'] + self.res_hat_prev #Update the frame residual used in reconstruction
            # else:
            self.res_hat_prev = res_hat #Update the frame residual used in reconstruction
        #Skip residual coding and use previously decoded residual
        prediction = anim_frame + self.res_hat_prev
        
        pred_quality = self.metric.calc(inter_frame,prediction)

        self.avg_quality.update(pred_quality)
        avg_quality = self.avg_quality.avg
        # print(pred_quality, avg_quality, res_bits)
        if self.adaptive_metric in ['psnr', 'ms_ssim', 'fsim'] and avg_quality >= self.thresh:
            #We have a winner! : encode the keypoints
            return anim_frame, kp_inter_frame, True, res_bits
        elif self.adaptive_metric in ['lpips', 'dists'] and avg_quality <= self.thresh:
            #We have a winner! : encode the keypoints
            return anim_frame, kp_inter_frame, True, res_bits
        else:
            return anim_frame, kp_inter_frame, False, res_bits

class KPCoder:
    def __init__(self, kp_output_dir:str, q_step:int=64) -> None:
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step

        #coding info
        self.rec_sem=[]

    def encode_kp(self, current_frame_kp: List[float], frame_idx: int)->None:
        ### residual
        kp_difference=(np.array(current_frame_kp)-np.array(self.rec_sem[-1])).tolist()
        ## quantization

        kp_difference=[i*self.q_step for i in kp_difference]
        kp_difference= list(map(round, kp_difference[:]))

        frame_idx = str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+str(frame_idx)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8
        # sum_bits += bits          

        #### decoding for residual
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        res_difference_dec=[i/self.q_step for i in res_difference_dec]

        rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return bits

    def encode_metadata(self, metadata: List[int])->None:
        bin_file=self.kp_output_dir+'/metadata.bin'
        final_encoder_expgolomb(metadata,bin_file)     

        bits=os.path.getsize(bin_file)*8
        return bits

def compress_motion_keypoints(kp_frame: Dict[str, torch.Tensor], frame_idx_str, kp_output_dir)->str:
    #Extract the keypoints
    kp_value = kp_frame['value']
    kp_value_list = kp_value.tolist()
    kp_value_list = str(kp_value_list)
    kp_value_list = "".join(kp_value_list.split())

    with open(kp_output_dir+'/frame'+frame_idx_str+'.txt','w')as f:
        f.write(kp_value_list)  

    kp_value_frame=json.loads(kp_value_list)###20
    kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
    return kp_value_frame

def to_tensor(frame: np.ndarray)->torch.Tensor:
    return torch.tensor(frame[np.newaxis].astype(np.float32))

def read_config_file(config_path):
    '''Simply reads a yaml configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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
    parser.add_argument("--rate_idx", default=0, type=int,help="The RD point for coding the frame residuals [1-5]")
    parser.add_argument("--int_value", default=0.0, type=float,help="Interpolation value between RD points for residual coding [0.0 - 1.0]")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor #KP coding 
    QP=opt.Iframe_QP #Reference frame coding
    original_seq=opt.original_seq
    Iframe_format=opt.Iframe_format
    ######
    # residual coding params
    rate_idx = opt.rate_idx
    int_value = opt.int_value
    gop_size = opt.gop_size
    ######
    device = opt.device
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    ## RDAC
    RDAC_config_path='./GFVC/RDAC/checkpoint/RDAC-256.yaml'
    RDAC_checkpoint_path='./GFVC/RDAC/checkpoint/RDAC-checkpoint.pth.tar'         
    RDAC_Analysis_Model, RDAC_Synthesis_Model = load_RDAC_checkpoints(RDAC_config_path, RDAC_checkpoint_path, cpu=cpu)
    modeldir = 'RDAC' 
    model_dirname='../experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
    ## Load config parameters
    config_params = read_config_file(RDAC_config_path)
    ###################
    start=time.time()
    kp_output_dir =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'     
    os.makedirs(kp_output_dir,exist_ok=True)     # folder to store the encoded motion keypoints                 

    res_output_dir =model_dirname+'/res/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'   
    os.makedirs(res_output_dir,exist_ok=True)     # folder to store the encoded motion keypoints                 

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                      

    listR,listG,listB=RawReader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    seq_kp_integer = []
    sum_bits = 0

    image_coder = ReferenceImageCoder(dir_enc=dir_enc, qp=int(QP), height=height,
                                      width=width, format=Iframe_format,enc_name='lic',device='cpu') #Always the run the autoregressive models on CPU
    

    
    kp_coder = KPCoder(kp_output_dir=kp_output_dir,q_step=Qstep)

    #Adaptive reference info
    adaptive_animator = AdaptiveEncoder(res_output_dir, adaptive_metric=opt.adaptive_metric, 
                                          adaptive_thresh=opt.adaptive_thresh,
                                          res_coding_params={'rate_idx':rate_idx,'q_value':int_value},
                                          device=device)

    #Coding loop
    tot_kp_bits = 0
    for frame_idx in tqdm(range(0, frames)):            
        frame_idx_str = str(frame_idx).zfill(4)  
        current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]] 
        if frame_idx == 0 or frame_idx%gop_size==0:      # The first frame must be an I-frame      
            adaptive_animator.reference_frames_idx.append(frame_idx)
            adaptive_animator.res_coder.metadata.append(0)
            img_rec, ref_bits = image_coder.compress_reference(current_frame,frame_idx)                         
            sum_bits += ref_bits
            with torch.no_grad(): 
                if not isinstance(img_rec, torch.Tensor):
                    reference = to_tensor(img_rec).to(device)    # require GPU | changed to use the available device
                else:
                    reference = img_rec.detach().clone().to(device)
                kp_reference = RDAC_Analysis_Model(reference)
                kp_value_frame = compress_motion_keypoints(kp_reference, frame_idx_str, kp_output_dir)  
                kp_coder.rec_sem.append(kp_value_frame)  
                
                #Update the adaptive coder
                ref_info = {'reference': reference,'kp_reference':kp_reference, 'frame_idx':frame_idx}
                adaptive_animator.current_reference_frame_info = ref_info
                adaptive_animator.res_hat_prev = None
        else: #Subsequent frames can be encoded as inter_frames if they meet the adaptation threshold
            inter_frame = cv2.merge(current_frame)
            inter_frame = resize(inter_frame, (width, height))[..., :3]
            with torch.no_grad(): 
                inter_frame = to_tensor(inter_frame).permute(0, 3, 1, 2).to(device)
                ### KP Extraction
                kp_inter_frame = RDAC_Analysis_Model(inter_frame) ################
                ## ------------------------
                ## Animate the target frame and evaluate the quality
                anim_prediction, kp_inter_frame, encode, res_bits = adaptive_animator.evaluate(inter_frame, kp_inter_frame, frame_idx)
                sum_bits += res_bits
                # print("Res bits: ",res_bits)
                ## ------------------------
                if encode:
                    kp_value_frame = compress_motion_keypoints(kp_inter_frame, frame_idx_str, kp_output_dir)
                    kp_bits = kp_coder.encode_kp(kp_value_frame, frame_idx) 
                    sum_bits += kp_bits 
                    tot_kp_bits += kp_bits
                    adaptive_animator.res_coder.metadata.append(1)
                    # print("KP bits: ", kp_bits)
                else:
                    #Adding this target to buffer and compressing it as a new reference
                    #compress the current frame and transmit as the latest reference
                    adaptive_animator.reference_frames_idx.append(frame_idx)
                    img_rec, ref_bits = image_coder.compress_reference(current_frame,frame_idx)
                     
                    if not isinstance(img_rec, torch.Tensor):
                        new_reference = to_tensor(img_rec).to(device)    # require GPU | changed to use the available device
                    else:
                        new_reference = img_rec.detach().clone()
                    
                    kp_new_reference = RDAC_Analysis_Model(new_reference)
                    
                    kp_value_frame = compress_motion_keypoints(kp_new_reference, frame_idx_str, kp_output_dir)
                    kp_coder.rec_sem.append(kp_value_frame)       

                    ref_info = {'reference': new_reference,'kp_reference':kp_new_reference, 'frame_idx':frame_idx}
                    adaptive_animator.current_reference_frame_info = ref_info
                    adaptive_animator.res_coder.metadata.append(0)
                    sum_bits += ref_bits 
                    # print("Ref bits: ", ref_bits)

    #Write any metadata here and finish the coding process
    m_kp_bits = kp_coder.encode_metadata(adaptive_animator.reference_frames_idx)
    # print("KP metadata: ", m_kp_bits, 'bits')
    sum_bits += m_kp_bits
    #Encode residual metadata
    m_res_bits = adaptive_animator.res_coder.encode_metadata()
    # print("Res metadata: ", m_res_bits, 'bits')
    sum_bits += m_res_bits
    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   






