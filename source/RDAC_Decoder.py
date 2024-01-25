import os
import time
import json
import torch
import numpy as np
from GFVC.utils import *
from GFVC.RDAC_utils import *
from image_codecs.lic import LICDec
from argparse import ArgumentParser
from skimage.transform import resize
from GFVC.RDAC.animate import normalize_kp
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import List, Dict, Any
from GFVC.RDAC.entropy_coders.residual_entropy_coder import ResEntropyDecoder


class ReferenceImageDecoder:
    def __init__(self, dir_enc:str, qp:int=22, height: int=256, width: int=256, format='YUV420', enc_name='vtm', device='cpu', model_name='cheng2020attn') -> None:
        self.enc_name = enc_name
        self.device = device
        self.dir_enc = dir_enc
        self.qp = qp
        self.height= height
        self.width = width
        self.format = format
        if self.enc_name == 'lic':
            self.lic_decoder = LICDec(self.dir_enc,model_name=model_name, quality=self.qp, device=self.device)
        else:
            self.lic_decoder = None

    def vtm_decompress_reference_yuv(self, frame_idx:int):
        frame_idx_str = str(frame_idx)
        os.system("./image_codecs/vtm/decode.sh "+self.dir_enc+'frame'+frame_idx_str)
        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8

        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.dir_enc+'frame'+frame_idx_str+'_dec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_yuv[0]
        img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
        img_rec.tofile(f_dec)                         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1
        return img_rec , bits          
                    
    def vtm_decompress_reference_rgb(self, frame_idx: int)->np.ndarray:
        frame_idx_str = str(frame_idx)
        os.system("./image_codecs/vtm/decode_rgb444.sh "+self.dir_enc+'frame'+frame_idx_str)
        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8

        f_temp=open(self.dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec.tofile(f_dec) 
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1       
        return img_rec, bits   

    def decompress_reference(self,frame_idx:int)->np.ndarray:
        if self.enc_name == 'vtm':
            if self.format == 'YUV420':
                img_rec, bits = self.vtm_decompress_reference_yuv(frame_idx)
            else:
                img_rec, bits = self.vtm_decompress_reference_rgb(frame_idx)
        elif self.enc_name == 'lic':
            dec_info = self.lic_decoder.decompress(frame_idx)
            img_rec = dec_info['x_hat']
            bits = dec_info['bits']
        return img_rec, bits

class KPDecoder:
    def __init__(self, kp_output_dir:str, q_step:int=64, device='cpu') -> None:
        self.device= device
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step

        #coding info
        self.rec_sem=[]
        self.reference_frame_idx = [0]

    def decode_kp(self, frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_save=self.kp_output_dir+'/frame'+frame_index+'.bin'            
        kp_dec = final_decoder_expgolomb(bin_save)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quantization
        kp_difference_dec=[i/self.q_step for i in kp_difference]
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer,kp_value= listformat_kp_DAC(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_value=json.loads(kp_value)
        kp_target_value=torch.Tensor(kp_value).to(self.device)          
        kp_target_decoded = {'value': kp_target_value.reshape((1,10,2))  }
        return kp_target_decoded

    def load_metadata(self)->None:
        bin_file=self.kp_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        self.reference_frame_idx = [int(i) for i in metadata]
        return os.path.getsize(bin_file)*8

def frame2tensor(frame: np.ndarray)->torch.Tensor:
    return torch.tensor(frame[np.newaxis].astype(np.float32))

def tensor2frame(frame:torch.Tensor)->np.ndarray:
    pred = frame.detach().cpu().numpy()[0]
    return (pred*255).astype(np.uint8) 

class AdaptiveDecoder:
    def __init__(self, generator, output_dir:str, res_inp_path:str,res_coding_params:Dict[str,Any],adaptive_metric:str='psnr', device:str='cpu'):
        self.device = device
        self.output_dir = output_dir
        self.res_inp_path = res_inp_path #path to encoded frame residuals
        self.residual_coding_params = res_coding_params

        #Residual entropy decoder [Wrapper for the Arithmetic Decoder]
        self.res_decoder = ResEntropyDecoder(input_path=self.res_inp_path)
        self.prev_res_hat = None
        #Models
        self.generator =generator

    def generate_animation(self, reference_frame: torch.Tensor, kp_target: Dict[str, torch.Tensor], kp_reference: Dict[str, torch.Tensor])->torch.Tensor:
        return self.generator.animate(reference_frame, kp_target, kp_reference)

    def get_prediction(self, reference_frame, kp_target, kp_reference):
        kp_norm = self.normalize(kp_reference, kp_target)
        anim_frame = self.generate_animation(reference_frame, kp_norm, kp_reference)
        return anim_frame
    
    def normalize(self,kp_reference, kp_target, relative=False,adapt_movement_scale=False):
        return normalize_kp(kp_source=kp_reference, kp_driving=kp_target,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)

    def decode(self, reference, kp_target, kp_reference, frame_idx:int)->Dict[str, Any]:
        #Generate Animation
        anim_frame = self.generate_animation(reference, kp_target, kp_reference)
        #Decode residual and reconstruct
        res_bits = 0
        if self.res_decoder.metadata[frame_idx] == 1:
            #read the encoded frame residual latent and decode it
            bitstring, res_bits = read_bitstring(self.res_inp_path, frame_idx)
            res_hat, _ = self.generator.sdc.rans_decompress(bitstring['strings'], bitstring['shape'], 
                                                                         rate_idx=self.residual_coding_params['rate_idx'],
                                                                         q_value=self.residual_coding_params['q_value'])
            # res_latent_info = self.res_decoder.decode_res(frame_idx)
            # res_latent_hat = torch.tensor(res_latent_info['res_latent_hat'], dtype=torch.float32).reshape((1,72,8,8)).to(self.device)
            # res_hat = self.generator.sdc.ae_decompress(res_latent_hat,rate_idx=self.residual_coding_params['rate_idx'], q_value=self.residual_coding_params['q_value'])
            prediction = anim_frame + res_hat
            self.prev_res_hat = res_hat
        else:
            prediction = anim_frame + self.prev_res_hat
        return prediction, res_bits

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
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--adaptive_metric", default='PSNR', type=str,help="RD adaptation metric (for selecting reference frames to keep in buffer)")
    parser.add_argument("--adaptive_thresh", default=20, type=float,help="Reference selection threshold")
    parser.add_argument("--rate_idx", default=0, type=int,help="The RD point for coding the frame residuals [1-5]")
    parser.add_argument("--int_value", default=0.0, type=float,help="Interpolation value between RD points for residual coding [0.0 - 1.0]")
    parser.add_argument("--gop_size", default=32, type=int,help="Max number of of frames to animate from a single reference")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    Iframe_format=opt.Iframe_format
    # residual coding params
    rate_idx = opt.rate_idx
    int_value = opt.int_value
    #####
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    device = opt.device
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    
    ## RDAC
    RDAC_config_path='./GFVC/RDAC/checkpoint/RDAC-256.yaml'
    RDAC_checkpoint_path='./GFVC/RDAC/checkpoint/RDAC-checkpoint.pth.tar'         
    RDAC_Analysis_Model, RDAC_Synthesis_Model = load_RDAC_checkpoints(RDAC_config_path, RDAC_checkpoint_path, cpu=cpu)
    modeldir = 'RDAC' 
    model_dirname='../experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
    ## Load config parameters
    config_params = read_config_file(RDAC_config_path)
    ###################################################

    kp_output_dir =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'

    res_input_dir =model_dirname+'/res/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'  
    os.makedirs(res_input_dir,exist_ok=True)     # folder to store the encoded motion keypoints                 

    dir_dec=model_dirname+'/dec/'
    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'.rgb'

    
    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm     

    dir_bit=model_dirname+'/resultBit'+'_RQP' +str(rate_idx)+'/'
    os.makedirs(dir_bit,exist_ok=True)        


    f_dec=open(decode_seq,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    image_decoder = ReferenceImageDecoder(dir_enc, qp=int(QP), height=height,width=width, format=Iframe_format,enc_name='lic',device='cpu')
    
    kp_decoder = KPDecoder(kp_output_dir,q_step=Qstep, device=device)
    sum_bits += kp_decoder.load_metadata()
    
    rdac_decoder = AdaptiveDecoder(generator=RDAC_Synthesis_Model,
                                   output_dir=dir_dec, res_inp_path=res_input_dir,
                                   res_coding_params={'rate_idx':rate_idx,'q_value':int_value}, 
                                   device=device)
    
    #load residual decoder metadata i.e. skip flags
    sum_bits += rdac_decoder.res_decoder.load_metadata()
    for frame_idx in tqdm(range(0, frames)):            
        frame_idx_str = str(frame_idx).zfill(4)   
        if frame_idx in kp_decoder.reference_frame_idx:      # I-frame                    
            img_rec, ref_bits = image_decoder.decompress_reference(frame_idx) 
            sum_bits += ref_bits 
            if isinstance(img_rec, torch.Tensor):
                out_ref = tensor2frame(img_rec)
                out_ref.tofile(f_dec)
            else:
                img_rec.tofile(f_dec)

            with torch.no_grad(): 
                if isinstance(img_rec, torch.Tensor):
                    reference = img_rec.to(device)
                else:
                    reference = frame2tensor(img_rec).to(device)
                kp_reference = RDAC_Analysis_Model(reference) 

                ####
                kp_value = kp_reference['value']
                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                kp_value_frame=json.loads(kp_value_list)###20
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                kp_integer=str(kp_value_frame) 
                #Use the reference kp to reconstruct the subsequent frames
                kp_decoder.rec_sem.append(kp_integer)
        else:
            with torch.no_grad():
                kp_target_decoded = kp_decoder.decode_kp(frame_idx)                  
                # generated frame
                gene_start = time.time()

                prediction, res_bits = rdac_decoder.decode(reference,kp_target_decoded,kp_reference, frame_idx)
                sum_bits += res_bits
                gene_end = time.time()
                gene_time += gene_end - gene_start
                pred = tensor2frame(prediction)
                pred.tofile(f_dec)                              

                frame_index=str(frame_idx).zfill(4)
                bin_save=kp_output_dir+'/frame'+frame_index+'.bin'
                kp_bits=os.path.getsize(bin_save)*8
                sum_bits += kp_bits

    f_dec.close()     
    end=time.time()

    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'_RQP' +str(rate_idx)+'.txt', totalResult, fmt = '%.5f')            


