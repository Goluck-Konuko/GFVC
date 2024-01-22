import os
import time
import json
import torch
import numpy as np
from GFVC.utils import *
from GFVC.DAC_utils import *
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import List



class ReferenceImageDecoder:
    def __init__(self, dir_enc:str, qp:int=22, height: int=256, width: int=256, format='YUV420') -> None:
        self.dir_enc = dir_enc
        self.qp = qp
        self.height= height
        self.width = width
        self.format = format

    def decompress_reference_yuv(self, frame_idx:int):
        frame_idx_str = str(frame_idx)
        os.system("./vtm/decode.sh "+self.dir_enc+'frame'+frame_idx_str)
        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8

        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.dir_enc+'frame'+frame_idx_str+'_dec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_yuv[0]
        img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
        img_rec.tofile(f_dec)                         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1
        return img_rec , bits          
                    
    def decompress_reference_rgb(self, frame_idx: int)->np.ndarray:
        frame_idx_str = str(frame_idx)
        os.system("./vtm/decode_rgb444.sh "+self.dir_enc+'frame'+frame_idx_str)
        bin_file=self.dir_enc+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8

        f_temp=open(self.dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec.tofile(f_dec) 
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1       
        return img_rec, bits   

    def decompress_reference(self,frame_idx:int)->np.ndarray:
        if self.format == 'YUV420':
            img_rec, bits = self.decompress_reference_yuv(frame_idx)
        else:
            img_rec, bits = self.decompress_reference_rgb(frame_idx)
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

def to_tensor(frame: np.ndarray)->torch.Tensor:
    return torch.tensor(frame[np.newaxis].astype(np.float32))


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
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    Iframe_format=opt.Iframe_format
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    device = opt.device
    if device =='cuda' and torch.cuda.is_available():
        cpu = False
    else:
        cpu = True
        device = 'cpu'
    
    ## DAC
    DAC_config_path='./GFVC/DAC/checkpoint/DAC-256.yaml'
    DAC_checkpoint_path='./GFVC/DAC/checkpoint/DAC-checkpoint.pth.tar'         
    DAC_Analysis_Model, DAC_Synthesis_Model = load_DAC_checkpoints(DAC_config_path, DAC_checkpoint_path, cpu=cpu)
    modeldir = 'DAC' 
    model_dirname='../experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
###################################################

    kp_output_dir =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'   

    dir_dec=model_dirname+'/dec/'
    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm     

    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)        


    f_dec=open(decode_seq,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    image_decoder = ReferenceImageDecoder(dir_enc, qp=QP, height=height,width=width, format=Iframe_format)
    kp_decoder = KPDecoder(kp_output_dir,q_step=Qstep, device=device)
    kp_decoder.load_metadata()

    for frame_idx in tqdm(range(0, frames)):            
        frame_idx_str = str(frame_idx).zfill(4)   
        if frame_idx in kp_decoder.reference_frame_idx:      # I-frame                      
            img_rec, bits = image_decoder.decompress_reference(frame_idx)                                     
            with torch.no_grad(): 
                reference = to_tensor(img_rec).to(device)
                kp_reference = DAC_Analysis_Model(reference) 

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
            kp_target_decoded = kp_decoder.decode_kp(frame_idx)                  
            # generated frame
            gene_start = time.time()

            prediction = make_DAC_prediction(reference, kp_target_decoded, kp_reference, DAC_Synthesis_Model, cpu=cpu) #######################

            gene_end = time.time()
            gene_time += gene_end - gene_start
            pre=(prediction*255).astype(np.uint8)  
            pre.tofile(f_dec)                              

            frame_index=str(frame_idx).zfill(4)
            bin_save=kp_output_dir+'/frame'+frame_index+'.bin'
            bits=os.path.getsize(bin_save)*8
        sum_bits += bits

    f_dec.close()     
    end=time.time()

    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'.txt', totalResult, fmt = '%.5f')            


