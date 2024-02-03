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

class RefereceImageDecoder:
    def __init__(self,img_input_dir:str,frame_format:str='YUV420', width:int=256,height:int=256):
        self.img_input_dir = img_input_dir
        self.frame_format = frame_format
        self.width = width
        self.height = height

    def vtm_yuv_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=self.img_input_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.img_input_dir+'frame'+str(frame_idx)+'_dec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_yuv[0]
        img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
        img_rec.tofile(f_dec)                         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1  
        return img_rec, bits                                    

    def vtm_rgb_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode_rgb444.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=dir_enc+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        f_temp=open(self.img_input_dir+'frame'+str(frame_idx)+'_dec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec.tofile(f_dec) 
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1       
        return img_rec, bits
    
    def decompress(self, frame_idx:int):
        if self.frame_format == 'YUV420':
            img_rec, bits = self.vtm_yuv_decompress(frame_idx)
        elif self.frame_format == 'RGB444':
            img_rec, bits = self.vtm_rgb_decompress(frame_idx)
        else:
            raise NotImplementedError(f"Frame format '{self.frame_format}' not implemented!")
        return img_rec, bits

class KPDecoder:
    def __init__(self,kp_input_dir:str, q_step:int=64):
        self.kp_input_dir = kp_input_dir
        self.q_step = q_step
        self.rec_sem = []

    def decode_kp(self, frame_idx:int):
        frame_index=str(frame_idx).zfill(4)
        bin_file=kp_input_dir+'/frame'+frame_index+'.bin' 
        bits=os.path.getsize(bin_file)*8          
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quanzation
        kp_difference_dec=[i/self.q_step for i in kp_difference]
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  

        kp_previous=self.rec_sem[frame_idx-1]    #json.loads(str(seq_kp_integer[frame_idx-1]))      
        kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer,kp_value=listformat_kp_DAC(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer) 
        return kp_value, bits

    def get_kp_list(self, kp_value:torch.Tensor, frame_idx:int)->List[str]:
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        with open(self.kp_input_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
            f.write(kp_value_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame
    
    def read_metadata(self)->None:
        bin_file=self.kp_input_dir+'metadata.bin'
        bits=os.path.getsize(bin_file)*8  

        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        reference_frame_idx = [int(i) for i in metadata]
        return reference_frame_idx, bits

class AdaptiveDecoder:
    def __init__(self) -> None:
        pass

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

    kp_input_dir =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'   

    dir_dec=model_dirname+'/dec/'
    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

    #This folder is created by the encoder
    enc_input_dir =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
 
    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)        


    f_dec=open(decode_seq,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
    sum_bits = 0

    ref_decoder = RefereceImageDecoder(enc_input_dir,Iframe_format)
    kp_decoder = KPDecoder(kp_input_dir,Qstep)
    ref_frame_idx, mt_bits = kp_decoder.read_metadata()
    sum_bits += mt_bits
    for frame_idx in tqdm(range(0, frames)):             
        if frame_idx in ref_frame_idx:      # I-frame                      
            img_rec, ref_bits = ref_decoder.decompress(frame_idx)
            sum_bits+=ref_bits
            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.to(device)    # require GPU | changed to use cpu when GPU not available

                kp_reference = DAC_Analysis_Model(reference) 

                ####
                kp_value_frame = kp_decoder.get_kp_list(kp_reference['value'], frame_idx)
                kp_decoder.rec_sem.append(kp_value_frame)
        else:
            kp_value, kp_bits = kp_decoder.decode_kp(frame_idx)                   

            kp_inter_frame={}
            kp_value=json.loads(kp_value)
            kp_inter_frame['value']=torch.Tensor(kp_value).reshape((1,10,2)).to(device)          

            # generated frame
            gene_start = time.time()

            prediction = make_DAC_prediction(reference, kp_inter_frame, kp_reference, DAC_Synthesis_Model, cpu=cpu) #######################

            gene_end = time.time()
            gene_time += gene_end - gene_start
            pre=(prediction*255).astype(np.uint8)  
            pre.tofile(f_dec)                              
            sum_bits += kp_bits

    f_dec.close()     
    end=time.time()

    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'.txt', totalResult, fmt = '%.5f')            


