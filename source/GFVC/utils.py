import yaml
from tqdm import tqdm
import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Protocol


def raw_reader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
            for i in sublist:
                alist.append (i)
        except TypeError: #不能迭代的就是直接取出放入alist
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):#判断是否还有列表
            a =+ 1
            break
    if a==1:
        return printlist(alist) #还有列表，进行递归
    if a==0:
        return alist  
    
    
###只能读取yuv420 8bit 
def yuv420_to_rgb444(yuvfilename, W, H, startframe, totalframe, show=False, out=False):
    # 从第startframe（含）开始读（0-based），共读totalframe帧
    arr = np.zeros((totalframe,H,W,3), np.uint8)
    
    plt.ion()
    with open(yuvfilename, 'rb') as fp:
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(8 * seekPixels) #跳过前startframe帧
        for i in range(totalframe):
            #print(i)
            oneframe_I420 = np.zeros((H*3//2,W),np.uint8)
            for j in range(H*3//2):
                for k in range(W):
                    oneframe_I420[j,k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420,cv2.COLOR_YUV2BGR_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(5)
            if out:
                outname = yuvfilename[:-4]+'_'+str(startframe+i)+'.png'
                cv2.imwrite(outname,oneframe_RGB)
            arr[i] = cv2.cvtColor(oneframe_RGB,cv2.COLOR_BGR2RGB)
    return arr


def to_tensor(frame: np.ndarray)->torch.Tensor:
    return torch.tensor(frame[np.newaxis].astype(np.float32))

def read_config_file(config_path):
    '''Simply reads a yaml configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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

def frame2tensor(frame: np.ndarray)->torch.Tensor:
    frame = frame/255.0
    return torch.tensor(frame[np.newaxis].astype(np.float32))

def tensor2frame(frame:torch.Tensor)->np.ndarray:
    pred = frame.detach().cpu().numpy()[0]
    return (pred*255).astype(np.uint8) 

import os
from image_codecs.lic import LICEnc, LICDec
from skimage.transform import resize
from typing import  List
from image_codecs.lic import LICEnc
class ReferenceImageCoder:
    def __init__(self,img_output_dir:str,qp:int, iframe_format:str, width:int=256, height:int=256,
                 codec_name:str='vtm', model_name:str='cheng2020attn',device:str='cpu'):
        self.iframe_format = iframe_format
        self.qp = qp
        self.width=width
        self.height=height
        self.img_output_dir = img_output_dir
        self.codec_name = codec_name
        self.device = device
        if self.codec_name == 'lic':
            self.lic_enc = LICEnc(self.img_output_dir,model_name=model_name, quality=int(self.qp))
        else:
            self.lic_enc = None


    def vtm_yuv_compress(self, frame_idx_str:str)->tuple:
        # wtite ref and cur (rgb444) to file (yuv420) 
              
        os.system("./image_codecs/vtm/encode.sh "+self.img_output_dir+'frame'+frame_idx_str+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################

        bin_file=self.img_output_dir+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_rgb=yuv420_to_rgb444(self.img_output_dir+'frame'+frame_idx_str+'_rec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_rgb[0]
        img_rec = img_rec
        return img_rec, bits               
    
    def vtm_rgb_compress(self, frame_idx_str)->tuple:
        # wtite ref and cur (rgb444) 
        os.system("./image_codecs/vtm/encode_rgb444.sh "+self.img_output_dir+'frame'+frame_idx_str+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################
        
        bin_file=self.img_output_dir+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_rec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1                  
        return img_rec, bits  
    
    def compress(self, reference_frame: List[np.ndarray], frame_idx:int)->tuple:
        frame_idx_str= str(frame_idx).zfill(4) 
        if self.codec_name == 'vtm':
            if self.iframe_format=='YUV420':
                f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_org.yuv','w')
                img_input_rgb = cv2.merge(reference_frame)
                img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                img_input_yuv.tofile(f_temp)
                f_temp.close()   

                img_rec, bits = self.vtm_yuv_compress(frame_idx_str)
            elif self.iframe_format=='RGB444':
                f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_org.rgb','w')
                img_input_rgb = cv2.merge(reference_frame)
                img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
                img_input_rgb.tofile(f_temp)
                f_temp.close()

                img_rec, bits = self.vtm_rgb_compress(frame_idx_str)
            else:
                raise NotImplementedError(f"Coding in format {self.frame_format} not implemented")
        else:
            ref = torch.tensor(np.array(reference_frame)/255.0, dtype=torch.float32).unsqueeze(0)
            dec_info = self.lic_enc.compress(ref, frame_idx_str)
            img_rec = dec_info['x_hat']
            bits = dec_info['bits']
        return img_rec, bits
  

class RefereceImageDecoder:
    '''Wrapper for Image decoders [VTM, BPG, LICs]'''
    def __init__(self,img_input_dir:str,qp:int=4,iframe_format:str='YUV420', width:int=256,height:int=256, dec_name='vtm', model_name="cheng2020attn",device='cpu'):
        self.img_input_dir = img_input_dir
        self.iframe_format = iframe_format
        self.width = width
        self.height = height
        self.qp = qp
        self.device = device
        #Learned image decoder
        self.dec_name = dec_name
        if self.dec_name == 'lic':
            self.lic_dec = LICDec(self.img_input_dir,model_name=model_name, quality=self.qp, device=self.device)
        else:
            self.lic_dec = None

    def vtm_yuv_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=self.img_input_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.img_input_dir+'frame'+str(frame_idx)+'_dec.yuv', self.width, self.height, 0, 1, False, False)                  
        return rec_ref_yuv[0]  , bits                                    

    def vtm_rgb_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode_rgb444.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=self.img_input_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        f_temp=open(self.img_input_dir+'frame'+str(frame_idx)+'_dec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB            
        return img_rec, bits
    
    def decompress(self, frame_idx:int):
        frame_idx_str = str(frame_idx).zfill(4)
        if self.dec_name == 'vtm':
            if self.iframe_format == 'YUV420':
                img_rec, bits = self.vtm_yuv_decompress(frame_idx_str)
            elif self.iframe_format == 'RGB444':
                img_rec, bits = self.vtm_rgb_decompress(frame_idx_str)
            else:
                raise NotImplementedError(f"Frame format '{self.iframe_format}' not implemented!")
        else:
            dec_info = self.lic_dec.decompress(frame_idx_str)
            img_rec = dec_info['x_hat']
            bits = dec_info['bits']
        return img_rec, bits
