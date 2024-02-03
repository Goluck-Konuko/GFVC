##A wrapper for SOTA Learned Image Codecs from the compressAI library
import os
import time
import torch
from typing import Dict, Any, List
from compressai.zoo import cheng2020_attn

lic_models = {
    'cheng2020attn': cheng2020_attn,
}


def count_bytes(strings):
    total_bytes = 0
    for s in strings:
        total_bytes += len(s[-1])
    return total_bytes

class LICEnc:
    def __init__(self,dec_dir:str,model_name:str='cheng2020attn',metric='mse', quality:int=6, device:str='cpu') -> None:
        self.codec = lic_models[model_name](quality=quality, pretrained=True).to(device)
        self.codec.eval()
        self.dec_dir = dec_dir
        os.makedirs(self.dec_dir, exist_ok=True)

    def compress(self, img, frame_idx:int=0)->Dict[str, Any]:
        enc_start = time.time()
        enc_info = self.codec.compress(img)
        enc_time = time.time()-enc_start
        bits = write_bitstring(enc_info, self.dec_dir, frame_idx)

        dec_start = time.time()
        dec_info, d_bits = read_bitstring(self.dec_dir, frame_idx)
        dec_info = self.codec.decompress(**dec_info)
        dec_time = time.time()-dec_start
        rec = dec_info['x_hat']
        return {'bits': bits,'x_hat': rec,'time':{'enc_time': enc_time, 'dec_time': dec_time}}

class LICDec:
    def __init__(self,dec_dir:str,model_name:str='cheng2020_attn', quality:int=6,metric:str='mse', device:str='cpu') -> None:
        self.codec = lic_models[model_name](quality=quality,metric=metric, pretrained=True).to(device)
        self.dec_dir = dec_dir
    
    def decompress(self, frame_idx:str):
        dec_start = time.time()
        #read bitstring and metadata
        bit_info, bits = read_bitstring(self.dec_dir, frame_idx)
        dec_info = self.codec.decompress(**bit_info)
        dec_time = time.time()-dec_start
        rec = dec_info['x_hat']
        return {'bits': bits,'x_hat': rec,'time':dec_time}

def write_bitstring(enc_info:Dict[str,Any], dec_dir:str, frame_idx:int)->float:
    strings = enc_info['strings']
    shape = f"{enc_info['shape'][0]}_{enc_info['shape'][1]}_"
    out_path = dec_dir+'/'+shape+str(frame_idx)
    y_string = strings[0][0]
    z_string = strings[1][0]

    #write both strings to binary file
    with open(f"{out_path}_y.bin", 'wb') as y:
        y.write(y_string)
    bits = os.path.getsize(f"{out_path}_y.bin")*8
    with open(f"{out_path}_z.bin", 'wb') as z:
        z.write(z_string)
    bits += os.path.getsize(f"{out_path}_z.bin")*8
    return bits

import re
import numpy as np
def read_bitstring(dec_dir:str, frame_idx:int):
    #locate the correct files in the dec_dir
    #The process is clumsy but should work for now
    bin_files = [x for x in os.listdir(dec_dir) if x.endswith('.bin')]
    y_pattern = re.compile(r'_{}_y\.bin'.format(frame_idx))
    z_pattern = re.compile(r'_{}_z\.bin'.format(frame_idx))
    y_file = [file for file in bin_files if y_pattern.search(file)][-1]
    z_file = [file for file in bin_files if z_pattern.search(file)][-1]

    rec_shape = y_file.split("_")
    shape = [int(rec_shape[0]), int(rec_shape[0])]
    with open(f"{dec_dir}/{y_file}", 'rb') as y_out:
        y_string = y_out.read()
    bits = os.path.getsize(f"{dec_dir}/{y_file}")*8

    with open(f"{dec_dir}/{z_file}", 'rb') as z_out:
        z_string = z_out.read()
    bits += os.path.getsize(f"{dec_dir}/{z_file}")*8
    dec_info = {'strings': [[y_string],[z_string]], 'shape':shape}
    return dec_info, bits

if __name__ == "__main__":
    img = torch.randn((1,3,256,256))
    dir_dec = "output"
    # os.makedirs(dir_dec, exist_ok=True)
    # codec = LICEnc(dir_dec)
    codec = LICDec(dir_dec)
    out = codec.decompress(0)
    print(out.keys())