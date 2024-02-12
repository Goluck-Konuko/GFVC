

'''
An implementation of simple entropy coding using The standard Arithmetic encoder and 
PPM MODEL from:
https://github.com/nayuki/Reference-arithmetic-coding

The entropy coder here is optimized to re-use the frequency table for efficient
keypoint encoding.
TO-DO:
    :: Optimize the frequency table update procedure and PpmModel order selection
    :: Possibly introduce a NN likelihood estimation and use NeuralFrequencyTable table in
    :: the PPM model in place of the SimpleFrequencyTable
'''
import os
import json
import torch
import contextlib
import numpy as np
from copy import copy
from tempfile import mkstemp
from typing import Dict, Any, List
from .ppm_model import PpmModel
from .arithmetic_coder import ArithmeticEncoder,ArithmeticDecoder, BitOutputStream, BitInputStream


def read_bitstring(filepath:str):
    '''
    input: Path to a binary file
    returns: binary string of file contents
    '''
    with open(filepath, 'rb') as bt:
        bitstring = bt.read()
    return bitstring

class BasicEntropyCoder:
    '''A simple wrapper aroung the standard arithmetic codec'''
    def __init__(self):
        self.previous_res = None

    def quantize(self, tgt):
        return torch.round((tgt)).detach().cpu()

    def dequantize(self, tgt):
        return tgt
    
    
import time
class ResEntropyCoder:
    '''Using PPM context model and an arithmetic codec with persistent frequency tables'''
    def __init__(self, model_order=0, eof=256,q_step=128, out_path:str=None) -> None:
        super().__init__()
        self.history, self.dec_history = [], []
        self.eof = eof
        self.ppm_model = PpmModel(model_order, self.eof+1, self.eof)
        self.dec_ppm_model = PpmModel(model_order, self.eof+1, self.eof)

        self.inputs = []
        self.encoded = []
        self.previous_res = None
        self.first_temporal = False

        #output info
        self.res_output_dir = out_path
        self.q_step = q_step

    def mid_rise_quantizer(self, arr,levels=256):
        arr = np.array(arr)
        min_val, max_val = np.min(arr), np.max(arr)
        range_val = max_val - min_val
        step_size = max(1, range_val / levels)
        quantized_arr = np.round(np.floor((arr - min_val) / step_size) * step_size + min_val).astype(np.int16)
        return quantized_arr
    
    def mid_rise_dequantizer(self, quantized_arr, levels=256):
        quantized_arr = np.array(quantized_arr)
        min_val, max_val = np.min(quantized_arr), np.max(quantized_arr)
        range_val = max_val - min_val
        step_size = max(1, range_val / levels)
        dequantized_arr = (quantized_arr / step_size) * step_size + min_val
        return dequantized_arr
     
    def encode_res_2(self,res_latent: torch.tensor,frame_idx:int, levels=128):
        frame_idx = str(frame_idx).zfill(4)
        bin_file_path=self.res_output_dir+'/frame_res'+frame_idx+'.bin'
        
        shape = res_latent.shape
        r_flat = res_latent.cpu().flatten().numpy().astype(np.int16)
        r_flat = self.mid_rise_quantizer(r_flat,levels)
        # #create a compressed bitstring
        if self.previous_res is not None:
            r_delta = r_flat - self.previous_res
        else:
            r_delta = r_flat
        info_out  = self.compress_residual(r_delta, bin_file_path)
        #     r_hat = info_out['res']+self.previous_res
        # else:
        #     info_out  = self.compress(r_flat)
        #     r_hat = info_out['res']
        # self.previous_res = r_hat
        r_hat =  self.mid_rise_dequantizer(info_out['res_latent_hat'],levels)
        if self.previous_res is not None:
            res_hat = r_hat + self.previous_res
        else:
            res_hat = r_hat
        res_hat = np.reshape(res_hat, shape)
        info_out['res_latent_hat'] = res_hat
        return info_out
    
    def encode_res(self,res_latent, frame_idx:int=0):
        res_latent = self.convert_to_list(res_latent, frame_idx)
        #convert them to non-negative exp-golomb codes
        frame_idx = str(frame_idx).zfill(4)
        bin_file_path=self.res_output_dir+'/frame_res'+str(frame_idx)+'.bin'
        
        if self.previous_res is not None:
            r_delta = list(np.array(res_latent) - np.array(self.previous_res))
        else:
            r_delta = list(res_latent)
        r_delta = dataconvert_expgolomb(r_delta)

        info_out  = self.compress_residual(r_delta, bin_file_path)
        r_delta_hat = reverse_dataconvert_expgolomb(info_out['res_latent_hat'])

        if self.previous_res is not None:
            r_hat = np.array(r_delta_hat) + np.array(self.previous_res)
        else:
            r_hat = r_delta_hat
        self.previous_res = r_hat
        info_out['res_latent_hat'] = r_hat
        return info_out

    def compress_residual(self, res: np.ndarray, bin_file_path:str)->Dict[str, Any]:
        tmp, tmp_path = mkstemp("inp_temp.bin")
        #convert into a bitstring
        raw_bitstring = np.array(res).tobytes()
        #compact the bitstring
        with open(tmp_path, "wb") as raw:
            raw.write(raw_bitstring)
        # Set up encoder and model. In this PPM model, symbol 256 represents EOF;
        # its frequency is 1 in the order -1 context but its frequency
        # is 0 in all other contexts (which have non-negative order).
        #create an output path
        # tmp_out, tmp_out_path = mkstemp("out_temp.bin")
        enc_start = time.time()
        # Perform file compression
        with open(tmp_path, "rb") as inp, \
            contextlib.closing(BitOutputStream(open(bin_file_path, "wb"))) as bitout:
            enc = ArithmeticEncoder(32, bitout)
            while True:
                # Read and encode one byte
                symbol = inp.read(1)
                if len(symbol) == 0:
                    break
                symbol = symbol[0]
                self.encode_symbol(self.ppm_model, self.history, symbol, enc)
                self.ppm_model.increment_contexts(self.history, symbol)
                
                if self.ppm_model.model_order >= 1:
                    # Prepend current symbol, dropping oldest symbol if necessary
                    if len(self.history) == self.ppm_model.model_order:
                        self.history.pop()
                    self.history.insert(0, symbol)
            
            self.encode_symbol(self.ppm_model, self.history, self.eof, enc)  # EOF
            enc.finish()  # Flush remaining code bits
        enc_time = time.time()-enc_start
        bits= os.path.getsize(bin_file_path)*8
        #get the decoding
        dec_start = time.time()
        res_hat = self.decompress_residual(bin_file_path)
        dec_time = time.time()-dec_start
        os.close(tmp)
        os.remove(tmp_path)
        return {'bits': bits,'res_latent_hat': res_hat,
                'time':{'enc_time':enc_time,'dec_time':dec_time}}

    def decompress_residual(self, in_path:str):
        dec_p, dec_path = mkstemp("decoding.bin")
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            bitin = BitInputStream(inp)
            dec = ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = self.decode_symbol(dec, self.dec_ppm_model, self.dec_history)
                if symbol == self.eof:  # EOF symbol
                    break
                out.write(bytes((symbol,)))
                self.dec_ppm_model.increment_contexts(self.dec_history, symbol)
                
                if self.dec_ppm_model.model_order >= 1:
                    # Prepend current symbol, dropping oldest symbol if necessary
                    if len(self.dec_history) == self.dec_ppm_model.model_order:
                        self.dec_history.pop()
                    self.dec_history.insert(0, symbol)
        #read decoded_bytes
        with open(dec_path, 'rb') as dec_out:
            decoded_bytes = dec_out.read()
        
        kp_res = list(np.frombuffer(decoded_bytes, dtype=np.int16))
        ######
        #Temporary patch for buffer error here.. NEED TO FIX LATER
        # out = []
        # for idx in range(0, len(kp_res), 4):
        #     out.append(kp_res[idx])
        ######
        os.close(dec_p)
        os.remove(dec_path)
        return kp_res #out

    def encode_symbol(self, model, history, symbol, enc):
        # Try to use highest order context that exists based on the history suffix, such
        # that the next symbol has non-zero frequency. When symbol 256 is produced at a context
        # at any non-negative order, it means "escape to the next lower order with non-empty
        # context". When symbol 256 is produced at the order -1 context, it means "EOF".
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break
            else:  # ctx is not None
                if symbol != self.eof and ctx.frequencies.get(symbol) > 0:
                    enc.write(ctx.frequencies, symbol)
                    return
                # Else write context escape symbol and continue decrementing the order
                enc.write(ctx.frequencies, self.eof)
        # Logic for order = -1
        enc.write(model.order_minus1_freqs, symbol)

    def decode_symbol(self, dec, model, history):
        # Try to use highest order context that exists based on the history suffix. When symbol 256
        # is consumed at a context at any non-negative order, it means "escape to the next lower order
        # with non-empty context". When symbol 256 is consumed at the order -1 context, it means "EOF".
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break
            else:  # ctx is not None
                symbol = dec.read(ctx.frequencies)
                if symbol < self.eof:
                    return symbol
                # Else we read the context escape symbol, so continue decrementing the order
        # Logic for order = -1
        return dec.read(model.order_minus1_freqs)

    def convert_to_list(self,res_frame_latent: torch.Tensor, frame_idx)->str:
        #Extract the keypoints
        res_frame_list = res_frame_latent.cpu().flatten().numpy().astype(np.int16)
        res_frame_list = res_frame_list.tolist()
        with open(self.res_output_dir+'/frame_res'+str(frame_idx)+'.txt','w')as f:
            f.write("".join(str(res_frame_list).split()))  
        return res_frame_list

    def encode_metadata(self, metadata: List[int])->None:
        data = copy(metadata)
        bin_file=self.res_output_dir+'/metadata.bin'
        final_encoder_expgolomb(data,bin_file)     
        bits=os.path.getsize(bin_file)*8
        return bits
    
    def load_metadata(self)->None:
        bin_file=self.res_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        self.metadata = [int(i) for i in metadata]
        return os.path.getsize(bin_file)*8



class ResEntropyDecoder(BasicEntropyCoder):
    '''Using PPM context model and an arithmetic codec with persistent frequency tables'''
    def __init__(self, model_order=0, eof=256,q_step=128, input_path:str=None) -> None:
        super().__init__()
        self.history, self.dec_history = [], []
        self.eof = eof
        self.dec_ppm_model = PpmModel(model_order, self.eof+1, self.eof)

        self.inputs = []
        self.encoded = []
        self.previous_res = None
        self.first_temporal = False
        
        #output info
        self.res_output_dir = input_path
        self.q_step = q_step

        self.metadata = None #This is just a checklist of whether the residual at this frame idx is encoded or skipped
    
    def mid_rise_dequantizer(self, quantized_arr, levels=256):
        quantized_arr = np.array(quantized_arr)
        min_val, max_val = np.min(quantized_arr), np.max(quantized_arr)
        range_val = max_val - min_val
        step_size = max(1, range_val / levels)
        dequantized_arr = (quantized_arr / step_size) * step_size + min_val
        return dequantized_arr

    def decode_res(self,frame_idx:int=0, levels=128):
        #convert them to non-negative exp-golomb codes
        frame_idx = str(frame_idx).zfill(4)
        bin_file_path=self.res_output_dir+'/frame_res'+str(frame_idx)+'.bin'

        info_out  = self.decompress_residual(bin_file_path)
        # r_delta_hat = reverse_dataconvert_expgolomb(info_out['res_latent_hat'])
        r_delta_hat = self.mid_rise_dequantizer(info_out['res_latent_hat'],levels)
        if self.previous_res is not None:
            r_hat = np.array(r_delta_hat) + np.array(self.previous_res)
        else:
            r_hat = r_delta_hat
        self.previous_res = r_hat
        info_out['res_latent_hat'] = r_hat
        return info_out

    def decompress_residual(self, in_path:str):
        dec_start = time.time()
        dec_p, dec_path = mkstemp("decoding.bin")
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            bitin = BitInputStream(inp)
            dec = ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = self.decode_symbol(dec, self.dec_ppm_model, self.dec_history)
                if symbol == self.eof:  # EOF symbol
                    break
                out.write(bytes((symbol,)))
                self.dec_ppm_model.increment_contexts(self.dec_history, symbol)
                
                if self.dec_ppm_model.model_order >= 1:
                    # Prepend current symbol, dropping oldest symbol if necessary
                    if len(self.dec_history) == self.dec_ppm_model.model_order:
                        self.dec_history.pop()
                    self.dec_history.insert(0, symbol)
        #read decoded_bytes
        bits = os.path.getsize(in_path)*8
        with open(dec_path, 'rb') as dec_out:
            decoded_bytes = dec_out.read()
        
        kp_res = list(np.frombuffer(decoded_bytes, dtype=np.int16))
        ######
        #Temporary patch for buffer error here.. NEED TO FIX LATER
        # out = []
        # for idx in range(0, len(kp_res), 8):
        #     out.append(kp_res[idx])
        ######
        os.close(dec_p)
        os.remove(dec_path)
        dec_time = time.time()-dec_start
        return {'res_latent_hat':kp_res, 'dec_time':dec_time, 'bits': bits}

    def decode_symbol(self, dec, model, history):
        # Try to use highest order context that exists based on the history suffix. When symbol 256
        # is consumed at a context at any non-negative order, it means "escape to the next lower order
        # with non-empty context". When symbol 256 is consumed at the order -1 context, it means "EOF".
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break
            else:  # ctx is not None
                symbol = dec.read(ctx.frequencies)
                if symbol < self.eof:
                    return symbol
                # Else we read the context escape symbol, so continue decrementing the order
        # Logic for order = -1
        return dec.read(model.order_minus1_freqs)

    def load_metadata(self)->None:
        bin_file=self.res_output_dir+'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)   
        metadata = [int(i) for i in metadata]
        return metadata, os.path.getsize(bin_file)*8

def dataconvert_expgolomb(symbol):
    '''Creates non-negative interger list'''
    out = np.array(symbol)
    # print(out[:10])
    mask = create_mask(out, lambda x: x<=0).astype(np.int8)
    neg_val = -(out*mask*2)
    pos_val = (out*2-1)*(1-mask)
    return list(neg_val+pos_val)


def reverse_dataconvert_expgolomb(symbol_list):
    """
    Reverses the effect of dataconvert_expgolomb function.
    """
    mask= create_mask(np.array(symbol_list), lambda x: x%2==0)
    neg_val = (-1*np.array(symbol_list)//2)*mask 
    pos_val = ((np.array(symbol_list)+1)//2)*(1-mask)
    return list(neg_val+pos_val)

def create_mask(array, condition):
    """
    Creates a boolean mask for a NumPy array based on a given condition.

    Parameters:
    - array: NumPy array
    - condition: A boolean condition (e.g., array > 0)

    Returns:
    - mask: Boolean array of the same shape as the input array
    """
    mask = condition(array)
    return mask


##encoder: input:Inter-frame residual output: bin file
def final_encoder_expgolomb(datares,outputfile, MODEL_ORDER = 0):
     # Must be at least -1 and match ppm-decompress.py. Warning: Exponential memory usage at O(257^n).
    with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:  #arithmeticcoding.
    
        enc = ArithmeticEncoder(256, bitout) #########arithmeticcoding.
        #print(enc)
        model = PpmModel(MODEL_ORDER, 3, 2)  ##########ppmmodel.
        #print(model)
        history = []

        # Read and encode one byte
        symbol=datares
        #print(symbol)
        
        # 数值转换
        symbol = dataconvert_expgolomb(symbol)
        #print(symbol)
        
        # 二进制0/1字符串
        symbollist = list_binary_expgolomb(symbol)
        #print(int(symbollist))
             
        
        ###依次读取这串拼接的数，从左到右输出
        for ii in symbollist:
            #print(ii)
            i_number=int(ii)
            
            encode_symbol(model, history, i_number, enc)

            model.increment_contexts(history, i_number)
            if model.model_order >= 1:
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, i_number) ###########
            #print(history)
        encode_symbol(model, history, 2, enc)  # EOF ##########
        enc.finish()  #        


###正整数10进制转二进制一元码，并且将其顺序合并成0/1二进制字符串
def list_binary_expgolomb(symbol):
    ### 10进制转为2进制
    for i in range(len(symbol)):
        n = symbol[i]
        symbol[i]=exponential_golomb_encode(n)
                    
    #print(symbol)
    
    ##将list的所有数字拼接成一个
    m='1' ##1:标识符
    for x in symbol:
        m=m+str(x)
    #print(int(m))
    return m

def exponential_golomb_encode(n):
    unarycode = ''
    golombCode =''
    ###Quotient and Remainder Calculation
    groupID = np.floor(np.log2(n+1))
    temp_=groupID
    #print(groupID)
    
    while temp_>0:
        unarycode = unarycode + '0'
        temp_ = temp_-1
    unarycode = unarycode#+'1'

    index_binary=bin(n+1).replace('0b','')
    golombCode = unarycode + index_binary
    return golombCode


##### Encode symbol based PPM MODEL 
def encode_symbol(model, history, symbol, enc):
    # Try to use highest order context that exists based on the history suffix, such
    # that the next symbol has non-zero frequency. When symbol 256 is produced at a context
    # at any non-negative order, it means "escape to the next lower order with non-empty
    # context". When symbol 256 is produced at the order -1 context, it means "EOF".
    for order in reversed(range(len(history) + 1)):
        #print(order)
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            #print(ctx)
            if ctx is None:
                break
        else:# ctx is not None
            if symbol != 2 and ctx.frequencies.get(symbol) > 0: ##############
                enc.write(ctx.frequencies, symbol)
                return
            # Else write context escape symbol and continue decrementing the order
            enc.write(ctx.frequencies, 2) ##############
    # Logic for order = -1
    enc.write(model.order_minus1_freqs, symbol)

###正整数10进制转二进制一元码，并且将其顺序合并成0/1二进制字符串
def list_binary_unary(symbol):
    ### 10进制转为2进制
    for i in range(len(symbol)):                
        n = symbol[i]
        symbol[i]=unary(n)
        
    #print(symbol)
    
    ##将list的所有数字拼接成一个
    m=''
    for x in symbol:
        m=m+str(x)
    #print(int(m))
    return m

############################# unary code#######################################
###10进制数转二进制数（一元码）
def unary(q):
    code1 = []
    for i in range(q):
        code1.append(1)
    code1.append(0)
    code2 = [str(i) for i in code1]
    code = "".join(code2)
    return code


################################ 0-order exponential coding###############
## final decode: input: bin file; output: the 0/1 value
def final_decoder_expgolomb(inputfile,MODEL_ORDER = 0):
    # Must be at least -1 and match ppm-compress.py. Warning: Exponential memory usage at O(257^n).
    # Perform file decompression
    with open(inputfile, "rb") as inp:
        bitin = BitInputStream(inp)  #arithmeticcoding.

        dec = ArithmeticDecoder(256, bitin) ##############arithmeticcoding.
        model = PpmModel(MODEL_ORDER, 3, 2) #######ppmmodel.
        history = []
        
        datares_rec=[]
                    
        while True:
            symbol = decode_symbol(dec, model, history)
            if symbol ==2:
                break

            model.increment_contexts(history, symbol)
            datares_rec.append(symbol)
            if model.model_order >= 1:
                # Prepend current symbol, dropping oldest symbol if necessary
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, symbol) ####
        return datares_rec


##### Decode symbol based PPM MODEL 

def decode_symbol(dec, model, history):
    # Try to use highest order context that exists based on the history suffix. When symbol 256
    # is consumed at a context at any non-negative order, it means "escape to the next lower order
    # with non-empty context". When symbol 256 is consumed at the order -1 context, it means "EOF".
    for order in reversed(range(len(history) + 1)):
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:
                break
        else:  # ctx is not None
            symbol = dec.read(ctx.frequencies)
            if symbol < 2: #############
                return symbol
                # Else we read the context escape symbol, so continue decrementing the order
    # Logic for order = -1
    return dec.read(model.order_minus1_freqs)


import itertools
def data_convert_inverse_expgolomb(datares_rec):
    ##按照0-order所定义的解码方式进行数字划分切割
    list_new= expgolomb_split(datares_rec)
    #print(list_new)
    #print(len(list_new))
    
    true_ae_number=[]
    for subnum in range(len(list_new)):
        num=exponential_golomb_decode(list_new[subnum])
        #print(num)
        true_ae_number.append(num)
    #print(true_ae_number)

    #把解码后的残差变会原先的数值 （0，1，2，3，4——》0，1，-1，2，-2)
    for ii in range(len(true_ae_number)):
        if true_ae_number[ii] ==0:
            true_ae_number[ii]=0
        elif  true_ae_number[ii] >0 and true_ae_number[ii] %2 ==0:
            true_ae_number[ii]=-(int(true_ae_number[ii]/2))
        else:
            true_ae_number[ii]=int((true_ae_number[ii]+1)/2)
    #print(true_ae_number)
    return true_ae_number


### golombcode : 00100 real input[0,0,1,0,0]
def exponential_golomb_decode(golombcode):

    code_len=len(golombcode)

    ###Count the number of 1's followed by the first 0
    m= 0 ### 
    for i in range(code_len):
        if golombcode[i]==0:
            m=m+1
        else:
            ptr=i  ### first 0
            break

    offset=0
    for ii in range(ptr,code_len):
        num=golombcode[ii]
        offset=offset+num*(2**(code_len-ii-1))
    decodemum=offset-1
    
    return decodemum


def expgolomb_split(expgolomb_bin_number):
    x_list=expgolomb_bin_number
    
    del(x_list[0]) 
    x_len=len(x_list)
    
    sublist=[]
    while (len(x_list))>0:

        count_number=0
        i=0
        if x_list[i]==1:
            sublist.append(x_list[0:1])
            del(x_list[0])            
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number=count_number+num_times_zeros[0]
            sublist.append(x_list[0:(count_number*2+1)])
            del(x_list[0:(count_number*2+1)])
    return sublist


