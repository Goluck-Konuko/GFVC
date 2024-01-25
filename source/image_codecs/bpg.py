'''
This code is adapted from the benchmark scripts used in the compressai library for 
learning based image codecs 
(https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py)
[Thus relevant license and permissions are transferred here].

Simplified and optimized to use for reference frame coding in the animation-based
video codecs by Goluck Konuko [https://github.com/Goluck-Konuko]
'''
import time
import torch
import os,sys
import imageio.v3 as imageio
import subprocess
import numpy as np
from tempfile import mkstemp
from typing import Dict, Any, Union, List

#TO-DO :
#Implement a decoder that takes a bpg bitstring and decodes the frame

def read_bitstring(filepath:str):
    '''
    input: Path to a binary file
    returns: binary string of file contents
    '''
    with open(filepath, 'rb') as bt:
        bitstring = bt.read()
    return bitstring

class BPGEnc:
    """BPG from Fabrice Bellard."""
    def __init__(self, color_mode="rgb", encoder="x265",
                        subsampling_mode="420", bit_depth='8', 
                        encoder_path='bpgenc', decoder_path='bpgdec',
                        out_filepath:str=None):
        self.fmt = ".bpg"
        self.color_mode = color_mode
        self.encoder = encoder
        self.subsampling_mode = subsampling_mode
        self.bitdepth = bit_depth
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        #output file paths
        self.out_filepath = out_filepath

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    def _run_impl(self, in_filepath:str, quality:int, frame_idx:int=0)->Dict[str, Any]:
        if not self.out_filepath:
            fd0, dec_png_filepath = mkstemp(suffix=".png")
            fd1, out_bits_filepath = mkstemp(suffix=self.fmt)
        else:
            dec_png_filepath = os.path.join(self.out_filepath,f"{frame_idx}.png")
            out_bits_filepath = os.path.join(self.out_filepath,f"{frame_idx}.bpg")

        # Encode
        enc_start = time.time()
        run_command(self._get_encode_cmd(in_filepath, quality, out_bits_filepath))
        enc_time = time.time()-enc_start
        bits = os.path.getsize(out_bits_filepath)*8
        bitstring = read_bitstring(out_bits_filepath)
        # Decode
        dec_start = time.time()
        run_command(self._get_decode_cmd(out_bits_filepath, dec_png_filepath))
        dec_time = time.time()-dec_start
        # Read image
        rec = read_image(dec_png_filepath)
        if not self.out_filepath:
            #close temporary paths from file system
            os.close(fd0)
            os.remove(dec_png_filepath)
            os.close(fd1)
            os.remove(out_bits_filepath)
        out = {
            'bitstring': bitstring,
            'bits': bits,
            'x_hat': np.array(rec),
            'time':{'enc_time': enc_time, 'dec_time': dec_time}
        }
        return out

    def run(self,input_image: Union[np.ndarray,torch.Tensor],quality: int, frame_idx:int=0)->Dict[str, Any]:
        if isinstance(input_image, torch.Tensor):
            #detach and convert to array
            input_image = input_image.cpu().permute(0,2,3,1)[0].numpy()
            input_image = (255*input_image).astype(np.uint8)

        if not self.out_filepath:
            #create a temporary file for the input image
            fd_in, png_in_filepath = mkstemp(suffix=".png")
        else:
            png_in_filepath = os.path.join(self.out_filepath, f"org_{frame_idx}.png")
        imageio.imsave(png_in_filepath, input_image)
        in_file = png_in_filepath

        #compression
        info = self._run_impl(in_file, quality, frame_idx)
        if not self.out_filepath:
            os.close(fd_in)
            os.remove(png_in_filepath)
        return info

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"



    def _get_encode_cmd(self, in_filepath:str, quality:int, out_filepath:str)->List[str]:
        if not 0 <= quality <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath)->List[str]:
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd


class BPGDec:
    """BPG from Fabrice Bellard."""
    def __init__(self, decoder_path:str='bpgdec'):
        self.fmt = ".bpg"
        self.decoder_path = decoder_path

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    def _run_impl(self, in_filepath:str)->Dict[str, Any]:
        fd0, png_filepath = mkstemp(suffix=".png")
        # Decode
        dec_start = time.time()
        run_command(self._get_decode_cmd(in_filepath, png_filepath))
        dec_time = time.time()-dec_start
        # Read image
        rec = read_image(png_filepath)
        os.close(fd0)
        os.remove(png_filepath)
        out = {'bits': os.path.getsize(in_filepath)*8,
                'x_hat': np.array(rec),
                'time':dec_time}
        return out

    def run(self,in_filepath: str) ->Dict[str, Any]:
        #Decompression
        dec_info = self._run_impl(in_filepath)
        return dec_info

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.decoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.decoder_path)}"

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd

def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return imageio.imread(filepath)

def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4] 


if __name__ == "__main__":
    img_n = 8
    img = f"imgs/{img_n}.png"
    img_arr = imageio.imread(img)
    qp = 30
    bpg = BPGEnc()
    #pass img as np.ndarray :: #Images must be uint8
    out  = bpg.run(img_arr,qp)
    imageio.imsave(f"{img_n}_{qp}_decoded.png", out['decoded'])