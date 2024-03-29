import subprocess
import imageio
import os
import time
import shutil
import numpy as np

class VVC_Encoder:
    '''
        VVC VTM CODEC WRAPPER
    '''
    def __init__(self,sequence = None, qp = 50,fps=30,frame_dim=[256,256],config='GFVC/HDAC/bl_codecs/VVC_VTM_22_2/cfg_template.cfg', 
        seq_name='0',log_path='vvc_logs/'):
        self.qp = qp
        self.fps = fps
        self.n_frames = len(sequence)
        self.frame_dim = frame_dim #f"{frame_dim[0]}x{frame_dim[1]}"
        self.skip_frames = 0
        self.intra_period = -1 
        self.config_path = config
        self.input = sequence
        self.out_path = log_path
        self.seq_name = seq_name
        #inputs
        self.in_mp4_path = self.out_path+f'/in.mp4'
        self.in_yuv_path = self.out_path+f'/in.yuv'
        #outputs
        self.ostream_path = f"{self.out_path}.bin"
        self.dec_yuv_path = self.out_path+f'/out.yuv'
        self.dec_mp4_path = self.out_path+f'/out.mp4'
        #logging file
        self.log_path =  self.out_path+f'/out.log'
        
        self.create_outpaths()
        self.config_out_path = f"{self.out_path}/codec_config.cfg"
        self._create_config()

        #create yuv video
        if not os.path.exists(self.in_mp4_path):
            self._create_mp4()
        
        if not os.path.exists(self.in_yuv_path):
            self._mp4_2_yuv()
        self.compress = True
        if os.path.exists(self.dec_mp4_path):
            self.compress = False

    def create_outpaths(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        if not os.path.exists(self.out_path+f"out/logs"):
            os.makedirs(self.out_path+f"out/logs")

        if not os.path.exists(self.out_path+f"in"):
            os.makedirs(self.out_path+f"in")

        if not os.path.exists(self.out_path+f"out/bin/{self.seq_name}/"):
            os.makedirs(self.out_path+f"out/bin/{self.seq_name}/")
        if not os.path.exists(self.out_path+f"out/yuv/{self.seq_name}/"):
            os.makedirs(self.out_path+f"out/yuv/{self.seq_name}/")

        if not os.path.exists(self.out_path+f"out/mp4/{self.seq_name}/"):
            os.makedirs(self.out_path+f"out/mp4/{self.seq_name}/")
        
    def _create_config(self):
        '''
            Creates a configuration file for HEVC encoder
        '''
        with open(self.config_path, 'r') as file:
            template = file.read()
        #print(template)
        template = template.replace('inputYUV', str(self.in_yuv_path))
        template = template.replace('outStream', str(self.ostream_path))
        template = template.replace('outYUV', str(self.dec_yuv_path))
        template = template.replace('inputW', str(self.frame_dim[0]))
        template = template.replace('inputH', str(self.frame_dim[1]))
        template = template.replace('inputNrFrames', str(self.n_frames))
        template = template.replace('intraPeriod', str(self.intra_period))
        template = template.replace('inputSkip', str(self.skip_frames))
        template = template.replace('inputFPS', str(self.fps))
        template = template.replace('setQP', str(self.qp))
        with open(self.config_out_path, 'w+') as cfg_file:
            cfg_file.write(template)


    def _create_mp4(self):
        writer = imageio.get_writer(self.in_mp4_path, format='FFMPEG', mode='I',fps=self.fps, codec='libx264',pixelformat='yuv420p', quality=10)
        for frame in self.input:
            writer.append_data(frame)
        writer.close()	
		
    def _mp4_2_yuv(self):
        #check for yuv video in target directory
        subprocess.call(['ffmpeg','-nostats','-loglevel','error','-i',self.in_mp4_path,self.in_yuv_path, '-r',str(self.fps)])
		
    def _yuv_2_mp4(self):
        cmd = ['ffmpeg','-nostats','-loglevel','error', '-f', 'rawvideo', '-pix_fmt','yuv420p10le','-s:v',
               f"{self.frame_dim[0]}x{self.frame_dim[1]}", '-r', str(self.fps), '-i', self.dec_yuv_path,  self.dec_mp4_path]
        subprocess.call(cmd)
        frames = imageio.mimread(self.dec_mp4_path, memtest=False)
        return np.array(frames)

    def _get_metrics(self):
        original = imageio.mimread(self.in_mp4_path, memtest=False)
        decoded = imageio.mimread(self.dec_mp4_path, memtest=False)
        metrics = self.monitor.compute_metrics(original[:len(decoded)], decoded)
        return metrics  
	        
		
    def run(self):
        #cmd = ['vvencapp', '--preset', 'fast', '-i', self.in_yuv_path, '-s', self.frame_dim,'-q', str(self.qp),   '-f', str(self.n_frames),'-ip',str(self.intra_period), '-o', self.ostream_path]
        cmd = ["GFVC/HDAC/bl_codecs/VVC_VTM_22_2/bin/EncoderAppStatic", "-c", self.config_out_path,"-i", self.in_yuv_path]
        enc_start = time.time()
        with open(self.log_path, 'w+') as out:
            subprocess.call(cmd, stdout=out)
        # subprocess.call(cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        enc_time = time.time()-enc_start
        bits = os.path.getsize(self.ostream_path)*8
        dec_start = time.time()
        dec_cmd = ["GFVC/HDAC/bl_codecs/VVC_VTM_22_2/bin/DecoderAppStatic", "-b", self.ostream_path,'-o',self.dec_yuv_path]
        subprocess.call(dec_cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        dec_time = time.time() - dec_start

        dec_frames = self._yuv_2_mp4()
        shutil.rmtree(self.out_path)

        return {'dec_frames':dec_frames,'bits': bits,
                'time':{'enc_time': enc_time,'dec_time':dec_time}}


class VVC_Decoder:
    '''
        VVC VTM CODEC WRAPPER
    '''
    def __init__(self,fps=30,frame_dim=[256,256], bin_path='vvc_logs/'):
        self.fps = fps
        self.frame_dim=frame_dim
        self.out_path = bin_path
        os.makedirs(self.out_path, exist_ok=True)

        #inputs
        self.ostream_path = self.out_path+f'.bin'
        self.dec_yuv_path = self.out_path+f'/out.yuv'
        self.dec_mp4_path = self.out_path+f'/out.mp4'

    def _yuv_2_mp4(self):
        cmd = ['ffmpeg','-nostats','-loglevel','error', '-f', 'rawvideo', '-pix_fmt','yuv420p10le','-s:v',
               f"{self.frame_dim[0]}x{self.frame_dim[1]}", '-r', str(self.fps), '-i', self.dec_yuv_path,  self.dec_mp4_path]
        subprocess.call(cmd)
        frames = imageio.mimread(self.dec_mp4_path, memtest=False)
        return np.array(frames)

    def _get_metrics(self):
        original = imageio.mimread(self.in_mp4_path, memtest=False)
        decoded = imageio.mimread(self.dec_mp4_path, memtest=False)
        metrics = self.monitor.compute_metrics(original[:len(decoded)], decoded)
        return metrics  
	        
		
    def run(self):
        bits = os.path.getsize(self.ostream_path)*8
        dec_start = time.time()
        dec_cmd = ["GFVC/HDAC/bl_codecs/VVC_VTM_22_2/bin/DecoderAppStatic", "-b", self.ostream_path,'-o',self.dec_yuv_path]
        subprocess.call(dec_cmd, stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        dec_time = time.time() - dec_start

        dec_frames = self._yuv_2_mp4()
        shutil.rmtree(self.out_path)

        return {'dec_frames':dec_frames,'bits': bits,
                'time':{'dec_time':dec_time}}
