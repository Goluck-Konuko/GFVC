'''Data handling and packaging classes'''
import os
import numpy as np
from typing import List, Dict


class GFVCDataHandler:
    '''Handling'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
        self.rate_idx = kwargs['rate_idx']
        self.data_dir = f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/evaluation"
        self.bitrate_dir =  f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/resultBit"
        self.qps = kwargs['qps']
        self.metrics = kwargs['metrics']
        self.dataset_name = kwargs['dataset_name']
        
        self.data = self._load_data()

    def _load_data(self)->Dict[str,Dict[str,List[float]]]:
        '''Parses the metrics output files and returns the data in json format'''
        out = {}
        s_m_data = {}
        for qp in self.qps:
            # for m in self.metrics:
            #some sorting here to separate the sequences
            seqs = [x for x in os.listdir(self.data_dir) if self.dataset_name in x]
            bitrate_dir = self.bitrate_dir
            seqs = sorted([x for x in seqs if f"qp{qp}" in x], key= lambda x: x.split('_')[1])
            br_files = [x for x in os.listdir(bitrate_dir) if f"_qp{qp}" in x]
            # print(seqs)
            #read the evaluation data for each metric for all the sequences and compute averages
            qp_metrics = {}

            for m in self.metrics:
                m_data = []
                
                if m =="ssim":
                    #Needs special treatment to avoid confusion with ms_ssim in the list comprehension
                    target_files = [x for x in [f for f in seqs if m in f] if 'ms_ssim' not in x]
                else:
                    target_files = [f for f in seqs if m in f]
                
                for s in target_files:
                    s_data = self._read_data(f"{self.data_dir}/{s}")
                    m_data.append(s_data)
                    if m in ['lpips', 'dists']:
                        s_data = 1-np.mean(s_data)
                    else:
                        s_data = np.mean(s_data)
                    s_name = '_'.join(s.split('_')[:2])
                    if s_name not in s_m_data:
                        s_m_data[s_name] = {}
                    
                    if m not in s_m_data[s_name]:
                        s_m_data[s_name][m] = [] 
                    s_m_data[s_name][m].append(s_data)

                
                qp_metrics[m] = np.mean(m_data, axis=0).tolist() #compute the mean for all the sequences per frame
            
            #get the average BR, Encoding and Decoding Time
            bitrate, enc_time, dec_time = self._get_br_metrics(bitrate_dir,br_files)
            qp_metrics['bits'] = bitrate
            qp_metrics['enc_time'] = enc_time
            qp_metrics['dec_time'] = dec_time
            out[qp] = qp_metrics
        # print(s_m_data)
        return out

    def _read_data(self, path:str)-> List[float]:
        'Reads the text file and returns dtaa as a float list'
        with open(path, 'r') as out:
            s_data = out.read().splitlines()
        return [float(x) for x in s_data]
    
    def _get_br_metrics(self,bitrate_dir, br_files: List[str])->tuple:
        bits, enc_time, dec_time = 0 , 0 , 0
        for f in br_files:
            path = f"{bitrate_dir}/{f}"
            with open(path, 'r') as out:
                br_info = out.read().splitlines()[0]
            b,e,d = br_info.split(" ")
            bits += float(b)
            enc_time += float(e)
            dec_time += float(d)
        n_sequences = len(br_files)
        return bits/n_sequences, enc_time/n_sequences, dec_time/n_sequences


class DACDataHandler:
    '''Handling'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
        self.rate_idx = kwargs['rate_idx']
        self.data_dir = f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/evaluation"
        self.bitrate_dir =  f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/resultBit"
        self.qps = kwargs['qps']
        self.metrics = kwargs['metrics']
        self.dataset_name = kwargs['dataset_name']
        
        self.data = self._load_data()

    def _load_data(self)->Dict[str,Dict[str,List[float]]]:
        '''Parses the metrics output files and returns the data in json format'''
        out = {}
        for qp in self.qps:
            # for m in self.metrics:
            #some sorting here to separate the sequences
            seqs = [x for x in os.listdir(self.data_dir) if self.dataset_name in x]
            bitrate_dir = self.bitrate_dir
            seqs = sorted([x for x in seqs if f"_qp{qp}" in x], key= lambda x: x.split('_')[1])
            br_files = [x for x in os.listdir(bitrate_dir) if f"_qp{qp}" in x]
            # print(len(br_files))
            #read the evaluation data for each metric for all the sequences and compute averages
            qp_metrics = {}
            for m in self.metrics:
                m_data = []
                if m =="ssim":
                    #Needs special treatment to avoid confusion with ms_ssim in the list comprehension
                    target_files = [x for x in [f for f in seqs if m in f] if 'ms_ssim' not in x]
                else:
                    target_files = [f for f in seqs if m in f and 'result' not in f]
                target_files = [f for f in target_files if 'result' not in f]
                for s in target_files:
                    s_data = self._read_data(f"{self.data_dir}/{s}")
                    m_data.append(s_data)
                qp_metrics[m] = np.mean(m_data, axis=0).tolist() #compute the mean for all the sequences per frame
            # print(len(qp_metrics['psnr']))
            #get the average BR, Encoding and Decoding Time
            bitrate, enc_time, dec_time = self._get_br_metrics(bitrate_dir,br_files)
            qp_metrics['bits'] = bitrate
            qp_metrics['enc_time'] = enc_time
            qp_metrics['dec_time'] = dec_time
            out[qp] = qp_metrics
        
        return out

    def _read_data(self, path:str)-> List[float]:
        'Reads the text file and returns dtaa as a float list'
        with open(path, 'r') as out:
            s_data = out.read().splitlines()
        return [float(x) for x in s_data]
    
    def _get_br_metrics(self,bitrate_dir, br_files: List[str])->tuple:
        bits, enc_time, dec_time = 0 , 0 , 0
        for f in br_files:
            path = f"{bitrate_dir}/{f}"
            with open(path, 'r') as out:
                br_info = out.read().splitlines()[0]
            b,e,d = br_info.split(" ")
            bits += float(b)
            enc_time += float(e)
            dec_time += float(d)
        n_sequences = len(br_files)
        return bits/n_sequences, enc_time/n_sequences, dec_time/n_sequences

class RDACDataHandler(GFVCDataHandler):
    '''Handling >RDAC RD metrics'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
        self.rate_idx = kwargs['rate_idx']
        self.data_dir = f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/evaluation"
        if self.codec in ['rdac','rdacp']:
            self.bitrate_dir =  f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/resultBit_qqp"
        else:
            self.bitrate_dir =  f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/resultBit"
        self.qps = kwargs['qps']
        self.metrics = kwargs['metrics']
        self.dataset_name = kwargs['dataset_name']
        
        self.data = self._load_data()

    def _load_data(self)->Dict[str,Dict[str,List[float]]]:
        '''Parses the metrics output files and returns the data in json format'''
        out = {}
        for qp in self.qps:
            # for m in self.metrics:
            #some sorting here to separate the sequences
            seqs = [x for x in os.listdir(self.data_dir) if self.dataset_name in x]
            filter = f"qp4_qp{qp}"
            bitrate_dir = self.bitrate_dir+f"{qp}"
            seqs = sorted([x for x in seqs if filter in x], key= lambda x: x.split('_')[1])
            br_files = [x for x in os.listdir(bitrate_dir) if self.dataset_name in x]
            #read the evaluation data for each metric for all the sequences and compute averages
            qp_metrics = {}
            for m in self.metrics:
                m_data = []
                if m =="ssim":
                    #Needs special treatment to avoid confusion with ms_ssim in the list comprehension
                    target_files = [x for x in [f for f in seqs if m in f] if 'ms_ssim' not in x]
                else:
                    target_files = [f for f in seqs if m in f]
                for s in target_files:
                    s_data = self._read_data(f"{self.data_dir}/{s}")
                    m_data.append(s_data)
                qp_metrics[m] = np.mean(m_data, axis=0).tolist() #compute the mean for all the sequences per frame

            #get the average BR, Encoding and Decoding Time
            bitrate, enc_time, dec_time = self._get_br_metrics(bitrate_dir,br_files)
            qp_metrics['bits'] = bitrate
            qp_metrics['enc_time'] = enc_time
            qp_metrics['dec_time'] = dec_time
            out[qp] = qp_metrics
        return out

    
    def _get_br_metrics(self,bitrate_dir, br_files: List[str])->tuple:
        bits, enc_time, dec_time = 0 , 0 , 0
        for f in br_files:
            path = f"{bitrate_dir}/{f}"
            with open(path, 'r') as out:
                br_info = out.read().splitlines()[0]
            b,e,d = br_info.split(" ")
            bits += float(b)
            enc_time += float(e)
            dec_time += float(d)
        n_sequences = len(br_files)
        return bits/n_sequences, enc_time/n_sequences, dec_time/n_sequences

    def __repr__(self) -> str:
        return f"RD_QPS : {self.data.keys()}"


class HDACDataHandler(GFVCDataHandler):
    '''Handling HDAC RD metrics'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
        self.rate_idx = kwargs['rate_idx']
        self.data_dir = f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/evaluation"
        self.bitrate_dir =  f"experiment/{codec.upper()}/Iframe_{kwargs['iframe_format'].upper()}/resultBit"
        self.qps = kwargs['qps']
        self.metrics = kwargs['metrics']
        self.dataset_name = kwargs['dataset_name']
        
        self.data = self._load_data()

    def _load_data(self)->Dict[str,Dict[str,List[float]]]:
        '''Parses the metrics output files and returns the data in json format'''
        out = {}
        for qp in self.qps:
            # for m in self.metrics:
            #some sorting here to separate the sequences
            seqs = [x for x in os.listdir(self.data_dir) if self.dataset_name in x]
            filter = f"_qp{qp}_"
            bitrate_dir = self.bitrate_dir+f"_{qp}"
            seqs = sorted([x for x in seqs if filter in x], key= lambda x: x.split('_')[1])
            br_files = [x for x in os.listdir(bitrate_dir) if self.dataset_name in x]
            #read the evaluation data for each metric for all the sequences and compute averages
            qp_metrics = {}
            for m in self.metrics:
                m_data = []
                if m =="ssim":
                    #Needs special treatment to avoid confusion with ms_ssim in the list comprehension
                    target_files = [x for x in [f for f in seqs if m in f] if 'ms_ssim' not in x]
                else:
                    target_files = [f for f in seqs if m in f]
                for s in target_files:
                    s_data = self._read_data(f"{self.data_dir}/{s}")
                    m_data.append(s_data)
                qp_metrics[m] = np.mean(m_data, axis=0).tolist() #compute the mean for all the sequences per frame

            #get the average BR, Encoding and Decoding Time
            bitrate, enc_time, dec_time = self._get_br_metrics(bitrate_dir,br_files)
            qp_metrics['bits'] = bitrate
            qp_metrics['enc_time'] = enc_time
            qp_metrics['dec_time'] = dec_time
            out[qp] = qp_metrics
        return out

    
    def _get_br_metrics(self,bitrate_dir, br_files: List[str])->tuple:
        bits, enc_time, dec_time = 0 , 0 , 0
        for f in br_files:
            path = f"{bitrate_dir}/{f}"
            with open(path, 'r') as out:
                br_info = out.read().splitlines()[0]
            b,e,d = br_info.split(" ")
            bits += float(b)
            enc_time += float(e)
            dec_time += float(d)
        n_sequences = len(br_files)
        return bits/n_sequences, enc_time/n_sequences, dec_time/n_sequences

    def __repr__(self) -> str:
        return f"RD_QPS : {self.data.keys()}"


class AnchorDataHandler(GFVCDataHandler):
    '''Handling and loading HEVC and VVC RD metrics'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
        self.data_dir = f"experiment/{codec.upper()}/{kwargs['iframe_format'].upper()}/evaluation"
        
        self.qps = kwargs['qps']
        self.metrics = kwargs['metrics']
        self.dataset_name = kwargs['dataset_name']

        self.bitrate_dir =  f"experiment/{codec.upper()}/{self.dataset_name}_resultBit.txt"
        
        self.data = self._load_data()

    def _load_data(self)->Dict[str,Dict[str,List[float]]]:
        '''Parses the metrics output files and returns the data in json format'''
        out = {}
        
        for idx, qp in enumerate(self.qps):
            # for m in self.metrics:
            #some sorting here to separate the sequences
            seqs = [x for x in os.listdir(self.data_dir) if self.dataset_name in x]
            seqs = sorted([x for x in seqs if f"qp{qp}" in x], key= lambda x: x.split('_')[1])
            # br_files = [x for x in os.listdir(bitrate_dir) if filter in x]
            #read the evaluation data for each metric for all the sequences and compute averages
            qp_metrics = {}
            for m in self.metrics:
                m_data = []
                if m =="ssim":
                    #Needs special treatment to avoid confusion with ms_ssim in the list comprehension
                    target_files = [x for x in [f for f in seqs if m in f] if 'ms_ssim' not in x]
                else:
                    target_files = [f for f in seqs if m in f]
                for s in target_files:
                    s_data = self._read_data(f"{self.data_dir}/{s}")
                    m_data.append(s_data)
                qp_metrics[m] = np.mean(m_data, axis=0).tolist() #compute the mean for all the sequences per frame
            out[qp] = qp_metrics
        # print(out)
        #read bitrate metrics
        bitrates = self._get_br_metrics()
        out['bitrate'] =  bitrates
        return out
    
    def _get_br_metrics(self)->tuple:
        with open(self.bitrate_dir, 'r') as out:
            br_info = out.read().splitlines()
        br = np.array([[float(y) for y in (x.split(' '))] for x in br_info])
        br_avg = np.mean(br, axis=0).tolist()
        return br_avg
