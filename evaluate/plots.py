'''A wrapper class to plot Temporal metrics and RD curves'''
import os
import numpy as np
from typing import List, Dict, Any
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def plot_rd(metrics, out_path='plots'):
    models = list(metrics.keys())
    mts = (metrics[models[0]].keys())
    for mt in mts:
        if mt != 'bpp_loss':
            fig, ax = plt.subplots(1,1)
            for model in models:
                br = metrics[model]['bpp_loss']
                m = metrics[model][mt]
                ax.plot(br, m,marker='o',linewidth=3.5,markersize=15, label=model.upper())
            ax.set_ylabel(mt.upper(), fontsize=25)
            ax.set_xlabel('BPP', fontsize=25)
            ax.legend(loc='best', fontsize=25)
            plt.grid(alpha=0.2)
            # n = '_'.join(metrics.keys())
            out_path = f"{out_path}"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            plt.savefig(f"{out_path}/{mt}_rd.png", bbox_inches='tight')
            plt.close()

class RD_Data:
    '''Handling'''
    def __init__(self, codec:str, **kwargs) -> None:
        self.codec = codec
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
            seqs = sorted([x for x in seqs if f"QP{qp}" in x], key= lambda x: x.split('_')[1])
            br_files = [x for x in os.listdir(self.bitrate_dir) if f"QP{qp}" in x]
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
                qp_metrics[m] = np.mean(m_data, axis=0) #compute the mean for all the sequences per frame

            #get the average BR, Encoding and Decoding Time
            bitrate, enc_time, dec_time = self._get_br_metrics(br_files)
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
    
    def _get_br_metrics(self, br_files: List[str])->tuple:
        bits, enc_time, dec_time = 0 , 0 , 0
        for f in br_files:
            path = f"{self.bitrate_dir}/{f}"
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

class Plotter:
    def __init__(self,out_path:str, codecs:List[str], metrics:List[str], qps:List[int]) -> None:
        self.out_path = out_path
        self.codecs = codecs
        self.metrics = metrics
        self.qps = qps

    def plot_temporal_comparison(self, codec_data: Dict[str, RD_Data], qp:int=22)->None:
        '''Plots the evaluation metrics per frame'''
        plt.rcParams['figure.figsize'] = [14, 9] 
        plt.rcParams.update({'font.size': 25})
        output_path = f"{self.out_path}/PLOTS/TEMPORAL/{'_'.join(self.codecs)}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for metric in self.metrics:
            fig, ax = plt.subplots(1,1)
            for cd in self.codecs:
                m_data = codec_data[cd].data[qp][metric]
                ax.plot(range(len(m_data)), m_data,marker='o',linewidth=2.5,markersize=5, label=cd.upper())
            ax.set_ylabel(metric.upper(), fontsize=25)
            ax.set_xlabel('Frames', fontsize=25)
            ax.legend(loc='best', fontsize=25)
            plt.grid(alpha=0.2)
            fig.tight_layout()
            plt.savefig(f"{output_path}/{metric}_src_qp_{qp}.png", bbox_inches='tight')
            plt.close()

    def plot_rd_comparison(self, codec_data: Dict[str, RD_Data], fps:float =25.0)->None:
        '''Plots the RD evaluation curves'''
        plt.rcParams['figure.figsize'] = [14, 9] 
        plt.rcParams.update({'font.size': 25})
        output_path = f"{self.out_path}/PLOTS/RD/{'_'.join(self.codecs)}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for metric in self.metrics:
            fig, ax = plt.subplots(1,1)
            for cd in self.codecs:
                br, m_data = [], []
                for q in self.qps:
                    m = codec_data[cd].data[q][metric]
                    bitrate = ((codec_data[cd].data[q]['bits']/len(m))*fps)/1000
                    br.append(bitrate)
                    m_data.append(np.mean(m))
                ax.plot(br, m_data,marker='o',linewidth=3.5,markersize=15, label=cd.upper())
            ax.set_ylabel(metric.upper(), fontsize=25)
            ax.set_xlabel('Bitrate (kbps)', fontsize=25)
            ax.legend(loc='best', fontsize=25)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            plt.savefig(f"{output_path}/{metric}.png", bbox_inches='tight')
            plt.close()
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--codecs", default="fv2v,fomm,cfte,dac", type=lambda x: list(map(str, x.split(','))), help="codecs to evaluate")
    parser.add_argument("--metrics", default="psnr,ssim,ms_ssim,fsim,lpips,dists", type=lambda x: list(map(str, x.split(','))), help="metrics to be evaluated")
    parser.add_argument("--qps", default="22,32,42,52", type=lambda x: list(map(int, x.split(','))), help="QP points on the RD curve")
    parser.add_argument('--dataset_name', default='voxceleb', type=str, help="Name of the evaluation dataset [voxceleb | cfvqa]")
    parser.add_argument('--format', default="yuv420", type=str, help="Format for compressing the reference frame [yuv420 | rgb444]")
    args = parser.parse_args()
    
    
    codec_data = {}
    codec_params = {'qps': args.qps, 'metrics':args.metrics, 'iframe_format':args.format, 'dataset_name': args.dataset_name.upper()}
    for codec in args.codecs:
        path = f"experiment/{codec.upper()}/evaluation"
        codec_data[codec] = RD_Data(codec, **codec_params)

    ## Generate Plots
    output_path = "experiment"
    plotter = Plotter(out_path=output_path,codecs=args.codecs,metrics=args.metrics, qps=args.qps)
    plotter.plot_temporal_comparison(codec_data)
    plotter.plot_rd_comparison(codec_data)