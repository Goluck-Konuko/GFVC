'''A wrapper class to plot Temporal metrics and RD curves'''
import os
import numpy as np
from typing import List, Dict
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from data_handlers import *


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

class Plotter:
    def __init__(self,out_path:str, codecs:List[str], metrics:List[str], qps:List[int]) -> None:
        self.out_path = out_path
        self.codecs = codecs
        self.metrics = metrics

    def plot_temporal_comparison(self, codec_data: Dict[str, GFVCDataHandler], qp:int=22)->None:
        '''Plots the evaluation metrics per frame'''
        plt.rcParams['figure.figsize'] = [14, 9] 
        plt.rcParams.update({'font.size': 25})
        output_path = f"{self.out_path}/TEMPORAL/{'_'.join(self.codecs)}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        codecs = list(codec_data.keys())
        # qp_list = [x for x in list(codec_data[codecs[0]].data.keys()) if x != 'bitrate']
        for metric in self.metrics:
            fig, ax = plt.subplots(1,1)
            for codec in self.codecs:
                qp_list = codec_data[codec].qps 
                m_data = codec_data[codec].data[qp_list[-1]][metric]
                if metric in ['lpips', 'dists']:
                    m_data = 1-np.array(m_data)
                ax.plot(range(len(m_data)), m_data,marker='o',linewidth=2.5,markersize=5, label=codec.upper())
            if metric in ['lpips', 'dists']:
                metric = f"I-{metric}"
            ax.set_ylabel(metric.upper(), fontsize=25)
            ax.set_xlabel('Frames', fontsize=25)
            ax.legend(loc='best', fontsize=25)
            plt.grid(alpha=0.2)
            fig.tight_layout()
            plt.savefig(f"{output_path}/{metric}_src_qp_{qp}.png", bbox_inches='tight')
            plt.close()

    def plot_rd_comparison(self, codec_data: Dict[str, GFVCDataHandler], fps:float =25.0)->None:
        '''Plots the RD evaluation curves'''
        plt.rcParams['figure.figsize'] = [14, 9] 
        plt.rcParams.update({'font.size': 25})
        output_path = f"{self.out_path}/RD/{'_'.join(self.codecs)}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        codecs = list(codec_data.keys())
        # qp_list = [x for x in list(codec_data[codecs[0]].data.keys()) if x != 'bitrate']

        for metric in self.metrics:
            fig, ax = plt.subplots(1,1)
            for codec in codecs:
                br, m_data = [], []
                qp_list = codec_data[codec].qps
                if codec in ['hevc', 'vvc']:
                    br = codec_data[codec].data['bitrate']
                    for q in qp_list:
                        m = codec_data[codec].data[q][metric]
                        m_data.append(np.mean(m))
                else:
                    for q in qp_list:
                        m = codec_data[codec].data[q][metric]
                        bitrate = ((codec_data[codec].data[q]['bits']/len(m))*fps)/1000
                        br.append(bitrate)
                        m_data.append(np.mean(m))
                if metric in ['lpips', 'dists']:
                    m_data = 1-np.array(m_data)
                ax.plot(br[-len(m_data):], m_data,marker='o',linewidth=3.5,markersize=15, label=codec.upper())
            if metric in ['lpips', 'dists']:
                metric = f"I-{metric}"
            ax.set_ylabel(metric.upper(), fontsize=25)
            ax.set_xlabel('Bitrate (kbps)', fontsize=25)
            ax.legend(loc='best', fontsize=25)
            ax.grid()
            fig.tight_layout()
            plt.savefig(f"{output_path}/{metric}.png", bbox_inches='tight')
            plt.close()
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--codecs", default="cfte,fv2v,dac,dac_20,dac_40", type=lambda x: list(map(str, x.split(','))), help="codecs to evaluate")
    parser.add_argument("--metrics", default="psnr,ssim,ms_ssim,fsim,lpips,dists,msVGG,vmaf", type=lambda x: list(map(str, x.split(','))), help="metrics to be evaluated")
    parser.add_argument("--qps", default="32,35,38,42,45,51", type=lambda x: list(map(int, x.split(','))), help="QP points on the RD curve")
    parser.add_argument('--dataset_name', default='voxceleb', type=str, help="Name of the evaluation dataset [voxceleb | cfvqa]")
    parser.add_argument('--format', default="yuv420", type=str, help="Format for compressing the reference frame [yuv420 | rgb444]")
    parser.add_argument('--rate_idx', default=1, type=int, help="RD index for RDAC residual coding")
    
    args = parser.parse_args()
    codec_data = {}

    codec_params = {'metrics':args.metrics, 'iframe_format':args.format, 'dataset_name': args.dataset_name.upper(), 'rate_idx':args.rate_idx}
    for codec in args.codecs:
        if codec in ['fv2v','cfte','fomm','dac','dac_20','dac_40']:
            #Reference frame QP values
            qp_list = ['22',"32","42",'52']
        elif 'hdac' in codec:
            qp_list = ["35","38","42","45","51"]
        elif codec in ['vvc', 'hevc']:
            qp_list =  ["32","35","38","42","45","51"] # OLD CTC QPS ["22","32","42","51"] 
        elif codec in ['rdac','rdacp']:
            # RD index for residual coding
            qp_list = [0,1,2,3]
        else:
            # QP value for low latency coding with HEVC and VVC
            qp_list = args.qps
        codec_params.update({'qps': qp_list})
        path = f"experiment/{codec.upper()}/evaluation"
        if  codec in ['hevc','vvc']:
            data_handler = AnchorDataHandler
        elif 'hdac' in codec:
            data_handler = HDACDataHandler
        elif codec in ['dac','dac_10','dac_20','dac_40']:
            data_handler = DACDataHandler
        else:
            data_handler = GFVCDataHandler
        codec_data[codec] = data_handler(codec, **codec_params)

    ## Generate Plots
    dataset_name = args.dataset_name.upper()
    output_path = f"experiment/PLOTS/{dataset_name}"
    plotter = Plotter(out_path=output_path,codecs=args.codecs,metrics=args.metrics, qps=args.qps)
    plotter.plot_rd_comparison(codec_data)