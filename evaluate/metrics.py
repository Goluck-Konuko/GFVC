import os
import time
import torch
import numpy as np
from tqdm import tqdm
from eval_utils.utils import read_config_file
from argparse import ArgumentParser
from eval_utils.metrics import MultiMetric


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

    out = []
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        out.append(np.array([R,G,B]))
    return out

     
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="config/cfte.yaml", type=str, help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)
    
    seqlist= args['seq_list']
    Model=args['codec_name']  ## 'FV2V' OR 'FOMM' OR 'CFTE'
    if Model =='HDAC':
        qplist = ["51","42","45","38","35"]
    else:
        qplist= args['qp_list'] 

    width=args['width']
    height=args['height']

    testingdata_name=args['dataset']
    frames = args['num_frames']
   
    
    
    Iframe_format=args['iframe_format']   ## 'YUV420'  OR 'RGB444'

    if Model =='DAC':
        thresh = int(args['adaptive_thresh'])
    
    if Model in ['HEVC', 'VVC']:
        result_dir = './experiment/'+Model+'/'+Iframe_format+'/evaluation/'
    elif Model in ['HDAC']:
        bl_codec = args['base_layer_params']['base_codec'].upper()
        result_dir = './experiment/'+f"{Model}_{bl_codec}"+'/Iframe_'+Iframe_format+'/evaluation/' 
    else:
        result_dir = './experiment/'+Model+'/Iframe_'+Iframe_format+'/evaluation/'

    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    monitor = MultiMetric(metrics=args['metrics'], device=device)
    total_result = {}
    for m in monitor.metrics:
        total_result.update({f"totalResult_{m.upper()}": np.zeros((len(seqlist)+1,len(qplist)))})

    seqIdx=0
    for seq in tqdm(seqlist):
        qpIdx=0
        for qp in tqdm(qplist):
            start=time.time()    
            if not os.path.exists(result_dir):
                os.makedirs(result_dir) 

            ### You should modify the path of original sequence and reconstructed sequence
            # f_org=open('./dataset/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb','rb')
            f_org_path = './dataset/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
            org_seq = raw_reader_planar(f_org_path,width,height,frames)
            
            if Model in ['HEVC','VVC']:
                f_test_path = './experiment/'+Model+'/'+Iframe_format+'/dec/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_QP'+str(qp)+'.rgb'
            elif Model in ['HDAC']:
                f_test_path = './experiment/'+f"{Model}_{bl_codec}"+'/'+'Iframe_'+Iframe_format+'/dec/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_qp22'+'_th'+str(qp)+'.rgb'
            elif Model in ['DAC']:
                f_test_path = './experiment/'+Model+'/'+'Iframe_'+Iframe_format+'/dec/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_qp'+str(qp)+'_th'+str(thresh)+'.rgb'
            else:
                f_test_path = './experiment/'+Model+'/'+'Iframe_'+Iframe_format+'/dec/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_qp'+str(qp)+'.rgb'
            dec_seq = raw_reader_planar(f_test_path,width,height,frames)

            #open files to store the computed metrics
            output_files = {}
            accumulated_metrics = {}
            for m in monitor.metrics:
                if Model == 'DAC':
                    mt_out_path = result_dir+testingdata_name+'_'+str(seq)+'_qp'+str(qp)+f'_th_{thresh}'+f'_{m}.txt'
                else:
                    mt_out_path = result_dir+testingdata_name+'_'+str(seq)+'_qp'+str(qp)+f'_{m}.txt'
                output_files.update({m:open(mt_out_path,'w')})
                accumulated_metrics.update({m:0})


            for idx in tqdm(range(frames)):                 
                #compute the metrics
                out_metrics = monitor.compute_metrics(org_seq[idx], dec_seq[idx])
                #write the output values to text file and add
                for m in out_metrics:
                    output_files[m].write(str(out_metrics[m]))
                    output_files[m].write('\n')
                    accumulated_metrics[m] += out_metrics[m]
                    
            #compute the totals and close the output files
            for m in monitor.metrics:
                total_result[f"totalResult_{m.upper()}"][seqIdx][qpIdx] = accumulated_metrics[m]/frames
                output_files[m].close()
            end=time.time()
            print(testingdata_name+'_'+str(seq)+'_qp'+str(qp)+'.rgb',"success. Time is %.4f"%(end-start))
            qpIdx+=1
            # break
        seqIdx+=1
        # break

    np.set_printoptions(precision=5)

    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            for m in monitor.metrics:
                total_result[f"totalResult_{m.upper()}"][-1][qp] += total_result[f"totalResult_{m.upper()}"][seq][qp]

        for m in monitor.metrics:
            total_result[f"totalResult_{m.upper()}"][-1][qp] /= len(seqlist)
    
    #Save the results to text
    for m in monitor.metrics:
        if Model in ['RDAC']:
            mt_path = result_dir+testingdata_name+f"_result_rqp_{args['residual_coding_params']['rate_idx']}_{m}.txt"
        elif Model in ['DAC']:
            mt_path = result_dir+testingdata_name+f"_result_th_{thresh}_{m}.txt"
        else:
            mt_path = result_dir+testingdata_name+f'_result_{m}.txt'
        np.savetxt(mt_path, total_result[f"totalResult_{m.upper()}"], fmt = '%.5f')

