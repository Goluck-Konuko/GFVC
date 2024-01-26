# +
# get file size in python
import os
import numpy as np
from argparse import ArgumentParser
from eval_utils.utils import read_config_file


def get_all_file(dir_path):
    global files
    for filepath in os.listdir(dir_path):
        tmp_path = os.path.join(dir_path,filepath)
        if os.path.isdir(tmp_path):
            get_all_file(tmp_path)
        else:
            files.append(tmp_path)
    return files

def calc_files_size(files_path):
    files_size = 0
    for f in files_path:
        files_size += os.path.getsize(f)
    return files_size

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="config/hevc.yaml", type=str, help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)

    codec_name = args['codec_name']
    #['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']
    seqlist= args['seq_list'] 
    
    #[ "22", "32", "42", "52"]
    qplist= args['qp_list'] 

    #You should download the testing sequence and modify the dir.
    sequence_dir=args['sequence_dir'] 

    #'CFVQA' OR 'VOXCELEB'  ###You should choose which dataset to be encoded.
    test_dataset=args['dataset']
    frames = args['num_frames']


    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256

    input_format=args['iframe_format']     
        
    input_bin_file_path='./experiment/'+codec_name.upper()+'/'+input_format+'/enc/'
    save_path='./experiment/'+codec_name.upper()+'/'


    totalResult=np.zeros((len(seqlist)+1,len(qplist)))

    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, qp in enumerate(qplist):  
    
                
            path = input_bin_file_path+test_dataset+'_'+str(seq)+f'_{width}x{height}_25_8bit_420_QP'+str(qp)+'.bin'
            overall_bits=os.path.getsize(path)*8 
            print(overall_bits)

            totalResult[seqIdx][qpIdx]=overall_bits   
            
            
    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)

    np.set_printoptions(precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength

    np.savetxt(save_path+test_dataset+'_resultBit.txt', totalResult, fmt = '%.5f')                    
            

    
    
