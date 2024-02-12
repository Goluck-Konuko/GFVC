# +
import os
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
from GFVC.utils import read_config_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="config/fomm.yaml", type=str, help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)

    #['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']
    seqlist= args['seq_list'] 
    
    #[ "22", "32", "42", "52"]
    qplist= args['qp_list'] 

    #You should download the testing sequence and modify the dir.
    Sequence_dir=args['sequence_dir'] 

    #'CFVQA' OR 'VOXCELEB'  ###You should choose which dataset to be encoded.
    testingdata_name=args['dataset']
    frames = args['num_frames']
    gop_size = args['gop_size']

    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256

    
    


    #'FV2V' OR 'FOMM' OR 'CFTE' ###You should choose which GFVC model to be used.
    model_name=args['codec_name']
    quantization_factor=args['quantization_factor']

    ## "Encoder" OR 'Decoder'   ###You need to define whether to encode or decode a sequence.
    coding_mode=args['inference_mode'] 
    ## 'YUV420'  OR 'RGB444' ###You need to define what color format to use for encoding the first frame.
    iframe_format=args['iframe_format']  

    for qp in qplist:
        for seq in tqdm(seqlist):
            original_seq=Sequence_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
            cmd = "./run.sh "+model_name+" "+coding_mode+" "+original_seq+" "+str(frames)+" "+str(quantization_factor)+" "+str(qp)+" "+str(iframe_format)
            if model_name in ['DAC','RDAC','RDACP']:
                cmd += " " + args['adaptive_metric'] + " " + str(args['adaptive_thresh'])
            if model_name in ['RDAC','RDACP']:
                cmd += " " + str(args['residual_coding_params']['rate_idx']) + " " + str(args['residual_coding_params']['int_value'])
            if model_name in ['RDACP']:
                cmd += " " + str(args['residual_coding_params']['kp_deform']) + " " + str(args['residual_coding_params']['bm_deform'])
            
            cmd += " "+str(gop_size)
            os.system(cmd)  
            print(model_name+"_"+coding_mode+"_"+seq+"_"+str(qp)+" Finished")
            # break 
        # break
            
