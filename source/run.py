# +
import os
import yaml
from tqdm import tqdm
from argparse import ArgumentParser


def read_config_file(config_path):
    '''Simply reads the configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

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

    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256


    #'FV2V' OR 'FOMM' OR 'CFTE' ###You should choose which GFVC model to be used.
    Model=args['codec_name']
    quantization_factor=args['quantization_factor']

    ## "Encoder" OR 'Decoder'   ###You need to define whether to encode or decode a sequence.
    Mode=args['inference_mode'] 
    ## 'YUV420'  OR 'RGB444' ###You need to define what color format to use for encoding the first frame.
    Iframe_format=args['iframe_format']  



    for qp in qplist:
        for seq in tqdm(seqlist):
            original_seq=Sequence_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
            cmd = "./run.sh "+Model+" "+Mode+" "+original_seq+" "+str(frames)+" "+str(quantization_factor)+" "+str(qp)+" "+str(Iframe_format)
            if Model in ['DAC','RDAC']:
                cmd += " " + args['adaptive_metric'] + " " + str(args['adaptive_thresh'])
            os.system(cmd)  
            print(Model+"_"+Mode+"_"+seq+"_"+qp+" Finished")
            # break 
        # break
            
