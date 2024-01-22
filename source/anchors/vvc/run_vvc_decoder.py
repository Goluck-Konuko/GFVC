# +
import os
from utils import *
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="../config/vvc.yaml", type=str, help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)
  
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
  


    if input_format=='RGB444':
        os.makedirs("../experiment/RGB444/Dec/",exist_ok=True)     
        for qp in qplist:
            for seq in seqlist:
                os.system("./anchors/vvc/decode_rgb444.sh "+test_dataset+'_'+seq+" "+qp+" "+"../experiment/RGB444/EncodeResult/"+" "+"../experiment/RGB444/Dec/"+' &')   ########################
                print(seq+"_"+qp+" submiited")
                

    elif input_format=='YUV420':
        os.makedirs("../experiment/YUV420/Dec/",exist_ok=True)     
        
        os.makedirs("../experiment/YUV420/YUVtoRGB/",exist_ok=True)     
        
        for qp in qplist:
            for seq in seqlist:
                os.system("./anchors/vvc/decode_yuv420.sh "+test_dataset+'_'+seq+" "+qp+" "+"../experiment/YUV420/EncodeResult"+" "+"../experiment/YUV420/Dec/")   ########################
                print(seq+"_"+qp+" submited")    

                decode_seq="../experiment/YUV420/YUVtoRGB/"+test_dataset+'_'+seq+'_qp'+str(qp)+'_dec.rgb'         
                f_dec=open(decode_seq,'w') 

                #  read the rec frame (yuv420) and convert to rgb444
                rec_ref_yuv=yuv420_to_rgb444("../experiment/YUV420/Dec/"+test_dataset+'_'+seq+"_qp"+qp+'_dec.yuv', width, height, 0, frames, False, False) 
                
                for frame_idx in range(0, frames):            
                    img_rec = rec_ref_yuv[frame_idx]
                    img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                    img_rec.tofile(f_dec)     
                f_dec.close()   
                break
            break  



