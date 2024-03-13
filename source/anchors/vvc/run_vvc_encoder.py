# +
import os
import subprocess
from utils import *
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="../../config/vvc.yaml", type=str, help="Path to codec configuration file")
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


    os.makedirs(f"../../experiment/VVC/{input_format}/",exist_ok=True) 
    enc_dir = f"../../experiment/VVC/{input_format}/enc/"  
    os.makedirs(enc_dir,exist_ok=True) 

    if input_format=='RGB444':
        for qp in qplist:
            for seq in seqlist:
                original_seq=sequence_dir+test_dataset+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb' 
                os.system("./vvc/encode_rgb444.sh "+qp+" "+str(frames)+" "+str(width)+" "+str(height)+" "+original_seq+" "+enc_dir+" "+test_dataset+'_'+seq+" "+' &') 
                print(seq+"_"+qp+" submited")
        
    elif input_format=='YUV420':
        org_yuv_dir = f"../../experiment/VVC/{input_format}/org_yuv/"
        os.makedirs(org_yuv_dir,exist_ok=True)             
        #CONVERT THE Video file from RGB to YUV
        for seq in seqlist:    
            original_seq=sequence_dir+test_dataset+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
            
            # wtite ref and cur (rgb444) to file (yuv420)
            oriyuv_path=org_yuv_dir+test_dataset+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_420.yuv'
            if not os.path.exists(oriyuv_path):
                listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

                f_temp=open(oriyuv_path,'w')            
                for frame_idx in range(0, frames):            

                    img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                    img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                    img_input_yuv.tofile(f_temp)
                f_temp.close()   
            # break  
            
        #Run the batch coding of all sequences
        for qp in tqdm(qplist):
            for seq in tqdm(seqlist):            
                oriyuv_path=org_yuv_dir+test_dataset+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_420.yuv'
                # os.system("./vvc/encode_yuv420.sh "+qp+" "+str(frames)+" "+str(width)+" "+str(height)+" "+oriyuv_path+" "+enc_dir+" "+test_dataset+'_'+seq+" "+' &')   
                subprocess.call(["./vvc/encode_yuv420.sh",str(qp),str(frames),str(width),str(height),oriyuv_path, enc_dir, test_dataset+'_'+seq,'&'])   
                ########################       
                print(seq+"_"+qp+" submited")
                

                
