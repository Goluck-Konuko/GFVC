#!/bin/bash
if [ $1 == "HDAC" ]; then 
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --use_base_layer ${11} --base_codec ${12} --bl_qp ${13}  --bl_scale_factor ${14}  --gop_size ${15}
elif [ $1 == "RDACP" ]; then 
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --rate_idx ${11} --int_value ${12} --kp_deform ${13}  --bm_deform ${14}  --gop_size ${15}
elif [ $1 == "RDAC" ]; then 
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --rate_idx ${11} --int_value ${12} --gop_size ${13}
elif [ $1 == "DAC" ]; then
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --gop_size ${11}
else
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --gop_size $9
fi