#!/bin/bash

if [ $1 == "HDAC" ]; then 
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --num_kp ${11} --use_base_layer ${12} --base_codec ${13} --bl_qp ${14}  --bl_scale_factor ${15}  --gop_size ${16}
elif [ $1 == "DAC" ]; then
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --adaptive_metric $9 --adaptive_thresh ${10} --num_kp ${11} --gop_size ${12}
else
python $1_$2.py  --original_seq $3 --encoding_frames $4 --quantization_factor $5 --iframe_qp $6 --iframe_format $7 --ref_codec $8 --gop_size $9
fi