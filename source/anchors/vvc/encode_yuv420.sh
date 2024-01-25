#!/bin/sh

./vvc/bin/EncoderAppStatic_VTM22_2 -c ./vvc/cfg/encoder_lowdelay_P_vtm.cfg -c ./vvc/cfg/per-sequence/inputyuv420.cfg -q $1 -f $2 -wdt $3 -hgt $4 -i $5 -o $6$7_$3x$4_25_8bit_420_QP$1.yuv -b $6$7_$3x$4_25_8bit_420_QP$1.bin >>$6$7_$3x$4_25_8bit_420_QP$1.log

