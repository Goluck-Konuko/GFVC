#!/bin/sh

image_codecs/vtm/bin/EncoderAppStatic_VTM22_2 -c ./image_codecs/vtm/cfg/encoder_lowdelay_vtm.cfg -c ./image_codecs/vtm/cfg/per-sequence/inputyuv420.cfg -q $2 -i $1_org.yuv -wdt $3 -hgt $4 -o $1_rec.yuv -b $1.bin >>$1.log 

