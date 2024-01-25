#!/bin/sh

image_codecs/vtm/bin/EncoderAppStatic_VTM22_2 -c ./image_codecs/vtm/cfg/encoder_lowdelay_vtm.cfg -c ./vtm/cfg/per-sequence/inputrgb444.cfg -c ./vtm/cfg/formatRGB.cfg -q $2 -i $1_org.rgb -wdt $3 -hgt $4 -o $1_rec.rgb -b $1.bin >>$1.log 


