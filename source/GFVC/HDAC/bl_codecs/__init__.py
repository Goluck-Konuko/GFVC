from .hevc import HEVC_Encoder, HEVC_Decoder
from .vvc import VVC_Encoder, VVC_Decoder


bl_encoders = {'hevc': HEVC_Encoder,'vvc': VVC_Encoder}

bl_decoders = {'hevc': HEVC_Decoder, 'vvc': VVC_Decoder}