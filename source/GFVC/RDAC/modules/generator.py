import torch
import numpy as np
from torch import nn
from typing import Dict, Any
import torch.nn.functional as F
from GFVC.RDAC.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, OutputLayer
from GFVC.RDAC.modules.dense_motion import DenseMotionGenerator
from GFVC.RDAC.modules.residual_coder import ResidualCoder


class GeneratorRDAC(nn.Module):
    """
    Generator architecture using spatial attention layers and ConvNext Modules
    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                  num_bottleneck_blocks=3, estimate_occlusion_map=False, 
                  dense_motion_params=None,residual_coder_params=None, **kwargs):
        super(GeneratorRDAC, self).__init__()
        if dense_motion_params:
            self.dense_motion_network = DenseMotionGenerator(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,**dense_motion_params)
        else:
            self.dense_motion_network = None
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3),)         

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = block_expansion * (2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
    
        self.final = OutputLayer(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

        # spatial_difference_coder
        residual_features = residual_coder_params['residual_features']
        N, M = residual_features, int(residual_features*1.5)
        self.sdc = ResidualCoder(num_channels,num_channels, N, M,**residual_coder_params)
        
        self.tdc = ResidualCoder(num_channels,num_channels, N, M,**residual_coder_params)

    def deform_input(self, inp, deformation):
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def reference_ft_encoder(self, reference_frame):
        out = self.first(reference_frame)        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out

    def animated_frame_decoder(self, out):
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def motion_prediction_and_compensation(self,reference_frame: torch.tensor=None, 
                                           reference_frame_features: torch.tensor=None,
                                            **kwargs):
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features, dense_motion

    def deform_residual(self, params: Dict[str, Any]) -> torch.tensor:
        #generate a dense motion estimate
        dense_motion = self.dense_motion_network(reference_frame=params['prev_rec'], kp_target=params['kp_cur'],
                                                    kp_reference=params['kp_prev'])
        #apply it to the residual map
        warped_residual = self.deform_input(params['res_hat_prev'], dense_motion['deformation'])
        return warped_residual

    def animate_training(self, params):
        #perform animation of previous frame
        output_dict = {}

        motion_pred_params = { 
                            'reference_frame': params['reference_frame'],
                            'reference_frame_features': params['ref_fts'],
                            'kp_reference': params['kp_reference'],
                            'kp_target': params['kp_target']}

        #then animate current frame with residual info from the current frame
        def_ref_fts, _  = self.motion_prediction_and_compensation(**motion_pred_params)

        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        
        animated_frame = self.animated_frame_decoder(out_ft_maps)
        output_dict["prediction"] = animated_frame

        #Compute and encode a generalized difference between the animated and target images
        residual = params['target_frame'] - animated_frame
        output_dict['res'] = residual
        #if we have information about the previous frame residual
        #then we can use it as additional information to minimize the entropy of current frame
        if 'res_hat_prev' in params:
            residual_temp = (residual-params[f"res_hat_prev"])/2.0
            res_hat_temp, bpp, prob = self.tdc(residual_temp,rate_idx = params['rate_idx'])
            
            output_dict['res_temp_hat'] = res_hat_temp
            # #reconstitute actual synthesis residual from previous frame and current
            res_hat =  params["res_hat_prev"]+res_hat_temp*2.0
            # print("Rec temporal residual", res_hat.shape)
        else:     
            res_hat, bpp, prob = self.sdc(residual,rate_idx = params['rate_idx'])
        output_dict['rate'] = bpp
        output_dict['prob'] = prob
        output_dict["res_hat"] = res_hat        
        
        output_dict['enhanced_prediction'] = (animated_frame+res_hat).clamp(0,1)
        return output_dict
    
    def forward(self, **kwargs):     
        # Encoding (downsampling) part
        reference_frame = kwargs['reference']      
        ref_fts = self.reference_ft_encoder(reference_frame)
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {}
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': reference_frame,
                    'ref_fts': ref_fts,
                    'target_frame':kwargs[f'target_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}'],
                    'rate_idx': kwargs[f'rate_idx']}
            if idx>0:
                res_hat_prev = output_dict[f'res_hat_{idx-1}'] #.detach().clone()
                params.update({'res_hat_prev': res_hat_prev})
                
            output = self.animate_training(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict

    def animate(self, reference_frame, kp_reference, kp_target):     
        # Encoding (downsampling) part      
        ref_fts = self.reference_ft_encoder(reference_frame)
        
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation   
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features': ref_fts,
                    'kp_reference':kp_reference,
                    'kp_target':kp_target
                }
        def_ref_fts, _= self.motion_prediction_and_compensation(**motion_pred_params)

        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        #reconstruct the animated frame
        return self.animated_frame_decoder(out_ft_maps)
    
    def compress_spatial_residual(self,residual_frame:torch.Tensor, prev_latent:torch.Tensor,
                                  rate_idx=0, q_value=1.0,use_skip=False, skip_thresh=0.9, scale_factor=1.0)->Dict[str, Any]:
        res_info = self.sdc.rans_compress(residual_frame,prev_latent, rate_idx, q_value, use_skip, skip_thresh, scale_factor)
        return res_info

    def compress_temporal_residual(self,residual_frame:torch.Tensor, prev_latent:torch.Tensor,
                                  rate_idx=0, q_value=1.0,use_skip=False, skip_thresh=0.9, scale_factor=1.0)->Dict[str, Any]:
        res_info = self.tdc.rans_compress(residual_frame,prev_latent, rate_idx, q_value, use_skip, skip_thresh, scale_factor)
        return res_info

