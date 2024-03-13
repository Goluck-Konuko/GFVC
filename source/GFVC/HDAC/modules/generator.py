import torch
from torch import nn
import numpy as np
from typing import Dict, Any
import torch.nn.functional as F
from .motion_predictor import DenseMotionGenerator
from .nn_utils import ResBlock2d, SameBlock2d,UpBlock2d, DownBlock2d, OutputLayer ,ECABlock2d

class GeneratorHDAC(nn.Module):
    """
    Motion Transfer Generator with a scalable base layer encoder and 
    a conditional Multi-scale feature fusion
    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                 num_bottleneck_blocks=3, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False, scale_factor=0.25,**kwargs):
        super(GeneratorHDAC, self).__init__()
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

        #base_layer encoder
        self.base = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        base_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            base_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.base_down_blocks = nn.ModuleList(base_down_blocks)

        #Multi-layer fusion blocks
        main_bottleneck = []
        in_features = block_expansion * (2 ** num_down_blocks)*2

        #Regular residual block architecture
        for i in range(num_bottleneck_blocks):
            main_bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.main_bottleneck = nn.ModuleList(main_bottleneck)
        self.bt_output_layer = SameBlock2d(in_features, in_features//2, kernel_size=(3, 3), padding=(1, 1))
        #Upsampling layers
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = OutputLayer(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def base_layer_ft_encoder(self, base_layer_frame : torch.tensor )  -> torch.tensor :
        '''Extracts base layer features if available'''
        base_out = self.base(base_layer_frame)
        for i in range(len(self.down_blocks)):
            base_out = self.base_down_blocks[i](base_out)
        return base_out
    
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

    def animation_training(self, params):       
        # Encoding (downsampling) part      
        bl_fts = self.base_layer_ft_encoder(params['base_layer'])
        ref_fts = self.reference_ft_encoder(params['reference_frame'])          
        
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': ref_fts,
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }

        def_ref_fts, _  = self.motion_prediction_and_compensation(**motion_pred_params)      
        
        if np.random.rand()<0.25:
            #10 % of the time we don't pass the base layer features
            #Forces the network to learn pure animation as well
            bl_fts = torch.zeros_like(ref_fts).to(ref_fts.get_device())

        bt_input = torch.cat((def_ref_fts,bl_fts), dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
        bt_output = self.bt_output_layer(bt_input)

        output = {}
        output['context'] = bt_output.detach().clone()
        output['prediction'] = self.animated_frame_decoder(bt_output)
        return output
    
    def forward(self, **kwargs):     
        # Encoding (downsampling) part      
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {}
        reference_frame = kwargs['reference']
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': reference_frame,
                    'base_layer':kwargs[f'base_layer_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}']}
            output = self.animation_training(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict
    
    def update(self, output_dict, output,idx):
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict
    
    def predict(self, reference_frame:torch.Tensor, kp_reference:Dict[str, torch.Tensor], 
                kp_target:Dict[str, torch.Tensor], base_layer_frame=None):
        '''Prediction method at inference time'''   
        # Encoding (downsampling) part   
        ref_fts = self.reference_ft_encoder(reference_frame)    
        if base_layer_frame is not None:   
            bl_fts = self.base_layer_ft_encoder(base_layer_frame)
        else:
            bl_fts = torch.zeros_like(ref_fts).to(ref_fts.get_device())
                      
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features': ref_fts,
                    'kp_reference':kp_reference,
                    'kp_target':kp_target
                }

        def_ref_fts, _  = self.motion_prediction_and_compensation(**motion_pred_params)      
        
        bt_input = torch.cat([def_ref_fts,bl_fts], dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
        bt_output = self.bt_output_layer(bt_input)
        return self.animated_frame_decoder(bt_output)
    
