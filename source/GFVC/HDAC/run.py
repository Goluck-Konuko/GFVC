'''Training and Inference code for Hybrid Animation with deep animation models
#An extension of the the original HDAC implementation, with  improved training strategy as well as an implementation
of the VVC and VvEnC base layer coding.
#Intended as a test framework for MPEG SEI standardization for animation-based video enhancement in video 
conferencing applications. 
'''
import torch
import os
import yaml
from shutil import copy
from argparse import ArgumentParser
from time import gmtime, strftime
import models
import utils
from train import train
from test import test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", 
                        required=True, 
                        help="path to config")
    
    parser.add_argument("--mode", 
                        default="train", 
                        choices=["train","test"])
    
    parser.add_argument("--project_id", 
                        default='HDAC+', 
                        help="project name")
    
    parser.add_argument("--log_dir", 
                        default='log', 
                        help="path to log into")
    
    parser.add_argument("--verbose", 
                        dest="verbose", 
                        action="store_true", 
                        help="Print model architecture")
    
    parser.add_argument("--debug", 
                        dest="debug", 
                        action="store_true", 
                        help="Test on one batch to debug")
    
    parser.add_argument("--step", 
                        default=1,
                        type=int,
                        help="Training step")
    
    parser.add_argument("--num_workers", 
                            dest="num_workers", 
                            default=2, type=int, 
                            help="num of cpu cores for dataloading")
    
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    num_sources = config['dataset_params']['num_sources']-1
    model_id = os.path.basename(opt.config).split('.')[0]

    if opt.mode == 'train':
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += '_'+ strftime("%d_%m_%y_%H_%M_%S", gmtime())
    else:
        log_dir = os.path.join(opt.log_dir, f"{model_id}_test")

    #import Generator module
    generator_params = {
        **config['model_params']['common_params'],
        **config['model_params']['generator_params']}

    generator = models.GeneratorHDAC(**generator_params)  

    kpd_params = {**config['model_params']['common_params'],
                **config['model_params']['kp_detector_params']}
    kp_detector = models.KPD(**kpd_params)


    disc_type = config['model_params']['discriminator_params']['disc_type']
    if disc_type ==  'patch_gan_disc':
        #load a patch GAN discriminator instead
        discriminator = models.PatchDiscriminator(**config['model_params']['common_params'],
                                                    **config['model_params']['discriminator_params'])
    elif disc_type == 'multi_scale':
        discriminator = models.MultiScaleDiscriminator(**config['model_params']['common_params'],
                                                    **config['model_params']['discriminator_params'])
    else:
        raise NotImplementedError(f"Unknown discriminator type: {disc_type}")


    dataset = utils.FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.mkdir(log_dir+'/img_aug')
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)

        #pass config, generator, kp_detector and discriminator to the training module
        params = {  'project_id': opt.project_id,
                    'debug': opt.debug,
                    'model_id':model_id,
                    'log_dir': log_dir, 
                    'num_workers': opt.num_workers}

        train(config,dataset, generator,kp_detector,discriminator, **params)
    elif opt.mode == 'test':
        params = {  'model_id':model_id,'log_dir': log_dir}
        test(config,dataset, generator,kp_detector, **params)


    
