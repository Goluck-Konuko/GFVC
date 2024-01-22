import matplotlib
matplotlib.use('Agg')

import os
import yaml
import torch
import utils
from shutil import copy
from train import train
from time import gmtime, strftime
from argparse import ArgumentParser
from GFVC.RDAC.modules.generator import GeneratorRDAC
from GFVC.RDAC.modules.keypoint_detector import KPDetector
from GFVC.RDAC.modules.discriminator import MultiScaleDiscriminator

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", 
                        required=True, 
                        help="path to config")
    
    parser.add_argument("--mode", 
                        default="train", 
                        choices=["train"])
    
    parser.add_argument("--project_id", 
                        default='RDAC', 
                        help="project name")
    
    parser.add_argument("--log_dir", 
                        default='log', 
                        help="path to log into")
    
    parser.add_argument("--checkpoint", 
                        default=None, 
                        help="Use pretrained generator and kp detector")
    
    parser.add_argument("--device_ids", 
                        default="0", 
                        type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    
    parser.add_argument("--num_features", 
                        default=48,type=int, 
                        help="number of features in the RDAC Difference coder")
    
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
    
    parser.add_argument("--rate", 
                        default=3,
                        type=int,
                        help="target bitrate | 8 bitrates [0-8]")
    
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
        if opt.checkpoint is not None:
            log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        else:
            log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
            log_dir += f"_{opt.num_features}_{opt.rate}"
            log_dir += '_'+ strftime("%d_%m_%y_%H_%M_%S", gmtime())

    #import Generator module
    config['train_params']['target_rate'] = opt.rate
    generator_params = {
        **config['model_params']['common_params'],
        **config['model_params']['generator_params']}


    generator = GeneratorRDAC(**generator_params)

    if generator != None:
        print(f"Using: {generator.__class__.__name__}")

        if torch.cuda.is_available():
            generator.to(opt.device_ids[0])

    kp_detector = KPDetector(**config['model_params']['common_params'],
                            **config['model_params']['kp_detector_params'])
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    discriminator = MultiScaleDiscriminator(**config['model_params']['common_params'],
                                                **config['model_params']['discriminator_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])

    dataset = utils.FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.mkdir(log_dir+'/img_aug')
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    #pass config, generator, kp_detector and discriminator to the training module
    params = {  'project_id': opt.project_id,
                'debug': opt.debug,
                'model_id':model_id,
                'step': opt.step,
                'checkpoint': opt.checkpoint, 
                'log_dir': log_dir, 
                'device_ids': opt.device_ids,
                'num_workers': opt.num_workers}

    train(config,dataset, generator,kp_detector,discriminator, **params)


    
