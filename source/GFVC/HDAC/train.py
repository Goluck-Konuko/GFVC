import os
import torch
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import utils
import models


def load_pretrained_model(model, path ,name: str='generator', device:str='cpu', strict= False):
    cpk = torch.load(path, map_location=device)
    if name in cpk:
        model.load_state_dict(cpk[name], strict=strict)
    return model


def train(config,dataset,generator, kp_detector,discriminator,**kwargs ):
    train_params = config['train_params'] 
    debug = kwargs['debug'] 
    # create optimizers for generator, kp_detector and discriminator
    parameters = list(generator.parameters()) + list(kp_detector.parameters())
    gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'], betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=train_params['lr'],  betas=(0.5, 0.999))  
      

    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_cpk_path = config['dataset_params']['cpk_path'] 
    if pretrained_cpk_path != '':
        #Retrain from a saved checkpoint specified in the config file
        generator = load_pretrained_model(generator, path=pretrained_cpk_path,name='generator', device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        gen_optimizer = load_pretrained_model(gen_optimizer, path=pretrained_cpk_path,name='gen_optimzer', device=device)

        discriminator = load_pretrained_model(discriminator, path=pretrained_cpk_path,name='discriminator', device=device)
        disc_optimizer = load_pretrained_model(disc_optimizer, path=pretrained_cpk_path,name='disc_optimizer', device=device)
    
    scheduler_generator = MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    generator_full = models.TrainerModel(kp_detector, generator, discriminator,config) 
    
    scheduler_discriminator = MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    
    disc_type = config['model_params']['discriminator_params']['disc_type']
    discriminator_full = models.DiscriminatorModel(discriminator, train_params, disc_type=disc_type) 

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        discriminator_full = discriminator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = utils.DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with utils.Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if device =='cuda':
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward()
                
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                #forward and backprop on the discriminator
                losses_discriminator = discriminator_full(x, generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward()
                
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                logger.log_iter(losses=losses)
                
                if debug:
                    break

            scheduler_generator.step()
            scheduler_discriminator.step()

            state_dict = {'generator': generator,
                        'kp_detector': kp_detector, 
                        'gen_optimizer': gen_optimizer,
                        'discriminator':discriminator, 
                        'disc_optimizer':disc_optimizer}
            
            logger.log_epoch(epoch, state_dict, inp=x, out=generated)
            if debug:
                break

class CustomDataParallel(torch.nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
            

            
