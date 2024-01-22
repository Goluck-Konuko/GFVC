import os
import torch
import utils
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from GFVC.RDAC.modules.model import TrainerModel, DiscriminatorModel


class ConfigureOptimizers:
    def __init__(self, generator, kp_detector=None, step_wise=False) -> None:
        self.step_wise= step_wise #If true, elements from the previous stage are frozen
        self.generator = generator
        self.kp_detector = kp_detector

    def get_animation_optimizer(self, config={'lr':2e-4}, betas=(0.5, 0.999)):
        #No previous stage so nothing to freeze
        parameters = list(self.generator.parameters()) + list(self.kp_detector.parameters())
        optimizer = torch.optim.Adam(parameters, lr=config['lr'], betas=betas)
        return optimizer

    def get_rdac_optimizer(self, config={'lr':1e-4, 'lr_aux':1e-3},temporal=False, betas=(0.5, 0.999)):
        for param in self.generator.parameters():
            param.requires_grad = False

        for param in self.kp_detector.parameters():
            param.requires_grad = False
        parameters = []
    
        if temporal:
            tdc_net = self.generator.tdc.train()
            parameters += list([p for n, p in tdc_net.named_parameters() if not n.endswith(".quantiles")])
            aux_parameters = list([p for n, p in tdc_net.named_parameters() if n.endswith(".quantiles")])

            if self.generator.ref_fusion_net is not None:
                parameters += list([p for n, p in self.generator.ref_fusion_net.named_parameters()]) #+ list(self.generator.bottleneck.parameters()) 

        else:
            sdc_net = self.generator.sdc.train()
            parameters += list(set(p for n, p in sdc_net.named_parameters() if not n.endswith(".quantiles")))            
            aux_parameters = list(set(p for n, p in sdc_net.named_parameters() if n.endswith(".quantiles"))) 

        for param in aux_parameters:
            param.requires_grad = True
            
        for param in parameters:
            param.requires_grad = True       

        optimizer = torch.optim.AdamW(parameters, lr=config['lr'])
        aux_optimizer = torch.optim.AdamW(aux_parameters, lr=config['lr_aux'])
        return optimizer, aux_optimizer
        
    def get_rec_optimizer(self, config={'lr':1e-4, 'lr_aux':1e-3},betas=(0.5, 0.999)):
        # if self.step_wise:
        for param in self.generator.parameters():
            param.requires_grad = False

        for param in self.kp_detector.parameters():
            param.requires_grad = False


        parameters = list(self.generator.rec_network.parameters())
        for param in parameters:
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(parameters, lr=config['lr'],betas=betas)
        return optimizer

def load_pretrained_model(model_and_optimizer_arch, path , device:str='cpu', strict= False):
    out = []
    cpk = torch.load(path, map_location=device)
    # if name == 'generator':
    #     if list(cpk[name].keys())[0][:7]=='module.':
    #         from collections import OrderedDict
    #         new_state_dict = OrderedDict()
    #         for k, v in cpk[name].items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #     else:
    #         new_state_dict = cpk[name]
    #     model.load_state_dict(new_state_dict, strict=strict)
    # else:
    for item in model_and_optimizer_arch:
        if item in cpk and model_and_optimizer_arch[item] != None:
            out.append(model_and_optimizer_arch[item].load_state_dict(cpk[item], strict=strict))
        else:
            out.append(model_and_optimizer_arch[item])
    return out

def detach_value(value: torch.Tensor)->torch.Tensor:
    return value.mean().detach().data.cpu().numpy().item()

def train(config,dataset,generator, kp_detector,discriminator,**kwargs ):
    train_params = config['train_params'] 
    debug = kwargs['debug'] 
    adversarial_training = train_params['adversarial_training']

    # create optimizers for generator, kp_detector and discriminator
    step = kwargs['step']
    step_wise = config['train_params']['step_wise']
    configure = ConfigureOptimizers(generator, kp_detector, step_wise=step_wise)
    aux_optimizer = None
    #Be sure to update this in the config path so that each training step is initialized with the correct checkpoint
    pretrained_cpk_path = config['dataset_params']['cpk_path'] 
    if step == 0:
        optimizer = configure.get_animation_optimizer()    
    elif step == 1:
        # Train the spatial difference coder
        optimizer, aux_optimizer = configure.get_rdac_optimizer()
    elif step == 2:
        # Train TDC
        optimizer, aux_optimizer = configure.get_rdac_optimizer(temporal=True)     
    elif step == 3:
        # train the denoising network
        optimizer = configure.get_rec_optimizer()
    else:
        raise NotImplementedError("Unknown training step")

    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if adversarial_training:
        optimizer_discriminator = torch.optim.Adam(list(discriminator.parameters()), lr=train_params['lr_discriminator'],  betas=(0.5, 0.999))  
    else:
        optimizer_discriminator = None

    if kwargs['checkpoint'] is not None:
        start_epoch = utils.Logger.load_cpk(kwargs['checkpoint'],generator, discriminator, kp_detector,
                                    optimizer, optimizer_discriminator)
    elif pretrained_cpk_path is not None:
        #get a pretrained dac_model based on current rdac config
        model_and_optimizer_arch = {'generator': generator, 'kp_detector':kp_detector,
                  'optimizer': optimizer, 'discriminator': discriminator,
                  'disc_optimizer':optimizer_discriminator, 'aux_optimizer':aux_optimizer}
        generator,  kp_detector, optimizer, discriminator, optimizer_discriminator, aux_optimizer = load_pretrained_model(model_and_optimizer_arch,pretrained_cpk_path, device=device )

    scheduler_generator = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,last_epoch= -1) 
    generator_full = TrainerModel(kp_detector, generator, discriminator,config) 
    
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)
    discriminator_full = DiscriminatorModel(discriminator, train_params) 

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        generator_full = generator_full.cuda()
        if  torch.cuda.device_count()> 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in range(num_gpus)])
            generator_full = CustomDataParallel(generator_full)
        if adversarial_training:
            discriminator_full = discriminator_full.cuda()
            if torch.cuda.device_count()>1:
                discriminator_full = CustomDataParallel(discriminator_full)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = utils.DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)

    with utils.Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x, generated = None, None
            for x in dataloader:
                if torch.cuda.is_available():
                    for item in x:
                        x[item] = x[item].cuda()                         
                params = {**kwargs}

                params.update({'variable_bitrate':config['model_params']['generator_params']['residual_coder_params']['variable_bitrate']})
                params.update({'bitrate_levels':config['model_params']['generator_params']['residual_coder_params']['levels']})
            
                losses_generator, generated = generator_full(x, **params)
                losses_ = {} 
                if 'distortion' in generated:
                    losses_.update({'distortion': detach_value(generated['distortion']),
                                    'rate':detach_value(generated['rate'])})
                
                if 'perp_distortion' in generated:
                    losses_.update({'perp_distortion':detach_value(generated['perp_distortion'])})
                
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values) 
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                if aux_optimizer is not None:
                    if step==1:
                        aux_loss = generator.sdc.aux_loss()
                    elif step==2:
                        aux_loss = generator.tdc.aux_loss()
                    else:
                        raise NotImplementedError(f"Training step '{step}' does not include an auxilliary optimizer!")
                    aux_loss.backward()
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()
                else:
                    aux_loss = 0

                losses_discriminator = discriminator_full(x, generated)
                loss_values = [val.mean() for val in losses_discriminator.values()]
                disc_loss = sum(loss_values)
                disc_loss.backward()
                
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                losses.update({**losses_})
                if aux_loss>0:
                    losses.update({"aux_loss":aux_loss.mean().detach().data.cpu().numpy().item()})
                logger.log_iter(losses=losses)
                
                if debug:
                    break

            scheduler_generator.step()
            if adversarial_training:
                scheduler_discriminator.step()

            state_dict = {'generator': generator,
                        'kp_detector': kp_detector, 'gen_optimizer': optimizer}
            
            if discriminator is not None:
                state_dict.update({'discriminator':discriminator, 'disc_optimizer':optimizer_discriminator})

            if aux_optimizer is not None:
                state_dict.update({'aux_optimizer': aux_optimizer})

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
            

            
