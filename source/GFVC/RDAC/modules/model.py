import torch
import numpy as np
from torch import nn
from torchvision import models
from torch.autograd import grad
import torch.nn.functional as F
from GFVC.RDAC.modules.util import AntiAliasInterpolation2d, make_coordinate_grid

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class TrainerModel(nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_extractor=None, generator=None, discriminator=None, config=None):
        super(TrainerModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.num_target_frames = config['dataset_params']['num_sources']-1
        self.scales = self.train_params['scales']
        self.scale_factor = config['model_params']['generator_params']['scale_factor']
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, self.scale_factor)

        if discriminator is not None:
            self.disc_scales = self.discriminator.scales
        else:
            self.disc_scales = [1]
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)

        self.mse = nn.MSELoss()
        self.loss_weights = self.train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()

    # def downsample(self, frame, sf=0.5):
    #     return F.interpolate(frame, scale_factor=(sf, sf),mode='bilinear', align_corners=True)

    # def upsample(self, frame):
    #     return F.interpolate(frame, scale_factor=(2, 2),mode='bilinear', align_corners=True)
        

    def compute_perp_loss(self, real,generated):
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(generated)
        value_total = 0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
            for i, _ in enumerate(self.loss_weights['perceptual']):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += self.loss_weights['perceptual'][i] * value
        return value_total
             
    def compute_content_loss(self, real, generated, layer=1):
        real_fts = self.vgg(real)
        gen_fts = self.vgg(generated)
        #take layer one for low level details that are likely to be present in the residual 
        #signal after animation
        content_loss = self.mse(real_fts[layer],gen_fts[layer] )
        return content_loss
    

    def gram_matrix(self, frame):
        # get the batch size, channels, height, and width of the image
        (bs, ch, h, w) = frame.size()
        f = frame.view(bs, ch, w * h)
        G = f.bmm(f.transpose(1, 2)) / (ch * h * w)
        return G
    
    def compute_style_loss(self, real, generated):
        #we want the style of the generated frame to match the real image
        style_fts = self.vgg(real)
        gen_fts = self.vgg(generated)

        # Get the gram matrices
        style_gram = [self.gram_matrix(fmap) for fmap in style_fts]
        gen_gram = [self.gram_matrix(fmap) for fmap in gen_fts]
        style_loss = 0.0
        for idx, gram in enumerate(style_gram):
            style_loss += self.mse(gen_gram[idx], gram)
        return style_loss


    def forward(self, x, **kwargs):
        anim_params = {**x}
        if self.scale_factor != 1:
            for item in anim_params:
                anim_params[item] = self.down(anim_params[item])
        anim_params.update({'num_targets':self.num_target_frames})
        #get reference frame KP
        kp_reference = self.kp_extractor(x['reference'])
        anim_params.update({"kp_reference":kp_reference})

        use_base_layer = self.config['dataset_params']['base_layer']  
        if kwargs['variable_bitrate']:
            rate_idx = np.random.choice(range(kwargs['bitrate_levels']))
            rd_lambda_value = self.config['train_params']['rd_lambda'][rate_idx]
        else:
            rate_idx =  self.config['train_params']['target_rate']
            rd_lambda_value  = self.config['train_params']['rd_lambda'][rate_idx]
        anim_params.update({f'rd_lambda_value': rd_lambda_value , f'rate_idx':rate_idx})

        kp_targets ={}
        for idx in range(self.num_target_frames):
            kp_target_prev = self.kp_extractor(x[f'target_{idx}'])
            kp_targets.update({f'kp_target_{idx}': kp_target_prev})
        
        anim_params.update(**kp_targets)
        B,C,H,W = x['reference'].shape
        
        generated = self.generator(**anim_params)
        # print(generated.keys())
        generated.update({'kp_reference': kp_reference, **kp_targets})
        loss_values = {}

        rd_loss = 0.0
        rate = 0.0
        perceptual_loss = 0.0
        enh_perp_loss = 0.0
        perp_distortion = 0.0
        style_loss = 0.0
        content_loss = 0.0
        gen_gan = 0.0
        feature_matching = 0.0
        equivariance_value = 0.0
        equivariance_jacobian = 0.0
        
        for idx in range(self.num_target_frames):
            tgt = self.num_target_frames-1 
            
            target = x[f'target_{idx}']
            # print("Target: ", target.shape)
            
            if f'sr_prediction_{idx}' in generated:
                prediction = generated[f'sr_prediction_{idx}']
            elif f'enhanced_prediction_{idx}' in generated:
                prediction = generated[f'enhanced_prediction_{idx}']
                # print("Using enhanced pred: ", idx)
            else:
                prediction = generated[f'prediction_{idx}']

            if use_base_layer:
                lambda_value = x[f'lambda_value'].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                perceptual_loss +=  lambda_value * self.compute_perp_loss(target,prediction)
            else:
                perceptual_loss += self.compute_perp_loss(target,prediction)
            
            #compute rd loss
            if f'rate_{idx}' in generated:
                perp_distortion = self.compute_perp_loss(target,prediction)
                rd_loss += anim_params[f'rd_lambda_value'] * perp_distortion + generated[f"rate_{idx}"]
                if f'wdist_{idx}' in generated:
                    rd_loss += 10*generated[f"wdist_{idx}"]
                rate += generated[f"rate_{tgt}"]

            if self.loss_weights['style_loss'] != 0:
                style_loss += self.compute_style_loss(target, prediction)

            #compute the gan losses
            pyramide_real = self.pyramid(target)
            pyramide_generated = self.pyramid(prediction)
                        
            if self.loss_weights['generator_gan'] != 0 and self.discriminator != None:
                discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_targets[f"kp_target_{idx}"]))
                discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_targets[f"kp_target_{idx}"]))

                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    if not torch.isnan(value):
                        value_total += self.loss_weights['generator_gan'] * value
                gen_gan += value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            if not torch.isnan(value) and not torch.isinf(value):
                                value_total += self.loss_weights['feature_matching'][i] * value
                        feature_matching += value_total

            if  (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
                transform = Transform(target.shape[0], **self.train_params['transform_params'])
                transformed_frame = transform.transform_frame(target)
                transformed_kp = self.kp_extractor(transformed_frame)

                ## Value loss part
                if self.loss_weights['equivariance_value'] != 0:
                    value = torch.abs(kp_targets[f"kp_target_{idx}"]['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                    equivariance_value += self.loss_weights['equivariance_value'] * value

                ## jacobian loss part
                if self.config['model_params']['common_params']['estimate_jacobian'] and self.loss_weights['equivariance_jacobian'] != 0:
                    jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                        transformed_kp['jacobian'])

                    normed_target = torch.inverse(kp_targets[f"kp_targets_{idx}"]['jacobian'])
                    normed_transformed = jacobian_transformed
                    value = torch.matmul(normed_target, normed_transformed)

                    eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                    value = torch.abs(eye - value).mean()
                    equivariance_jacobian += self.loss_weights['equivariance_jacobian'] * value
            
        loss_values['perceptual'] = perceptual_loss/self.num_target_frames
        if rd_loss>0:
            loss_values['rd_loss'] = rd_loss/self.num_target_frames
            # generated['perp_distortion'] = (enh_perp_loss)/self.num_target_frames
            generated['rate'] = rate/self.num_target_frames

        if gen_gan >0:
            loss_values['gen_gan'] = gen_gan/self.num_target_frames

        if feature_matching >0:
            loss_values['feature_matching'] = feature_matching/self.num_target_frames

        if equivariance_value > 0:
            loss_values['equivariance_value'] = equivariance_value/self.num_target_frames

        if equivariance_jacobian>0:
            loss_values['equivariance_jacobian'] = equivariance_jacobian/self.num_target_frames


        if style_loss>0:
            loss_values['style_loss'] = style_loss/self.num_target_frames
        
        if content_loss>0:
            loss_values['content_loss'] = content_loss/self.num_target_frames

        return loss_values, generated


class DiscriminatorModel(nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    def __init__(self, discriminator=None, train_params=None, **kwargs):
        super(DiscriminatorModel, self).__init__()
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def compute_multiscale(self, real, decoded, kp_target):
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(decoded)
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))

        loss = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            loss += self.loss_weights['discriminator_gan'] * value.mean()
        return loss

    def forward(self, x, generated):
        loss = 0.0
        num_targets = len([tgt for tgt in x.keys() if 'target' in tgt])
        for idx in range(num_targets):
            real = x[f'target_{idx}']
            if f'sr_prediction_{idx}' in generated:
                prediction = generated[f'sr_prediction_{idx}'].detach()
            elif 'enhanced_prediction_0' in generated:
                prediction = generated[f'enhanced_prediction_{idx}'].detach()
            else:
                prediction = generated[f'prediction_{idx}'].detach()

            loss += self.compute_multiscale(real, prediction, generated[f'kp_target_{idx}'])
        
        loss = (loss/num_targets)*self.train_params['loss_weights']['discriminator_gan']
        return {'disc_gan': loss}
