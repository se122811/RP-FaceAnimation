import math

import torch
from torch._C import device

from trainers.base_gan import BaseTrainer
from util.trainer import accumulate, get_optimizer
from loss.perceptual  import PerceptualLoss
from torch.autograd import Variable

import os
import sys

sys.path.append("/data/PIRender_hs")
sys.path.append("/data/PIRender_hs/Deep3DFaceRecon_pytorch")

from Deep3DFaceRecon_pytorch.models import create_model




class FaceTrainer(BaseTrainer):
    r"""Initialize lambda model trainer.

    Args:
        cfg (obj): Global configuration.    
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """
   
    def __init__(self, opt, 
                 net_G, net_G_ema, opt_G, sch_G,  
                 net_D, net_D_ema, opt_D, sch_D,
                 train_data_loader, val_data_loader=None):
       
        super(FaceTrainer, self).__init__(opt, net_G, net_G_ema, opt_G, sch_G, net_D, net_D_ema, opt_D, sch_D, train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.deep3d_opt = opt.Deep3DRecon_pytorch
        self.opt = opt
        self.device_index = torch.cuda.current_device()
        self.device = torch.device("cuda:"+str(self.device_index))
        
        
        # Deep 3d model 불러오기 
        self.model = create_model(self.deep3d_opt)
        self.model.setup(self.deep3d_opt)
        self.model.device = self.device
        self.model.parallelize()
        self.model.eval()
        self.adversarial_loss = torch.nn.BCELoss()
        
        
        
        

    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_warp',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp)
 
 
        self._assign_criteria(
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)
        
        
        self._assign_criteria(
            'perceptual_rendering',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)


  
    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight


    def optimize_parameters(self, data):
        self.gen_losses = {}
        opt = self.deep3d_opt
        device = self.device
        
        source_image, target_image = data['source_image'], data['target_image']
        source_image_3dmm = data['source_image_3dmm'] 
        target_image_3dmm = data['target_image_3dmm'] 
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']
        source_coeff = data['source_coeff']
        target_coeff = data['target_coeff']
        
        coeff = torch.cat((source_coeff,target_coeff),0)
        imgs_3dmm = torch.cat((source_image_3dmm,target_image_3dmm))        
        coeff = coeff[:,:257]
        
        data_input = {                
                        'pred_coeff': coeff,
                        'imgs': imgs_3dmm
                    }
        
        self.model.set_input(data_input)  
        self.model.test()
        _ = self.model.get_current_visuals()  # get image results
        _, pred_vertex, tri, pred_mask = self.model.compute_visuals()
        
        uv_map = self.model.uv_map_3ddfa()
        uv_map_input= torch.Tensor(uv_map.copy()).to(device)
      
        uv_map_input = uv_map_input.permute(0,3,1,2)
        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0) 

        output_dict = self.net_G(opt, input_image, uv_map_input, pred_vertex, tri, pred_mask, input_semantic, self.training_stage)
        

        if self.training_stage == 'gen':
            fake_img = output_dict['fake_image']
            warp_img = output_dict['warp_image']
            render_img = output_dict['uv_render_img']
            self.gen_losses["perceptual_final"] = self.criteria['perceptual_final'](fake_img, gt_image)
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)
            self.gen_losses["perceptual_rendering"] = self.criteria['perceptual_rendering'](render_img, input_image)
            
        else:
            warp_img = output_dict['warp_image']
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss
        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()
        
        
        """
        Discriminator
        """
        self.valid = Variable(torch.cuda.FloatTensor(10, 1).fill_(1.0), requires_grad=False)
        self.fake = Variable(torch.cuda.FloatTensor(10, 1).fill_(0.0), requires_grad=False)
        self.net_D.zero_grad()
        output_input_D = self.net_D(input_image)
        output_render_D = self.net_D(render_img)
        d_loss = 0
        real_loss = self.adversarial_loss(output_input_D, self.valid)
        fake_loss = self.adversarial_loss(output_render_D, self.fake)
        d_loss = (real_loss+fake_loss)/2
        
        d_loss.backward(retain_graph=True)
        self.opt_D.step()
        
        accumulate(self.net_G_ema, self.net_G_module, self.net_D_ema, self.net_D_module, self.accum)
        

    def _start_of_iteration(self, data, current_iteration):
        self.training_stage = 'gen' if current_iteration >= self.opt.trainer.pretrain_warp_iteration else 'warp'
        
        
        if current_iteration == self.opt.trainer.pretrain_warp_iteration:
            self.reset_trainer()
        return data

    def reset_trainer(self):
        self.opt_G = get_optimizer(self.opt.gen_optimizer, self.net_G.module)

    
    

    def _get_visualizations(self,data):
        # source_image, target_image = data['source_image'], data['target_image']
        # source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        # input_image = torch.cat((source_image, target_image), 0)
        # input_semantic = torch.cat((target_semantic, source_semantic), 0)   
        opt = self.deep3d_opt
        device = self.device
        
        source_image, target_image = data['source_image'], data['target_image']
        source_image_3dmm, target_image_3dmm= data["source_image_3dmm"], data['target_image_3dmm']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']
        source_coeff = data['source_coeff']
        target_coeff = data['target_coeff']
        coeff = torch.cat((source_coeff,target_coeff),0)
        imgs_3dmm = torch.cat((source_image_3dmm, target_image_3dmm))        
        
        # get test options
       
        data_input = {                
                        'pred_coeff': coeff,
                        'imgs': imgs_3dmm
                    }
        
        self.model.set_input(data_input)  
        self.model.test()
        _ = self.model.get_current_visuals()  # get image results
        _, pred_vertex, tri, pred_mask = self.model.compute_visuals()
        
        uv_map = self.model.uv_map_3ddfa()
        uv_map_input= torch.Tensor(uv_map.copy()).to(device)
        uv_map_input = uv_map_input.permute(0,3,1,2)
        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)
        _ = torch.cat((target_image, source_image), 0) 

             
             
        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
               opt, input_image, uv_map_input, pred_vertex, tri, pred_mask, input_semantic, self.training_stage)
            
            if self.training_stage == 'gen':
                fake_img = torch.cat([output_dict['warp_image'], output_dict['uv_render_img'], output_dict['fake_image']], 3)
            else:
                fake_img = output_dict['warp_image']

            # render_img = output_dict['uv_render_img']
            fake_source, fake_target = torch.chunk(fake_img, 2, dim=0)
            sample_source = torch.cat([source_image, fake_source, target_image], 3)
            sample_target = torch.cat([target_image, fake_target, source_image], 3)                    
            sample = torch.cat([sample_source, sample_target], 2)
            sample = torch.cat(torch.chunk(sample, sample.size(0), 0)[:3], 2)
            
        
        return sample
    
    

    def test(self, data_loader, output_dir, current_iteration=-1):
        pass

    def _compute_metrics(self, data, current_iteration):
        if self.training_stage == 'gen':
            # source_image, target_image = data['source_image'], data['target_image']
            # source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

            # input_image = torch.cat((source_image, target_image), 0)
            # input_semantic = torch.cat((target_semantic, source_semantic), 0)        
            opt = self.deep3d_opt
            device = self.device
            
            source_image, target_image = data['source_image'], data['target_image']
            source_image_3dmm, target_image_3dmm= data["source_image_3dmm"], data['target_image_3dmm']
            source_semantic, target_semantic = data['source_semantics'], data['target_semantics']
            source_coeff = data['source_coeff']
            target_coeff = data['target_coeff']
            coeff = torch.cat((source_coeff,target_coeff),0)
            imgs_3dmm = torch.cat((source_image_3dmm, target_image_3dmm))        
            
            # get test options
        
            data_input = {                
                            'pred_coeff': coeff,
                            'imgs': imgs_3dmm
                        }
            
            self.model.set_input(data_input)  
            self.model.test()
            _ = self.model.get_current_visuals()  # get image results
            _, pred_vertex, tri, pred_mask = self.model.compute_visuals()
            
            uv_map = self.model.uv_map_3ddfa()
            uv_map_input= torch.Tensor(uv_map.copy()).to(device)
            uv_map_input = uv_map_input.permute(0,3,1,2)
            input_image = torch.cat((source_image, target_image), 0)
            input_semantic = torch.cat((target_semantic, source_semantic), 0)
            gt_image = torch.cat((target_image, source_image), 0) 
            
            
            input_image = torch.cat((source_image, target_image), 0)
            input_semantic = torch.cat((target_semantic, source_semantic), 0)
            gt_image = torch.cat((target_image, source_image), 0)        
            metrics = {}


            with torch.no_grad():
                self.net_G_ema.eval()
                output_dict = self.net_G_ema(
                      opt, input_image, uv_map_input, pred_vertex, tri, pred_mask, input_semantic, self.training_stage)
                    
                fake_image = output_dict['fake_image']
                metrics['lpips'] = self.lpips(fake_image, gt_image).mean()
            return metrics
        
        
        
    
