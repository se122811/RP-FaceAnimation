"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from util import util 
# from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch
import cv2
import os
from .uv_utils import *
 

import trimesh
from scipy.io import savemat
from Sim3DR import rasterize


from .load_data import transfer_UV, transfer_BFM09, BFM, load_img, Preprocess, save_obj#, process_uv
from .reconstruction_mesh import reconstruction, render_img, transform_face_shape, estimate_intrinsic
from PIL import Image


from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    blending
)




class FaceReconModel(BaseModel):


    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        

        self.bfm = BFM(r'BFM/BFM_model_front.mat', self.device)
        self.lm3D = self.bfm.load_lm3d()
        self.pi_isTrain = opt.pi_isTrain
        self.cross_id = opt.cross_id
        self.Isaudio = opt.Isaudio
        

        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        
        # fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        # self.renderer = MeshRenderer(
        #     rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center))


        self.img_size = 256
        self.focal = 1015
        self.renderer = self.get_renderer(self.device)
        
        

        if self.isTrain:
            self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
                )
            # loss func name: (compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss
            self.comupte_color_loss = photo_loss
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = reflectance_loss

            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

   

    def get_renderer(self, device):
        R= torch.Tensor([[[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        cameras = FoVPerspectiveCameras(device=device, R=R, znear=5., zfar=15.,
                                        fov=2*np.arctan(self.opt.center/self.focal)*180./np.pi)

        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 

        if self.pi_isTrain == False:
            self.source_img = input['source_imgs'].to(self.device)
            
            
            if self.Isaudio == True:
                self.audio_input = input['audio_source'].to(self.device)
                self.audio_first = input['audiio_first'].to(self.device)
                ###시작점 차이###
                self.frame_index = input["frame_index"]
                if self.frame_index != 0:
                    self.output_coeff_first = input['source_coeff_first'].to(self.device)
            # if input['first_target_coeff'] != None:
            #     self.first_target_coeff = input['first_target_coeff'].to(self.device)
            #     self.first = False
            # else:
            #     self.first = True

            
        
        # self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        # self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        # self.trans_m = input['M'].to(self.device) if 'M' in input else None
        # self.image_paths = input['im_paths'] if 'im_paths' in input else None


        """
            id    :   0:80  ; 80
            exp   :  80:144 ; 64 --> expression(target)
            tex   : 144:224 ; 80
            angle : 224:227 ; 3 --> rotation(target)
            gamma : 227:254 ; 27
            trans : 254:257 ; 3 --> translation(target)
            crop  : 257:260 : 3
        """
        
    def forward(self):
        self.renderer = self.get_renderer(self.device)
        # self.uv_renderer = self.get_uv_render(self.device)
        
        if self.pi_isTrain:
            self.output_coeff = self.net_recon(self.input_img)
            # input_source_coeff  = self.output_coeff
            ind = int(self.output_coeff.shape[0]/2)
            
            source_coeff = self.output_coeff[:ind,:]
            target_coeff = self.output_coeff[ind:,:]
            temp = source_coeff.clone()
            
            source_coeff[:,80:144] = target_coeff[:,80:144]
            source_coeff[:,224:227] = target_coeff[:,224:227]
            source_coeff[:,254:257] = target_coeff[:,254:257]
            
            target_coeff[:,80:144] = temp[:,80:144]
            target_coeff[:,224:227] = temp[:,224:227]
            target_coeff[:,254:257] = temp[:,254:257]
            
            
            output_coeff = torch.cat((source_coeff,target_coeff),0)
            
        
        else:
            """
            source 이미지를 pred mask 제외한 이미지로 넣었을 때 inference 할 코드 구현하기
            """
            ### target image coeff ###
            self.target_coeff = self.net_recon(self.input_img)
            ### source image coeff ###
            self.source_coeff = self.net_recon(self.source_img)
            
            if self.Isaudio == True:
                self.trans_coeff_delta = self.audio_input - self.audio_first
                output_coeff  = self.source_coeff.clone()    
                ########## expression no delta 
                output_coeff[:,80:144] = self.audio_input[:,:64]
                
                #### pose targe에서 가져오기 ####
                output_coeff[:,224:227] = self.target_coeff[:,224:227]
                output_coeff[:,254:257] = self.target_coeff[:,254:257]
                self.source_coeff_first = output_coeff
                ### angle no detla ###
                # output_coeff[:,224:227] = self.audio_input[:,64:67]
                
                # if self.frame_index == 0:
                #     output_coeff[:,254:257] = output_coeff[:,254:257]
                #     self.source_coeff_first = output_coeff
                # else :  
                    
                #     output_coeff[:,254:257] = self.output_coeff_first[:,254:257] + self.trans_coeff_delta[:,67:]
                #     ### angle detla ###
                #     output_coeff[:,224:227] = self.output_coeff_first[:,224:227] + self.trans_coeff_delta[:,64:67]
                #     ### expression delta ####
                #     # output_coeff[:,80:144] = self.output_coeff_first[:,80:144] + self.trans_coeff_delta[:,:64]
            else :
                if self.cross_id == False:
                    #warp이미지를 위해 뽑는 input_source_coeff 
                    output_coeff  = self.source_coeff.clone()
                    output_coeff[:,80:144] = self.target_coeff[:,80:144]
                    output_coeff[:,224:227] = self.target_coeff[:,224:227]
                    output_coeff[:,254:257] = self.target_coeff[:,254:257]
                
                
                else:
                    ### source image - coeff 뽑기 ###
                    output_coeff  = self.source_coeff.clone()
                    # crop_norm_ratio = self.find_crop_norm_ratio(self.source_coeff.detach().cpu().numpy(), self.target_coeff.detach().cpu().numpy())
                    
                    output_coeff[:,80:144] = self.target_coeff[:,80:144]
                    output_coeff[:,224:227] = self.target_coeff[:,224:227]
                    output_coeff[:,254:256] = self.target_coeff[:,254:256]
                    
                    # output_coeff = self.source_coeff.clone()
                    # output_coeff[:,254:257] = self.target_coeff[:,254:257] * crop_norm_ratio[0]
                    
                    # if self.first == True:
                    #     output_coeff[:,254:256] = self.target_coeff[:,254:256]
                    # else:
                    #     self.trans_coeff_delta = self.target_coeff[:,254:257] - self.first_target_coeff[0,254:257]
                    #     output_coeff[:,254:257] = output_coeff[:,254:257] + self.trans_coeff_delta 
        
        '''
        if self.pi_isTrain:
            output_coeff = self.net_recon(self.input_img)        
        
        
        else:
            self.output_coeff = self.net_recon(self.source_img)
            output_coeff = self.output_coeff.repeat(self.target_coeff.shape[0],1)
        '''
            
            
        self.facemodel.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm, self.face_proj = \
            self.facemodel.compute_for_render(output_coeff)
        
        
        # self.source_pred_vertex, self.source_pred_tex, self.source_pred_color, self.source_pred_lm, self.source_face_proj = \
        #     self.facemodel.compute_for_render(output_coeff)



        #self.facemodel_buf ==> face indexmodel_buf ==> face index
        B,N,C = self.pred_vertex.shape
        deep3d_renderer_color = TexturesVertex(self.pred_color)
        deep3d_renderer_mesh = Meshes(self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), deep3d_renderer_color)
        deep3d_rendered_img = self.renderer(deep3d_renderer_mesh)
        deep3d_rendered_img = torch.clamp(deep3d_rendered_img,0,255)
        self.pred_mask = (deep3d_rendered_img[...,3]>0).float().unsqueeze(1)
        self.pred_face = deep3d_rendered_img.permute(0,3,1,2).contiguous()[:,:3,:,:]
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)
        
        B,N,C = self.pred_vertex.shape
        deep3d_renderer_color = TexturesVertex(self.pred_color)
        deep3d_renderer_mesh = Meshes(self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), deep3d_renderer_color)
        deep3d_rendered_img = self.renderer(deep3d_renderer_mesh)
        deep3d_rendered_img = torch.clamp(deep3d_rendered_img,0,255)
        self.pred_mask = (deep3d_rendered_img[...,3]>0).float().unsqueeze(1)
        self.pred_face = deep3d_rendered_img.permute(0,3,1,2).contiguous()[:,:3,:,:]
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)
      

    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        
     
        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, self.atten_mask * face_mask)

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)

        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma \
                        + self.loss_lm + self.loss_reflc
            

    def optimize_parameters(self, isTrain=True):
        self.forward()               
        self.compute_losses()

        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.opttimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        



    def compute_visuals(self):
        # basename = os.path.basename(self.image_paths[0]) 
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()

            # self.crop_input_img =  input_img_numpy.squeeze(0)
            # self.crop_input_img = self.crop_input_img[:,:,::-1]

            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            self.output_vis_pred = output_vis_numpy_raw

            output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            # if self.gt_lm is not None:
            #     gt_lm_numpy = self.gt_lm.cpu().numpy()
            #     pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
            #     output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
            #     output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            #     output_vis_numpy = np.concatenate((input_img_numpy, 
            #                         output_vis_numpy_raw, output_vis_numpy), axis=-2)
            # else:
            #     output_vis_numpy = np.concatenate((input_img_numpy, 
            #                         output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
            B,N,C = self.pred_vertex.shape

        if self.pi_isTrain:
            return  self.output_vis_pred, self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), self.pred_mask
        else:
            if self.Isaudio == True:
                if self.frame_index == 0: 
                    return  self.output_vis_pred, self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), self.pred_mask, self.target_coeff , self.source_coeff_first
                else:
                    return  self.output_vis_pred, self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), self.pred_mask, self.target_coeff
            else:
                return  self.output_vis_pred, self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), self.pred_mask, self.target_coeff


    def save_mesh(self, name):
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)



    def save_coeff(self,name):
        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)




    def _to_ctype(arr):
        if not arr.flags.c_contiguous:
            return arr.copy(order='C')
        return arr

    
    
    def uvmap_to_render(self, uv_map,  uv_h =256, uv_w=256, uv_c = 3, show_flag=False, wfp=None):     
        # (53215,2)
        g_uv_coords = load_uv_coords('BFM/BFM_UV.mat')
        # (35709,1) --> 53215에서 가져올 index만
        indices = load_idx("BFM/BFM_front_idx.mat") #todo: handle bfm_slim
        # (35709,)
        indices = np.squeeze(indices,1)
        # 해당하는 index에 있는 값들만 가져오기 --> g_uv_coords 
        g_uv_coords = g_uv_coords[indices, :]
        
       

        # if wfp is not None:
        # cv2.imwrite(name, res)
        # print(f'Save visualization result to {name}')
        
        # write_obj_with_colors(name, vertices,face_index, textures)
        g_uv_coords_tensor = torch.Tensor(g_uv_coords).to(self.device)
        res_tensor = uv_map.copy()
        res_tensor = torch.Tensor(res_tensor).to(self.device)
        
        B,N,C = self.pred_vertex.shape
        res_tensor = res_tensor.unsqueeze(0)
        res_tensor = res_tensor.repeat(B,1,1,1)
        
        #UV_Map으로 rendering 하기. 
        texture_img = TexturesUV(res_tensor, self.facemodel.face_buf.repeat(B,1,1), g_uv_coords_tensor.repeat(B,1,1))
        rerender_mesh = Meshes(self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), texture_img)
        rerender_img = self.renderer(rerender_mesh)
        
        #obj 파일 저장하기 
        # obj_name = write_obj_with_colors_texture2(name, vertices, res, face_index, res[:,:,::-1] , g_uv_coords)
        # meshes = uv_rendering(self.device, obj_name)
        # img = self.renderer(meshes.to(self.device))
        
        rerender_img = torch.clamp(rerender_img,0,255)
        self.rerender_face = rerender_img.permute(0,3,1,2).contiguous()[:,:3,:,:]
        rerender_output_vis = self.rerender_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
        rerender_output_vis = 255. * rerender_output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        
        return rerender_output_vis


    def uv_map_3ddfa(self, uv_h =256, uv_w=256, uv_c = 3, show_flag=False, wfp=None):   
        res_lst = []

        # textures = self.pred_tex[0].detach().cpu().numpy()
        textures = self.pred_tex.detach().cpu().numpy()

        # textures = np.squeeze(textures,0)
        
        face_index = self.facemodel.face_buf.detach().cpu().numpy()
        face_index = face_index.copy(order='C')
        face_index = face_index.astype(np.int32)


        a_temp = face_index[:,1].copy()
        b_temp = face_index[:,2].copy()
        face_index[:,1] = b_temp
        face_index[:,2] = a_temp
        
    
        # (53215,2)
        g_uv_coords = load_uv_coords('BFM/BFM_UV.mat')
        # (35709,1) --> 53215에서 가져올 index만
        indices = load_idx("BFM/BFM_front_idx.mat") #todo: handle bfm_slim
        # (35709,)
        indices = np.squeeze(indices,1)
        # 해당하는 index에 있는 값들만 가져오기 --> g_uv_coords 
        g_uv_coords = g_uv_coords[indices, :]
        # (35709,3)
        uv_coords = process_uv(g_uv_coords.copy(), uv_h=uv_h, uv_w=uv_w)
        
        
        for i in range(textures.shape[0]):
            tex = textures[i].copy(order='C')
            res = rasterize(uv_coords, face_index, tex, height=uv_h, width=uv_w, channel=uv_c)
            res = np.expand_dims(res, axis=0)
            res_lst.append(res)

        res = np.concatenate(res_lst, axis=0) if len(res_lst) > 1 else res_lst[0]


        return res
