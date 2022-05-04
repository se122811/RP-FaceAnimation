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
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)

        if is_train:
            # training parameters
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
            parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

            
            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser


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
        
  
        output_coeff = self.net_recon(self.input_img)
        # ind = int(self.output_coeff.shape[0]/2)
        
        # source_coeff = self.output_coeff[:ind,:]
        # target_coeff = self.output_coeff[ind:,:]
        # temp = source_coeff.clone()
        # source_coeff[:,80:144] = target_coeff[:,80:144]
        # source_coeff[:,224:227] = target_coeff[:,224:227]
        # source_coeff[:,254:257] = target_coeff[:,254:257]
        
        # target_coeff[:,80:144] = temp[:,80:144]
        # target_coeff[:,224:227] = temp[:,224:227]
        # target_coeff[:,254:257] = temp[:,254:257]
        
        # output_coeff = torch.cat((source_coeff,target_coeff),0)
        
        
        self.facemodel.to(self.device)

        # face_proj --> rendering을 위한 vertex            
        # self.pred_coeff = self.pred_coeff[:,:257]
        
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm, self.face_proj = \
            self.facemodel.compute_for_render(output_coeff)

        # self.pred_mask, _, self.pred_face = self.renderer(
        #     self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)  


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


        return  self.output_vis_pred, self.pred_vertex, self.facemodel.face_buf.repeat(B,1,1), self.pred_mask



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
    
        # input_img = (1 - self.pred_mask) * self.input_img
        
        
        # input_img = input_img.detach().cpu().permute(1,2,0).numpy()
        # input_img = input_img[:,:,::-1]
        # input_img = input_img * 255
        
        
        # pred_mask = self.pred_mask.detach().cpu().permute(1,2,0).numpy()
        # img = img[0, ..., :3].cpu().numpy()
        # img = img * pred_mask
        
        
        # img = img + input_img

        
        
        # img = img[0, ..., :3].cpu().numpy()
        # img = img[:,:,::-1]
        # img = img*255

        # cv2.imwrite(name, img)
        
    
        
        # res 

        # face_proj = self.face_proj.clone().squeeze(0).detach().cpu().numpy()
        
        # x = face_proj[:, 0]
        # y = face_proj[:, 1]


        # y_temp1 = y
        # y_temp2 = y
        # y_final = y


        # y_temp1_idx = np.argwhere(y_temp1>128)
        # y_temp1 = np.where(y_temp1>128, y_temp1 - 2*(y_temp1-128), y_temp1)
        # y_temp2_idx = np.argwhere(y_temp2<=128)
        # y_temp2 = np.where(y_temp2<=128, 128+(128-y_temp2), y_temp2)
        

        # y_final[y_temp1_idx.squeeze(1)] = y_temp1[y_temp1_idx.squeeze(1)]
        # y_final[y_temp2_idx.squeeze(1)] = y_temp2[y_temp2_idx.squeeze(1)]


        # colors = bilinear_interpolate(self.crop_input_img,x, y_final) / 255.
        # # colors = bilinear_interpolate(img, face_proj[:, 0], face_proj[:, 1]) / 255.
        
        # # `rasterize` here serves as texture sampling, may need to optimization
        
        # res_img = rasterize(uv_coords, face_index, colors, height=uv_h, width=uv_w, channel=uv_c)
        # res_lst.append(res)

        # # if wfp is not None:
        # cv2.imwrite(name, res_img)
        # print(f'Save visualization result to {wfp}')

        #return res, res_img
        
        
        

    


    
    # def uv_map(self, name):
        
    #     # (35709,2)
    #     uv_pos = transfer_UV()

    #     # (35790,3)
    #     tex_coords = process_uv(uv_pos.copy())
    #     tex_coords = torch.tensor(tex_coords, dtype=torch.float32).unsqueeze(0).to(self.device) 

    #     # face_texture = self.pred_tex / 255.0
    #     face_texture= torch.unsqueeze(self.pred_tex[0], 0)

    #     images = render_img(tex_coords, face_texture, self.bfm, 256, 256.0 - 1.0, 256.0 - 1.0, 0.0, 0.0)
    #     # images = render_img(tex_coords, face_texture, self.bfm, 3500, 3500.0 -s 1.0, 3500.0 - 1.0, 0.0, 0.0)
        
    #     images = images.detach().cpu().numpy()
    #     images = np.squeeze(images)

    #     # from PIL import Image
    #     images = np.uint8(images[:, :, :3] * 255.0)
    #     img = Image.fromarray(images)


    #     img.save(name)