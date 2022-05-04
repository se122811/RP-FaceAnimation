"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from util import util 
from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch

import trimesh
from scipy.io import savemat

from .load_data import transfer_UV,process_uv,transfer_BFM09, BFM, load_img, Preprocess, save_obj
from .reconstruction_mesh import reconstruction, render_img, transform_face_shape, estimate_intrinsic
from PIL import Image
import pickle
from menpo.image import BooleanImage, MaskedImage
from menpo.transform.piecewiseaffine.base import barycentric_vectors
from menpo3d.rasterize import rasterize_barycentric_coordinate_images,rasterize_mesh_from_barycentric_coordinate_images
from menpo.shape import TexturedTriMesh
import menpo.io as mio


def pixels_to_check(start, end, n_pixels):
    pixel_locations = np.empty((n_pixels, 2), dtype=int)
    tri_indices = np.empty(n_pixels, dtype=int)

    n = 0
    for i, ((s_x, s_y), (e_x, e_y)) in enumerate(zip(start, end)):
        for x in range(s_x, e_x):
            for y in range(s_y, e_y):
                pixel_locations[n] = (x, y)
                tri_indices[n] = i
                n += 1

    return pixel_locations, tri_indices



def pixel_locations_and_tri_indices(mesh):
    vertex_trilist = mesh.points[mesh.trilist]
    start = np.floor(vertex_trilist.min(axis=1)[:, :2])
    end = np.ceil(vertex_trilist.max(axis=1)[:, :2])
    start = start.astype(int)
    end = end.astype(int)
    n_sites = np.product((end - start), axis=1).sum()
    return pixels_to_check(start, end, n_sites)


def z_values_for_bcoords(mesh, bcoords, tri_indices):
    return mesh.barycentric_coordinate_interpolation(
        mesh.points[:, -1][..., None], bcoords, tri_indices
    )[:, 0]


def alpha_beta(i, ij, ik, points):
    ip = points - i
    dot_jj = np.einsum("dt, dt -> t", ij, ij)
    dot_kk = np.einsum("dt, dt -> t", ik, ik)
    dot_jk = np.einsum("dt, dt -> t", ij, ik)
    dot_pj = np.einsum("dt, dt -> t", ip, ij)
    dot_pk = np.einsum("dt, dt -> t", ip, ik)

    d = 1.0 / (dot_jj * dot_kk - dot_jk * dot_jk)
    alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d
    beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d
    return alpha, beta  


def xy_bcoords(mesh, tri_indices, pixel_locations):
    i, ij, ik = barycentric_vectors(mesh.points[:, :2], mesh.trilist)
    i = i[:, tri_indices]
    ij = ij[:, tri_indices]
    ik = ik[:, tri_indices]
    a, b = alpha_beta(i, ij, ik, pixel_locations.T)
    c = 1 - a - b
    bcoords = np.array([c, a, b]).T
    return bcoords

def tri_containment(bcoords):
    alpha, beta, _ = bcoords.T
    return np.logical_and(np.logical_and(alpha >= 0, beta >= 0), alpha + beta <= 1)


def rasterize_barycentric_coordinates(mesh, image_shape):
        height, width = int(image_shape[0]), int(image_shape[1])
        # 1. Find all pixel-sites that may need to be rendered to
        #    + the triangle that may partake in rendering
        yx, tri_indices = pixel_locations_and_tri_indices(mesh)

        # 2. Limit to only pixel sites in the image
        out_of_bounds = np.logical_or(
            np.any(yx < 0, axis=1), np.any((np.array([height, width]) - yx) <= 0, axis=1)
        )
        in_image = ~out_of_bounds
        yx = yx[in_image]
        tri_indices = tri_indices[in_image]

        # # Optionally limit to subset of pixels
        # if n_random_samples is not None:
        #     # 2. Find the unique pixel sites
        #     xy_u = unique_locations(yx, width, height)
        #
        #     xy_u = pixel_sample_uniform(xy_u, n_random_samples)
        #     to_keep = np.in1d(location_to_index(yx, width),
        #                       location_to_index(xy_u, width))
        #     yx = yx[to_keep]
        #     tri_indices = tri_indices[to_keep]

        bcoords = xy_bcoords(mesh, tri_indices, yx)

        # check the mask based on triangle containment
        in_tri_mask = tri_containment(bcoords)

        # use this mask on the pixels
        yx = yx[in_tri_mask]
        bcoords = bcoords[in_tri_mask]
        tri_indices = tri_indices[in_tri_mask]

        # Find the z values for all pixels and calculate the mask
        z_values = z_values_for_bcoords(mesh, bcoords, tri_indices)

        # argsort z from smallest to biggest - use this to sort all data
        sort = np.argsort(z_values)
        yx = yx[sort]
        bcoords = bcoords[sort]
        tri_indices = tri_indices[sort]

        # make a unique id per-pixel location
        pixel_index = yx[:, 0] * width + yx[:, 1]
        # find the first instance of each pixel site by depth
        _, z_buffer_mask = np.unique(pixel_index, return_index=True)

        # mask the locations one last time
        yx = yx[z_buffer_mask]
        bcoords = bcoords[z_buffer_mask]
        tri_indices = tri_indices[z_buffer_mask]
        return yx, bcoords, tri_indices


def rasterize_barycentric_coordinate_images(mesh, image_shape):
        h, w = image_shape
        yx, bcoords, tri_indices = rasterize_barycentric_coordinates(mesh, image_shape)

        tri_indices_img = np.zeros((1, h, w), dtype=int)
        bcoords_img = np.zeros((3, h, w))
        mask = np.zeros((h, w), dtype=np.bool)
        mask[yx[:, 0], yx[:, 1]] = True
        tri_indices_img[:, yx[:, 0], yx[:, 1]] = tri_indices
        bcoords_img[:, yx[:, 0], yx[:, 1]] = bcoords.T

        mask = BooleanImage(mask)
        return (
            MaskedImage(bcoords_img, mask=mask.copy(), copy=False),
            MaskedImage(tri_indices_img, mask=mask.copy(), copy=False),
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
        
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

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
        
        
        self.tcoords_full = mio.import_pickle('/data_2/OSTeC/models/topology/tcoords_full.pkl')
        self.tcoords = mio.import_pickle('/data_2/OSTeC/models/topology/tcoords.pkl')
        self.mask = mio.import_pickle('/data_2/OSTeC/models/topology/mask_full2crop.pkl')
        self.tight_mask = mio.import_pickle('/data_2/OSTeC/models/topology/mask_full2tightcrop.pkl')
        self.template = mio.import_pickle('/data_2/OSTeC/models/topology/all_all_all_crop_mean.pkl')
        self.lms_ind = mio.import_pickle('/data_2/OSTeC/models/topology/all_all_all_lands_ids.pkl')
        self.img_shape = [1024, 1024] # 2048

        self.uv_shape = [1024, 1024]   
        uv_mesh = self.tcoords.copy().points[:, ::-1]
        uv_mesh[:, 0] = 1 - uv_mesh[:, 0]
        uv_mesh *= self.uv_shape
        self.uv_mesh = np.concatenate([uv_mesh, uv_mesh[:, 0:1] * 0], 1)
        self.uv_trilist = self.template.trilist




    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

        # for target coeff
        self.target_input_img = input['target_imgs'].to(self.device) 
        self.target_atten_mask = input['target_msks'].to(self.device) if 'msks' in input else None
        self.target_gt_lm = input['target_lms'].to(self.device)  if 'lms' in input else None
        self.target_trans_m = input['target_M'].to(self.device) if 'M' in input else None



    '''
    output_coeff
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
    '''

    def forward(self):
        source_coeff = self.net_recon(self.input_img)
        target_output_coeff = self.net_recon(self.target_input_img)
        self.facemodel.to(self.device)

        output_coeff = source_coeff
        output_coeff[:, 80: 144] = target_output_coeff[:, 80: 144]
        
        self.source_pred_verts, self.source_pred_tex, _, _ = self.facemodel.compute_for_render(source_coeff)

        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)

        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
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
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        


    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)


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



    def uv_map(self, name):
        uv_pos = transfer_UV()
        tex_coords = process_uv(uv_pos.copy())
        tex_coords = torch.tensor(tex_coords, dtype=torch.float32).unsqueeze(0).to(self.device) 


        # face_texture = self.pred_tex / 255.0
        face_texture= self.source_pred_tex

        images = render_img(tex_coords, face_texture, self.bfm, 600, 600.0 - 1.0, 600.0 - 1.0, 0.0, 0.0)
        # images = render_img(tex_coords, face_texture, self.bfm, 3500, 3500.0 -s 1.0, 3500.0 - 1.0, 0.0, 0.0)
        
        images = images.detach().cpu().numpy()
        images = np.squeeze(images)

        # from PIL import Image
        images = np.uint8(images[:, :, :3] * 255.0)
        img = Image.fromarray(images)


        img.save(name)


    def render_uv_image(self):
        uv_tmesh = TexturedTriMesh(self.uv_mesh, self.gt_lm, self.mask, trilist=self.uv_trilist)
        
        bcs = rasterize_barycentric_coordinate_images(uv_tmesh, self.uv_shape)
        img = rasterize_mesh_from_barycentric_coordinate_images(uv_tmesh, *bcs)
        
        img.pixels = np.clip(img.pixels, 0.0, 1.0)

        return img
    
    
    
    # def uv_map_ostec(self, name):
    #     bfm_file = './BFM/BFM.mat'
    #     bfm = face3d.morphable_model.MorphabelModel(bfm_file)
        
    #     self.index_ind = self.bfm.kpt_ind
    #     bfm_uv_file = './BFM/BFM_UV.mat'

    #     uv_coords = face3d.morphable_model.load.load_uv_coords(bfm_uv_file)
    #     self.uv_size = (224,224)
    #     self.mask_stxr =  0.1
    #     self.mask_styr = 0.33
    #     self.mask_etxr = 0.9
    #     self.mask_etyr =  0.7
    #     self.tex_h , self.tex_w, self.tex_c = self.uv_size[1] , self.uv_size[0],3
    #     texcoord = np.zeros_like(uv_coords)
    #     texcoord[:, 0] = uv_coords[:, 0] * (self.tex_h - 1)
    #     texcoord[:, 1] = uv_coords[:, 1] * (self.tex_w - 1)
    #     texcoord[:, 1] = self.tex_w - texcoord[:, 1] - 1
    #     self.texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
    #     self.X_ind = self.bfm.kpt_ind
    #     self.mask_image_names = ['mask_white', 'mask_blue', 'mask_black', 'mask_green']
    #     self.mask_aug_probs = [0.4, 0.4, 0.1, 0.1]
        
        





