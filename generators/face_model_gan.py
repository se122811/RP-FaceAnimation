import functools
import numpy as np
import sys
sys.path.append("/data/face/PIRender")
sys.path.append("/data/face/PIRender/Deep3DFaceRecon_pytorch")
sys.path.append("/data/face/PIRender/models")

from Deep3DFaceRecon_pytorch.models.uv_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from util import flow_util
from generators.base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder, FineUVEncoder, FineUVDecoder


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




class FaceGenerator(nn.Module):
    def __init__(
        self,
        mapping_net, 
        warpping_net, 
        editing_net,
        uv_editing_net, 
        common
        ):
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        #layer: 3, num_res_blocks: 2, base_nc: 64
        self.editing_net = EditingNet(**editing_net, **common)
        self.uv_editing_net = UV_EditingNet(**uv_editing_net, **common)
        
        self.device_index = torch.cuda.current_device()
        self.device = torch.device("cuda:"+str(self.device_index))
        self.transform = transforms.Compose(
            [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        
    
    
    def get_renderer(self, opt, device):
        R= torch.Tensor([[[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        focal = 1015
        img_size = 256
        
        # option 가져올 방법 찾기
        cameras = FoVPerspectiveCameras(device=device, R=R, znear=5., zfar=15.,
                                        fov=2*np.arctan(opt.center/focal)*180./np.pi)
        

        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=img_size,
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
    
    
    def uvmap_to_render(self, uv_map, input_image, pred_vertex, face_index, pred_mask, uv_h =256, uv_w=256, uv_c = 3, show_flag=False, wfp=None):     
        # (53215,2)
        g_uv_coords = load_uv_coords('/data/face/PIRender/BFM/BFM_UV.mat')
        # (35709,1) 
        indices = load_idx("/data/face/PIRender/BFM/BFM_front_idx.mat") 
        # (35709,)
        indices = np.squeeze(indices,1)
        g_uv_coords = g_uv_coords[indices, :]
        

        g_uv_coords_tensor = torch.Tensor(g_uv_coords).to(self.device)
        res_tensor = uv_map
        # res_tensor = torch.Tensor(res_tensor).to(self.device)
        
        B,N,C = pred_vertex.shape
        # res_tensor = res_tensor.unsqueeze(0)
        # res_tensor = res_tensor.repeat(B,1,1,1)
        res_tensor = res_tensor.permute(0,2,3,1)
        
        #UV_Map으로 rendering 하기. 
        texture_img = TexturesUV(res_tensor, face_index, g_uv_coords_tensor.repeat(B,1,1))
        rerender_mesh = Meshes(pred_vertex, face_index, texture_img)
        rerender_img = self.renderer(rerender_mesh)
        
        # rerender_img = torch.clamp(rerender_img,0,255)
        
        self.rerender_face = rerender_img.permute(0,3,1,2).contiguous()[:,:3,:,:]
        input_image = (input_image+1)/2*255
        self.rerender_face = (self.rerender_face+1)/2*255
 
        
        rerender_output_vis = self.rerender_face * pred_mask + (1 - pred_mask) * input_image
        # # rerender_output_vis_numpy = 255. * rerender_output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        # rerender_output_vis = 255. * rerender_output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        # return rerender_output_vis
        
        return rerender_output_vis
 
 
    def forward(
        self, 
        opt,
        input_image,
        input_uv_map,
        pred_vertex,
        tri,
        pred_mask,
        driving_source, 
        stage=None
        ):
        
        self.renderer = self.get_renderer(opt, self.device)
        
        if stage == 'warp':
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            
        else:
            ## uv_map 
            #print("input image: ", type(input_image), "vox_3dmm_img: ",  type(vox_3dmm_img), "driving_source: ", type(driving_source), "stage: ", stage)            
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            refine_uv_map = self.uv_editing_net(input_image, input_uv_map, descriptor)
            self.pred_vertex = pred_vertex
            self.face_index = tri
            self.pred_mask = pred_mask
            
            
            # rerender_output_vis = self.uvmap_to_render(input_uv_map, input_image, self.pred_vertex, self.face_index, pred_mask)
            rerender_output_vis = self.uvmap_to_render(refine_uv_map, input_image, self.pred_vertex, self.face_index, pred_mask)
            rerender_output_vis = (rerender_output_vis / 255.)*2-1
            # rerender_output = self.transform(rerender_output_vis)
            editing_input = torch.cat((input_image, rerender_output_vis), 0)
            
            
            # output['fake_image'] = self.editing_net(editing_input, editing_input, descriptor)
            output['fake_image'] = self.editing_net(editing_input,  output['warp_image'], descriptor)
            # output['fake_image'] = self.editing_net(input_image, input_image, descriptor)          
            output['uv_render_img'] = rerender_output_vis
    
            
        return output



class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        return out   


class WarpingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        base_nc, 
        max_nc, 
        encoder_layer, 
        decoder_layer, 
        use_spect
        ):
        super( WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'nonlinearity':nonlinearity, 'use_spect':use_spect}

        self.descriptor_nc = descriptor_nc 
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                       max_nc, encoder_layer, decoder_layer, **kwargs)

        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), 
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output={}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)

        deformation = flow_util.convert_flow_to_deformation(final_output['flow_field'])
        final_output['warp_image'] = flow_util.warp_image(input_image, deformation)
        
        return final_output


class EditingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        layer, 
        base_nc, 
        max_nc, 
        num_res_blocks, 
        use_spect):  
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)
        self.temp = nn.Linear(20,10)


    def forward(self, input_image, warp_image, descriptor):
        
        # input_image_gt = input_image[0].repeat(10,1,1,1)
        # input_image_rendering = input_image[1].repeat(10,1,1,1)
        # input_imag_f = torch.cat([input_image_gt, input_image_rendering],0)
        # warp_image_f = warp_image[0].repeat(10,1,1,1)
        # input_image_trans = input_imag_f.permute(1,2,3,0)
        
        input_image_trans = input_image.permute(1,2,3,0)
        input_image_fc = self.temp(input_image_trans)
        input_image_reverse = input_image_fc.permute(3,0,1,2)
        x = torch.cat([input_image_reverse, warp_image], 1)
        
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image
    
    
    
    
class UV_EditingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        layer, 
        base_nc, 
        max_nc, 
        num_res_blocks, 
        use_spect):  
        super(UV_EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)


    def forward(self, input_image, uv_map_image, descriptor):
        
        x = torch.cat([input_image, uv_map_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        
        return gen_image
    
    
    
    

