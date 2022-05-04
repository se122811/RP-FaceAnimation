import os
import cv2 
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import master_only_print as print
from data.vox_video_dataset import VoxVideoDataset
from config import Config

import sys
sys.path.append("/data/PIRender_hs/Deep3DFaceRecon_pytorch")
from Deep3DFaceRecon_pytorch.models import create_model
from Deep3DFaceRecon_pytorch.util.preprocess import align_img
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d



def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)


    args = parser.parse_args()
    return args

def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = results_dir+'.mp4' 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() 
    


def write2imagelist(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = results_dir+'.jpg' 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    
    cv2.imwrite(out_name, image_array[0][:,:,::-1])

    # out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    # for i in range(len(image_array)):
    #     out.write(image_array[i][:,:,::-1])
    # out.release() 
    




def write2image(result_dir,name, img):
    file_name = os.path.join(result_dir ,name + ".jpg")    
    image_numpy = img.squeeze().detach().cpu().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy[:,:,::-1]
    cv2.imwrite(file_name, image_numpy)


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)
    
    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()

    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)
    
    
    
    
    # create deep3drecon model
    device_index = torch.cuda.current_device()
    device = torch.device("cuda:"+str(device_index))
    torch.cuda.set_device(device)
    model = create_model(opt.Deep3DRecon_pytorch)
    model.setup(opt.Deep3DRecon_pytorch)
    model.device = device
    model.parallelize()
    model.eval()
    
    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)                          
    net_G = trainer.net_G_ema.eval()

    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    os.makedirs(output_dir, exist_ok=True)
    opt.data.cross_id = args.cross_id
    opt.data.Deep3d = opt.Deep3DRecon_pytorch
    
    
    i = 0
    dataset = VoxVideoDataset(opt.data, is_inference=True)
    with torch.no_grad():
        for video_index in range(dataset.__len__()):
            data = dataset.load_next_video()
            
            img_input_3dmm = data["img_input_3dmm"]
            source_img = data["source_imgs"]
        
            i+=1
            print(len(data['target_semantics']), " test: ",i)
            input_source = data['source_image'][None].cuda()
            input_image = data['source_image'].repeat(len(data['target_semantics']),1,1,1)
            name = data['video_name']

            uv_map_output_list, uv_map_input_list, source_image, output_images, gt_images, gt_images_temp, warp_images, rendering_images, rendering_images_syn = [],[],[],[],[],[],[],[],[]
            
            
            for frame_index in range(len(data['target_semantics'])):
                
                if frame_index == 0 :
                    data_input_3dmm = {
                        'imgs': img_input_3dmm[frame_index].unsqueeze(0),
                        'source_imgs': source_img,
                        # 'first_target_coeff' : None
                        
                    }
                        

                    model.set_input(data_input_3dmm)
                    model.test()
                    _ = model.get_current_visuals()  # get image results
                    _, pred_vertex, tri, pred_mask, target_coeff = model.compute_visuals()
                    uv_map = model.uv_map_3ddfa()
                    uv_map_input= torch.Tensor(uv_map.copy()).to(device)
                    uv_map_input = uv_map_input.permute(0,3,1,2)

                    #하나씩 넣기
                    target_semantic = data['target_semantics'][frame_index][None].cuda()
                    
                    # #하나씩 넣기
                    
                    # torch.Size([73, 27])
                    # target_semantic = data['target_semantics'].repeat(1,27).unsqueeze(0)
                
                else:
                    data_input_3dmm = {
                        'imgs': img_input_3dmm[frame_index].unsqueeze(0),
                        'source_imgs': source_img,
                        # 'first_target_coeff' :  target_coeff
                    }
                        

                    model.set_input(data_input_3dmm)
                    model.test()
                    _ = model.get_current_visuals()  # get image results
                    _, pred_vertex, tri, pred_mask, target_coeff = model.compute_visuals()
                    uv_map = model.uv_map_3ddfa()
                    uv_map_input= torch.Tensor(uv_map.copy()).to(device)
                    uv_map_input = uv_map_input.permute(0,3,1,2)

                    #하나씩 넣기
                    target_semantic = data['target_semantics'][frame_index][None].cuda()
                    



                output_dict = net_G(opt.Deep3DRecon_pytorch, input_source, uv_map_input, pred_vertex, tri, pred_mask, target_semantic)
                

                output_images.append(
                    output_dict['fake_image'].cpu().clamp_(-1, 1)
                    )
                uv_map_input_list.append(
                    output_dict['origin_uv_map'].cpu().clamp_(-1, 1)
                )
                rendering_images_syn.append(
                    output_dict['uv_render_img_syn'].cpu().clamp_(-1, 1)
                )
                uv_map_output_list.append(
                    output_dict['refine_uv_map'].cpu().clamp_(-1, 1)
                )
                rendering_images.append(
                    output_dict['uv_render_img'].cpu().clamp_(-1, 1)
                )
                warp_images.append(
                    output_dict['warp_image'].cpu().clamp_(-1, 1)
                    )                        
                gt_images.append(
                    data['target_image'][frame_index][None] 
                    )
                
                
                # gt_images_temp.append(
                #     data['target_image'][frame_index:frame_index+10]
                #     )
                # gt_images_tensor = torch.stack(gt_images_temp[0])
                
                # if frame_index == 0:
                #     gt_images = gt_images_tensor
                #     print(frame_index)
                # else:
                #     gt_images = torch.cat([gt_images, gt_images_tensor], 0)
                #     print(frame_index)
                
                # save_path = "/data/face/demo/Ours_concat_warp/face_concat_renderface/same/"
                # os.makedirs(save_path)
                
                write2image("/data/face/demo/face_3D/same/gen_images", name +"_{}".format(frame_index), output_dict['fake_image'])
                write2image("/data/face/demo/face_3D/same/gt_images", name +"_{}".format(frame_index), data['target_image'][frame_index])
                write2image("/data/face/demo/face_3D/same/uv_map_syn", name +"_{}".format(frame_index), output_dict['origin_uv_map'])
                write2image("/data/face/demo/face_3D/same/uv_render_image_syn", name +"_{}".format(frame_index), output_dict['uv_render_img_syn'])
                write2image("/data/face/demo/face_3D/same/uv_map", name +"_{}".format(frame_index),  output_dict['refine_uv_map'])
                write2image("/data/face/demo/face_3D/same/uv_render_image", name +"_{}".format(frame_index), output_dict['uv_render_img'])
                write2image("/data/face/demo/face_3D/same/warp_images", name +"_{}".format(frame_index), output_dict['warp_image'])
                write2image("/data/face/demo/face_3D/same/source_images", name +"_{}".format(frame_index), input_source )
                write2image("/data/face/demo/face_3D/same/face_render", name +"_{}".format(frame_index), output_dict['face_render'])
                
            gen_images = torch.cat(output_images, 0)
            gt_images = torch.cat(gt_images, 0)
            rendering_images_syn=torch.cat(rendering_images_syn,0)
            rendering_images=torch.cat(rendering_images,0)
            warp_images = torch.cat(warp_images, 0)
            uv_map_output_list = torch.cat(uv_map_output_list, 0)
            uv_map_input_list = torch.cat(uv_map_input_list, 0)

            # write2imagelist("{}/{}".format("/data/face/demo", name), input_image[0][None], gt_images[0][None], warp_images[0][None], uv_map_input_list[0][None], rendering_images_syn[0][None], uv_map_output_list[0][None],rendering_images[0][None], gen_images[0][None] )
            write2video("{}/{}".format("/data/face/demo/face_3D/same/video", name), input_image, gt_images, warp_images, uv_map_input_list, rendering_images_syn, uv_map_output_list,rendering_images, gen_images )
            print("write results to video {}/{}".format(output_dir, name))