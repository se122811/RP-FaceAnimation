
"""This script defines the visualizer for Deep3DFaceRecon_pytorch
"""
import numpy as np
import os
import sys
import ntpath
import time
import cv2
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library tensprboardX for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.use_html = opt.isTrain and not opt.no_html
        self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'logs', opt.name))
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    def display_current_results(self, visuals, total_iters, epoch, save_result):
        """Display current results on tensorboad; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        for label, image in visuals.items():
            self.writer.add_image(label, util.tensor2im(image), total_iters, dataformats='HWC')

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()
    # def uv_map_3ddfa(self, name, uv_h =256, uv_w=256, uv_c = 3, show_flag=False, wfp=None):   
    #     ver_lst = []
    #     res_lst = []

    #     textures = self.pred_tex[0].detach().cpu().numpy()

    #     # textures = np.squeeze(textures,0)
        
    #     face_index = self.facemodel.face_buf.detach().cpu().numpy()
    #     face_index = face_index.copy(order='C')
    #     face_index = face_index.astype(np.int32)


    #     a_temp = face_index[:,1].copy()
    #     b_temp = face_index[:,2].copy()
    #     face_index[:,1] = b_temp
    #     face_index[:,2] = a_temp


    #     g_uv_coords = load_uv_coords('BFM/BFM_UV.mat')
    #     indices = load_idx("BFM/BFM_front_idx.mat") #todo: handle bfm_slim
    #     indices = np.squeeze(indices,1)

    #     g_uv_coords = g_uv_coords[indices, :]
    #     uv_coords = process_uv(g_uv_coords.copy(), uv_h=uv_h, uv_w=uv_w)
    #     res = rasterize(uv_coords, face_index, textures, height=uv_h, width=uv_w, channel=uv_c)


    #     res_lst.append(res)
    #     res = res[:,:,::-1]

    #     # if wfp is not None:
    #     cv2.imwrite(name, res)
    #     print(f'Save visualization result to {wfp}')
        # D_loss_collection = {}
        # for name, value in losses.items():
        #     if 'G' in name or 'NCE' in name or 'idt' in name:
        #         G_loss_collection[name] = value
        #     else:
        #         D_loss_collection[name] = value
        # self.writer.add_scalars('G_collec', G_loss_collection, total_iters)
        # self.writer.add_scalars('D_collec', D_loss_collection, total_iters)
        for name, value in losses.items():
            self.writer.add_scalar(name, value, total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


class MyVisualizer:
    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the optio
        self.name = opt.name
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')
        
        if opt.phase != 'test':
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
            # create a logging file to store training losses
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)


    def display_current_results(self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
            add_image=True):
        """Display current results on tensorboad; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        # if (not add_image) and (not save_results): return
        

        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d'%(dataset, i + count),
                            image_numpy, total_iters, dataformats='HWC')

                if save_results:
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, total_iters))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)
    


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

    
    def display_current_results_video(self, visuals, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
            add_image=True):
        """Display current results on tensorboad; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        # if (not add_image) and (not save_results): return
        video_numpy= []

        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                video_numpy.append(image_numpy)
                
                if add_image:
                    self.writer.add_image(label + '%s_%02d'%(dataset, i + count),
                            image_numpy, total_iters, dataformats='HWC')

                if save_results:
                    save_path = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, total_iters))
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)

                    if name is not None:
                        img_path = os.path.join(save_path, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)
            
                    
        output_dir = save_path
        self.write2video("{}/{}".format(output_dir, name), video_numpy)
        print("write results to video {}/{}".format(output_dir, name))
        


    def plot_current_losses(self, total_iters, losses, dataset='train'):
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s'%dataset, value, total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, dataset='train'):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(dataset: %s, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            dataset, epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message