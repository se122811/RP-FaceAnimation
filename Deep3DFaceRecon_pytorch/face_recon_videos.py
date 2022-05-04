import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat


import torch 


from util.visualizer import MyVisualizer
from models import create_model
from options.inference_options import InferenceOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import mkdirs, tensor2im, save_image


import os



def get_data_path(root, keypoint_root):     
    filenames = list()
    keypoint_filenames = list()

    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS

    for ext in extensions:
        filenames += glob.glob(f'{root}/**/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    keypoint_filenames = sorted(glob.glob(f'{keypoint_root}/**/*.txt', recursive=True))
    assert len(filenames) == len(keypoint_filenames)
    return filenames, keypoint_filenames


def write2video(video_numpy, name):
    # root = '/data/face/Deep3DFaceRecon_pytorch/result_video'
    
    # result_dir = os.path.join(root, name)
    result_dir = "/data/Deep3DFaceRecon_pytorch/result_test/train/test.mp4"
    cat_video=None

    # for video in video_list:
    # video_numpy = video_numpy[:,:3,:,:].cpu().float().detach().numpy()
    # video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    
    video_numpy = video_numpy.astype(np.uint8)
    cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = result_dir 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() 


class VideoPathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        txt_filename = self.txt_filenames[index]

        frames = self.read_video(filename)
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([len(frames), -1, 2]) 
        out_images, out_trans_params = list(), list()
        
        for i in range(len(frames)):
            out_img, _, out_trans_param \
                = self.image_transform(frames[i], lm[i])
            out_images.append(out_img[None])
            out_trans_params.append(out_trans_param[None])

        return {
            'imgs': torch.cat(out_images, 0),
            'trans_param':torch.cat(out_trans_params, 0),
            'filename': filename
        }
        
        
    def read_video(self, filename):
        frames = list()
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def image_transform(self, images, lm):
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params        
 


def main(opt, model):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    filenames, keypoint_filenames = get_data_path(opt.input_dir, opt.keypoint_dir)
    visualizer = MyVisualizer(opt)

    dataset = VideoPathDataset(filenames, keypoint_filenames, opt.bfm_folder)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # can noly set to one here!
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )     

    batch_size = opt.inference_batch_size
    
    for data in tqdm(dataloader):
        num_batch = data['imgs'][0].shape[0] // batch_size

        pred_coeffs = list()
        pred_video = []
        
        if num_batch == 0:
            data_input = {                
                'imgs': data['imgs'][0,  : data['imgs'][0].shape[0]],
                # 'imgs': data['imgs'][0],
                'im_paths' : data['filename']
            }
                
            model.set_input(data_input)  
            model.test()
            
            pred_coeff = {key:model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
            pred_coeff = np.concatenate([
                pred_coeff['id'], 
                pred_coeff['exp'], 
                pred_coeff['tex'], 
                pred_coeff['angle'],
                pred_coeff['gamma'],
                pred_coeff['trans']], 1)

            pred_coeffs.append(pred_coeff) 

            visuals = model.get_current_visuals()  # get image results
            # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
            # save_results=True, count=i, name=img_name, add_image=False)


            vid = model.compute_visuals()
            # pred_video = np.concatenate((pred_video,vid),axis=0)
            # pred_video.cat(vid)
            
            pred_video.append(vid)

            
            full_video = pred_video[0]
            for i in range(1,len(pred_video)):
                full_video = np.concatenate((full_video, pred_video[i]),axis=0)

            pred_coeffs = np.concatenate(pred_coeffs, 0)
            pred_trans_params = data['trans_param'][0].cpu().numpy()
            name = data['filename'][0].split('/')[-2:]
        
            #write2video(full_video, name[1])

            name[-1] = os.path.splitext(name[-1])[0] + '.mat'
            os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
            savemat(
                os.path.join(opt.output_dir, name[-2], name[-1]), 
                {'coeff':pred_coeffs, 'transform_params':pred_trans_params}
            )
              
        else:
            for index in range(num_batch):
                if index != num_batch-1:
                    data_input = {                
                        'imgs': data['imgs'][0,index*batch_size:(index+1)*batch_size],
                        # 'imgs': data['imgs'][0],
                        'im_paths' : data['filename']
                    }
                else:
                    data_input = {                
                        'imgs': data['imgs'][0, index*batch_size : data['imgs'][0].shape[0]],
                        # 'imgs': data['imgs'][0],
                        'im_paths' : data['filename']
                    }
                    
                model.set_input(data_input)  
                model.test()


                
                pred_coeff = {key:model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
                pred_coeff = np.concatenate([
                    pred_coeff['id'], 
                    pred_coeff['exp'], 
                    pred_coeff['tex'], 
                    pred_coeff['angle'],
                    pred_coeff['gamma'],
                    pred_coeff['trans']], 1)

                pred_coeffs.append(pred_coeff) 

                visuals = model.get_current_visuals()  # get image results
                
    
                vid = model.compute_visuals()
                # pred_video = np.concatenate((pred_video,vid),axis=0)
                # pred_video.cat(vid)
                
                pred_video.append(vid)

            
            full_video = pred_video[0]
            for i in range(1,len(pred_video)):
                full_video = np.concatenate((full_video, pred_video[i]),axis=0)

            pred_coeffs = np.concatenate(pred_coeffs, 0)
            pred_trans_params = data['trans_param'][0].cpu().numpy()
            name = data['filename'][0].split('/')[-2:]
        
            #write2video(full_video, name[1])

            name[-1] = os.path.splitext(name[-1])[0] + '.mat'
            os.makedirs( os.path.join(opt.output_dir, name[-2]), exist_ok=True)
            savemat(
                os.path.join(opt.output_dir, name[-2], name[-1]), 
                {'coeff':pred_coeffs, 'transform_params':pred_trans_params}
            )
            

        # name = data['filename'][0].split('/')[-2:]
        # uv_map = model.uv_map_3ddfa()
        # rerender_output_vis = model.uvmap_to_render(uv_map)
        # # model.uv_map(os.path.join(opt.output_dir, name[-2], name[-1][:-4] +'.png'))
        # write2video(rerender_output_vis, os.path.join(opt.output_dir, name[-2], name[-1][:-4] +'.mp4'))

        

if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    model.device = 'cuda:0'
    model.parallelize()
    model.eval()

    main(opt, model)