import os
from re import L
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import glob

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



class CelebA(Dataset):
    def __init__(self, opt, is_inference):
        self.path = opt.path
        self.root = "/data/face/CelebA/align"
        # self.root = "/data/face/uvll"
        self.coeff_root = "/data/face/inference_CelebA/test"
        self.rendering_file = "/data/face/inference_CelebA/img_result/test"
        file_list = os.listdir("/data/face/inference_CelebA/img_result/test")
        # file_list = os.listdir("/data/face/uvll/co")
        
        self.mat_list = []
        self.img_list = []


        for file in file_list:
            if file.endswith(".png"):    
                mat_file = file[:-8] + '.mat'
                self.mat_list.append(mat_file)
              
            
        for img in self.mat_list :
            image_file = img[:-4] + '.jpg'
            self.img_list.append(image_file)

        # for file in file_list:
        #     if file.endswith(".mat"):
        #         self.mat_list.append(file)

        
        # for img in self.mat_list :
        #     image_file = img[:-4] + '.jpg'
        #     self.img_list.append(image_file)


        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius

        self.img_items, self.person_ids = self.get_img_index(self.img_list)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])


    def get_img_index(self, videos):
        video_items = []

        for video in videos:
            video_items.append(self.Image_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))


        return video_items, person_ids            

        


    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)

        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    

    def Image_Item(self, image_name):
        video_item = {}

        video_item['video_name'] = image_name
        video_item['person_id'] = image_name

        # with self.env.begin(write=False) as txn:
        #     key = format_for_lmdb(video_item['video_name'], 'length')
        #     length = int(txn.get(key).decode('utf-8'))
        
        video_item['num_frame'] = 1
        
        return video_item


    def __len__(self):
        return len(self.img_items)
    

    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        image_item = self.img_items[index]
        mat_item = self.mat_list[index]

        # coeff_file = io.loadmat(os.path.join(self.root,'celebA_coeff', mat_item))
        coeff_file = io.loadmat(os.path.join(self.root,'celebA_coeff', self.mat_list[self.video_index]))
        # img_file = os.path.join(self.root,"img_align_celeba" ,image_item)
        img_file = os.path.join(self.root,"img_align_celeba" ,self.img_items[self.video_index]['video_name']) 
        # img_file = os.path.join(self.root, self.img_items[self.video_index]['video_name']) 
        rendring_file = os.path.join(self.rendering_file, self.img_items[self.video_index]['video_name'])


        ####### source == target #######
        

        img1 = Image.open(img_file)
        img1 = img1.resize((256,256))
        data['source_image'] = self.transform(img1)

        img2 = Image.open(img_file)
        data['target_image'] = data['source_image']
    
    
        #input rendering image    
        img_3dmm = Image.open(rendring_file)
        data['img_3dmm_source'] = self.transform(img_3dmm)
        data['img_3dmm_target'] = self.transform(img_3dmm)
    
        
        data['target_semantics'] = []
        video_item = self.img_items[self.video_index]
        tmp = self.transform_semantic_celeba(coeff_file,0)
        tmp = tmp.repeat(1,27)
        data['target_semantics']= tmp
        data['source_semantics'] = data['target_semantics']
        data['video_name'] = video_item['video_name']
        
        
        return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]
    


    def transform_semantic_celeba(self, semantic, frame_index):
        # index = self.obtain_seq_index(frame_index, semantic.shape[0])
        # coeff_3dmm = semantic[index,...]

        # id_coeff = semantic['id']
        # ex_coeff = semantic['exp']
        # tex_coeff = semantic['tex']
        # angles = semantic['angle']
        # gamma = semantic['gamma']
        # translation = semantic['trans']
        # # crop = np.array([[0.00, 0.00, 0.00]])
        
        
        # crop = np.array([[1.0, 128.00, 128.00]])
        # coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        coeff = semantic['coeff']
        coeff_transform = semantic['transform_params'][0:,:3]
        
        
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff[0:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff[0:, 224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff[0:, 254:257] #translation     
        crop = coeff_transform[:,[2,0,1]] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        
        
        
        
        
        return torch.Tensor(coeff_3dmm).permute(1,0)


    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq



        


