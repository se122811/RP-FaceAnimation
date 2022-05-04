import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
from scipy import io

from data.celeba_dataset import CelebA
from data.vox_dataset import format_for_lmdb


class CelebAVideoDataset(CelebA):
    def __init__(self, opt, is_inference):
        super(CelebAVideoDataset, self).__init__(opt, is_inference)
        self.video_index = -1

    def __len__(self):
        return len(self.person_ids)

    def load_next_video(self):
        data={}
        self.video_index += 1

        video_item = self.img_items[self.video_index]
        
        coeff_file = io.loadmat(os.path.join(self.coeff_root, self.mat_list[self.video_index]))
        # img_file = os.path.join(self.root,"img_align_celeba" ,self.img_items[self.video_index]['video_name'])
        img_file = os.path.join(self.root,"test" ,self.img_items[self.video_index]['video_name'])
        
        rendring_file = os.path.join(self.rendering_file, self.img_items[self.video_index]['video_name'])
        rendring_file = rendring_file + '.png'

        # coeff_file = io.loadmat(os.path.join(self.root,'co', self.mat_list[self.video_index]))
        # img_file = os.path.join(self.root,"img" ,self.img_items[self.video_index]['video_name'])


        img1 = Image.open(img_file)
        img1 = img1.resize((256,256))
        
        #input rendering image    
        img_3dmm = Image.open(rendring_file)
        data['img_3dmm_source'] = self.transform(img_3dmm)
        data['img_3dmm_target'] = self.transform(img_3dmm)
    

        data['source_image'] = self.transform(img1)
        data['target_image'] = []
        data['target_semantics'] = []

        for frame_index in range(video_item['num_frame']):
            data['target_image'].append(data['source_image'])
            tmp = self.transform_semantic_celeba(coeff_file, frame_index)
            tmp = tmp.repeat(1, 27)
            data['target_semantics'].append(tmp)
        
        
        data['video_name'] = video_item['video_name']

        return data  
   
