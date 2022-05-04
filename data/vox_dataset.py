import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as trans_fn
import cv2

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class VoxDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
    
        
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        self.train_mode = "test" if is_inference else "train"

        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.person_ids = self.person_ids * 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        self.vox_3dmm_vid_path = "/data/PIRender_hs/dataset/vox_3dmm_vi"
        
    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids            

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    

    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        return video_item


    def video_capture(self, filename):
        frames = []
        cap = cv2.VideoCapture(filename)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                frames.append(img_pil)
            else:
                break

        # cap.release --> 오픈한 frame 닫기 
        cap.release()
        return frames



    def __len__(self):
        return len(self.person_ids)


    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        frame_source, frame_target = self.random_select_frames(video_item)

        
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], frame_source)
            img_bytes_1 = txn.get(key) 
            key = format_for_lmdb(video_item['video_name'], frame_target)
            img_bytes_2 = txn.get(key) 
            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))
        
        data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source, frame_target)
        data['target_semantics'] = self.transform_semantic(semantics_numpy, frame_target, frame_source)
        

        img1 = Image.open(BytesIO(img_bytes_1))
        data['source_image'] = self.transform(img1)

        img2 = Image.open(BytesIO(img_bytes_2))
        data['target_image'] = self.transform(img2)
        
        
        data['source_image_3dmm'] = torch.tensor(np.array(img1)/255., dtype=torch.float32).permute(2, 0, 1)
        data['target_image_3dmm'] = torch.tensor(np.array(img2)/255., dtype=torch.float32).permute(2, 0, 1)
        
        #vox 3dmm rendering video
        # vox_3dmm_vid_name = os.path.join(self.vox_3dmm_vid_path, self.train_mode, video_item['video_name'] + '.mp4')
        # vid_3dmm = self.video_capture(vox_3dmm_vid_name)
        
        # selected_vid_3dmm_source = vid_3dmm[frame_source]
        # selected_vid_3dmm_target = vid_3dmm[frame_target]
       
        # img_3dmm = Image.open(selected_vid_3dmm)
        # data['img_3dmm_source'] = self.transform(selected_vid_3dmm_source)
        # data['img_3dmm_target'] = self.transform(selected_vid_3dmm_target)
    
        data['source_coeff'] = semantics_numpy[frame_source,...]
        data['target_coeff'] = semantics_numpy[frame_target,...]
        

        return data
    

    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]


    ##### source image 
    def transform_semantic(self, semantic, frame_index, frame_source_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        
        source_index = self.obtain_seq_index(frame_source_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        source_coeff_3dmm = semantic[source_index,...]
        
        id_coeff = source_coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        tex_coeff = source_coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        gamma = source_coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([id_coeff, ex_coeff, tex_coeff, angles, gamma, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq