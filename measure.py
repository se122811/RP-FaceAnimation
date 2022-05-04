import torch
import numpy as np
import os

import os
import sys
import math
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import glob




class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    # @staticmethod
    def __call__(self,img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    
# class CelebA(Dataset):
#     def __init__(self):
#         self.pred_root = '/data/face/CelebA/align'
#         self.gt_root = '/data/face/PIRender/vox_result/celeba_image2'
#         self.img_lst = os.listdir(self.gt_root)
#         pass
#     def __len__(self):
#         return len()  
    
#     def __getitem__(self,idx):
#         pred_img_root = os.path.join(,self.img_lst[idx])
#         gt_img_root = 
        


if __name__ =='__main__':
    print('현재위치:',os.getcwd()) # /data/face/PIRender
    
    gt_root = '/data/face/CelebA/align/img_align_celeba/'
    # pred_root = '/data/face/PIRender/vox_result/celeba_image/'
    pred_root = '/data/face/PIRender/vox_result/celeba_image2/'
    
    gt_img = cv2.imread(os.path.join(gt_root,'000018.jpg')) # (218,178,3)
    gt_img_resize = cv2.resize(gt_img,dsize=(256,256))
    pred_img = cv2.imread(os.path.join(pred_root,'000018.jpg')) # (256,256,3)
    
    gt_img_torch = torch.tensor(gt_img_resize,dtype=torch.float32)
    pred_img_torch = torch.tensor(pred_img,dtype=torch.float32)
    
    print()
    # cv2.imwrite('')
    
    ##### Meausre #####
    ssim = SSIM()
    psnr = PSNR()
    
    ssim_val = ssim(gt_img_resize, pred_img)
    psnr_val = psnr(gt_img_torch, pred_img_torch)
    
    print("measure result")
    print("ssim:",ssim_val)
    print("psnr:",psnr_val)
    
    
    

# # test img check
# if __name__ == '__main__':
    
#     gt_root = '/data/face/CelebA/align/img_align_celeba/'
#     pred_root = '/data/face/PIRender/vox_result/celeba_image2'
      
#     img_idx = '202599.jpg'
    
#     gt_img = cv2.imread(os.path.join(gt_root,img_idx))
#     gt_img_resize = cv2.resize(gt_img,dsize=(256,256))
#     # pred_img = cv2.imread(os.path.join(pred_root,img_idx))
    
#     cv2.imwrite('temp/gt_align'+img_idx,gt_img)
#     cv2.imwrite('temp/gt_align_resize_'+img_idx,gt_img_resize)
#     # cv2.imwrite('pred_'+img_idx,pred_img)
