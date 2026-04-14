import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from torchvision import transforms
from loaders.moire import Moire
from PIL import Image

# metric
# spoofing    |   real
# 0 -> Flase      1 -> True

class FAS_Dataset(Dataset):
    def __init__(self, cfg, dataframe, base_dir, transform=None, is_train=True, is_physical=False):
        self.cfg = cfg
        self.is_train = is_train
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.moire = Moire()
        self.Transform = transform
        self.is_physical = is_physical
        self.init_aug()
        
    def init_aug(self):
        self.Saturation = self.cfg.MCIO.Random_Saturation
        self.Flip = self.cfg.MCIO.Random_Horizontal_Flip
        
    def Flip_Saturation(self, Img, is_train=True):
        if is_train == True:
            if self.Flip is True:
                Flip_Flag = np.random.randint(0,2)
                if Flip_Flag ==1:
                    Img = self.Image_Flip(Img)
            if self.Saturation is True:
                Saturation_Flag = np.random.randint(0,2)
                if Saturation_Flag == 1:
                    Img = self.Image_Saturation(Img)
        return Img

    def Image_Flip(self, Img):
        Img = cv2.flip(Img, 1)
        return Img
    
    def Image_Saturation(self,Img):
        Img = Img.astype(np.float32)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        Img[..., 1] *= np.random.uniform(0.8, 1.2)
        Img = cv2.cvtColor(Img, cv2.COLOR_HSV2BGR).astype(np.uint8)
        return Img

    def get_type_name(self, type_dict, value):
        for k, v in type_dict.items():
            if value in v:
                return k

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # db_slic = copy.deepcopy(self.database[idx])

        Img_path = os.path.join(self.base_dir, self.dataframe.iloc[idx, 0])
        is_spoof = self.dataframe.iloc[idx, 1] # True -> spoof, False -> live <----->  1 -> spoof, 0 -> live
        is_real = int(not is_spoof)

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.is_train:
            Img = self.Flip_Saturation(Img)
        
        if self.is_train == True and is_spoof == False and self.is_physical == True:
            prob_value = random.random()
            if prob_value < 0.1:
                Img = self.moire(Img)
                # color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                color_jitter = transforms.ColorJitter(brightness=(0.2, 2.5), contrast=(0.4, 1.5), saturation=0.4, hue=0.4)
                Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(Img_rgb)
                transformed_image_pil = color_jitter(image_pil)
                transformed_image_np = np.array(transformed_image_pil)
                Img = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)
                is_real = 0
        
        # numpy to torch tensor and normalize
        if self.Transform is not None:
            Img = self.Transform(Img)

        meta = {
            'Img_path': Img_path,
            'Is_real': is_real,
            # 'Domain':  domain,
            # 'Attack_type': attack_type,
        }
        return Img, meta

if __name__ == '__main__':
    print()


