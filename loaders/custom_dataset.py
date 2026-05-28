import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from torchvision import transforms
from loaders.moire import Moire
from PIL import Image
from torchvision.transforms import functional as F

class RandomLuxTransform:
    def __init__(self, lux_min=200, lux_max=550,
                 factor_min=0.6, factor_max=1.5):
        self.lux_min = lux_min
        self.lux_max = lux_max
        self.factor_min = factor_min
        self.factor_max = factor_max

    def __call__(self, img):
        lux = random.uniform(self.lux_min, self.lux_max)

        alpha = (lux - self.lux_min) / (self.lux_max - self.lux_min)
        factor = self.factor_min + alpha * (self.factor_max - self.factor_min)

        return F.adjust_brightness(img, factor)


class RandomCeilingLightEffect:
    """
    Tạo hiệu ứng điểm sáng ngẫu nhiên kiểu bóng đèn trần nhà
    """

    def __init__(
        self,
        num_lights=20,
        radius_range=(20, 80),
        intensity=0.4,
        blur_sigma=25,
        top_only=False,
    ):
        self.num_lights = num_lights
        self.radius_range = radius_range
        self.intensity = intensity
        self.blur_sigma = blur_sigma
        self.top_only = top_only

    def _generate_light_layer(self, shape):
        h, w = shape[:2]

        light_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for _ in range(self.num_lights):

            # Random vị trí
            x = random.randint(0, w - 1)

            if self.top_only:
                y = random.randint(0, h // 3)
            else:
                y = random.randint(0, h - 1)

            # Random kích thước
            radius = random.randint(
                self.radius_range[0],
                self.radius_range[1]
            )

            # Random màu ánh sáng
            color = (
                random.randint(220, 255),  # R
                random.randint(220, 255),  # G
                random.randint(180, 255),  # B
            )

            # Vẽ light blob
            cv2.circle(
                light_layer,
                (x, y),
                radius,
                color,
                -1
            )

        # Gaussian blur tạo glow
        light_layer = cv2.GaussianBlur(
            light_layer,
            (0, 0),
            sigmaX=self.blur_sigma
        )

        return light_layer

    def apply(self, image):
        """
        Apply effect vào ảnh

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB hoặc BGR

        Returns
        -------
        np.ndarray
        """

        img = image.copy()

        light_layer = self._generate_light_layer(img.shape)

        # Blend ánh sáng
        result = cv2.addWeighted(
            img,
            1.0,
            light_layer,
            self.intensity,
            0
        )

        return result

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
        # Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2HSV)
        Img[..., 1] *= np.random.uniform(0.8, 1.2)
        # Img = cv2.cvtColor(Img, cv2.COLOR_HSV2BGR).astype(np.uint8)
        Img = cv2.cvtColor(Img, cv2.COLOR_HSV2RGB).astype(np.uint8)
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
        # Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

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
            
            if prob_value < self.cfg.TRAIN.MOIRE: # 10%: Moire → spoof
                Img = self.moire(Img)
                # color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                color_jitter = transforms.ColorJitter(brightness=(0.2, 2.5), contrast=(0.4, 1.5), saturation=0.4, hue=0.4)
                # Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(Img)
                transformed_image_pil = color_jitter(image_pil)
                Img = np.array(transformed_image_pil)
                # Img = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)
                is_real = 0 # biến từ live -> spoof
            
            elif prob_value < self.cfg.TRAIN.RandomLux_Real: # 20%: Lux → real
                # live nhưng điều chỉnh ánh sáng
                transforms_lux = transforms.Compose([
                        RandomLuxTransform(200, 550),
                        # biến thiên local
                        transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.1
                        )
                    ])
                # Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                Img_pil = Image.fromarray(Img)
                Img_t = transforms_lux(Img_pil)
                Img = np.array(Img_t)
                # Img = cv2.cvtColor(Img_np, cv2.COLOR_RGB2BGR)
            
            elif prob_value < self.cfg.TRAIN.RandomCeilingLightEffect_Real: # 20%: CeilingLight(real)
                ceiling_light = RandomCeilingLightEffect(
                    num_lights=random.randint(5, 15),
                    radius_range=(20, 80),
                    intensity=random.uniform(0.2, 0.5),
                    blur_sigma=random.randint(15, 35),
                    top_only=random.random() < 0.5
                )
                Img = ceiling_light.apply(Img)
        
        if self.is_train == True and is_spoof == True and self.is_physical == True:
            prob_value = random.random()
            
            if prob_value < self.cfg.TRAIN.RandomLux_Spoof: # 20%: Lux → spoof
                # live nhưng điều chỉnh ánh sáng
                transforms_lux = transforms.Compose([
                        RandomLuxTransform(200, 550),
                        # biến thiên local
                        transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.1
                        )
                    ])
                # Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
                Img_pil = Image.fromarray(Img)
                Img_t = transforms_lux(Img_pil)
                Img = np.array(Img_t)
                # Img = cv2.cvtColor(Img_np, cv2.COLOR_RGB2BGR)
                
            elif prob_value < self.cfg.TRAIN.RandomCeilingLightEffect_Spoof: # 20%: CeilingLight(spoof)
                ceiling_light = RandomCeilingLightEffect(
                    num_lights=random.randint(5, 15),
                    radius_range=(20, 80),
                    intensity=random.uniform(0.2, 0.5),
                    blur_sigma=random.randint(15, 35),
                    top_only=random.random() < 0.5
                )
                Img = ceiling_light.apply(Img)
        
        # numpy to torch tensor and normalize
        Img = Image.fromarray(Img) # nếu không sài cái RemoveBlackBorders
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


