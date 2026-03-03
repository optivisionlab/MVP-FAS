import torchvision.transforms as transforms
from loaders.MCIO import MCIO_Dataset
from loaders.SFW import SFW_Dataset
from loaders.custom_dataset import FAS_Dataset
import pandas as pd
from sklearn.utils import shuffle
import os
from torchvision import transforms as T
from PIL import Image
import numpy as np


# may use this for protocol 2
class RemoveBlackBorders(object):

  def __call__(self, im):
        # Nếu là numpy → chuyển sang PIL
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)

        if isinstance(im, list):
            return [self.__call__(img) for img in im]

        V = np.array(im)
        if len(V.shape) == 3:
            V = np.mean(V, axis=2)

        X = np.sum(V, axis=0)
        Y = np.sum(V, axis=1)

        xs = np.nonzero(X)[0]
        ys = np.nonzero(Y)[0]

        if len(xs) == 0 or len(ys) == 0:
            return im

        x1, x2 = xs[0], xs[-1]
        y1, y2 = ys[0], ys[-1]

        return im.crop((x1, y1, x2, y2))

  def __repr__(self):
    return self.__class__.__name__


def get_MCIO_dataset(cfg,train='CIO',test='M',img_size= (224, 224), normalize=None,):

    train_dataset = MCIO_Dataset(cfg=cfg,datasets=train,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]),is_train=True)
    val_dataset = MCIO_Dataset(cfg=cfg, datasets=test,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), is_train=False)

    return train_dataset, val_dataset


def get_SFW_dataset(cfg,train='SF',test='W',img_size= (224, 224), normalize=None,):

    train_dataset = SFW_Dataset(cfg=cfg,datasets=train,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]),is_train=True)
    val_dataset = SFW_Dataset(cfg=cfg, datasets=test,
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), is_train=False)

    return train_dataset, val_dataset


def get_FAS_dataset(args, cfg, normalize=None, img_size=(224, 224)):

    train_df = shuffle(pd.read_csv(args.train_csv, usecols=['path', 'is_spoof']), random_state=args.seed)
    val_df = shuffle(pd.read_csv(args.val_csv, usecols=['path', 'is_spoof']), random_state=args.seed)

    train_dataset = FAS_Dataset(cfg=cfg, dataframe=train_df, base_dir=args.root_dir, 
                                transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), 
                                is_train=True)
    
    val_dataset = FAS_Dataset(cfg=cfg, dataframe=val_df, base_dir=args.root_dir, 
                              transform=transforms.Compose([transforms.ToTensor(), normalize, transforms.Resize(img_size)]), 
                              is_train=False)

    return train_dataset, val_dataset


def get_ALL_dataset(args, cfg, normalize=None, img_size=(224, 224)):
    full_df = pd.read_csv(args.full_dataset_csv)
    train_object = args.key_train.split(",")
    val_object = args.key_val.split(",")
    
    print("train_object : ", train_object)
    print("val_object: ", val_object)
    
    train_full_df = shuffle(full_df[full_df['object'].isin(train_object)], random_state=args.seed)
    val_full_df = shuffle(full_df[full_df['object'].isin(val_object)], random_state=args.seed)
    
    train_full_df.to_csv(os.path.join(cfg.LOG.SAVEDF, "train.csv"), index=False)
    val_full_df.to_csv(os.path.join(cfg.LOG.SAVEDF, "val.csv"), index=False)
    
    train_dataset = FAS_Dataset(cfg=cfg, dataframe=train_full_df[['path', 'is_spoof']], base_dir=args.root_dir, 
                                transform=transforms.Compose([
                                    RemoveBlackBorders(), 
                                    transforms.Resize(img_size), 
                                    transforms.ToTensor(), 
                                    normalize
                                ]), 
                                is_train=True)
    
    val_dataset = FAS_Dataset(cfg=cfg, dataframe=val_full_df[['path', 'is_spoof']], base_dir=args.root_dir, 
                              transform=transforms.Compose([
                                  RemoveBlackBorders(), 
                                  transforms.Resize(img_size), 
                                  transforms.ToTensor(), 
                                  normalize
                                ]), 
                              is_train=False)
    
    return train_dataset, val_dataset


def get_Dataset(args, cfg, SETTING="MCIO"):
    normalize = transforms.Normalize(mean=cfg.DATASET.Mean, std=cfg.DATASET.Std)
    
    if SETTING.upper() == "MCIO":
        train_dataset, val_dataset = get_MCIO_dataset(cfg,train=cfg.DATASET.TRAIN_DATASET,test=cfg.DATASET.TEST_DATASET,
                                                      img_size= (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE), normalize=normalize)
    elif SETTING.upper() == 'SFW':
        train_dataset, val_dataset = get_SFW_dataset(cfg, train=cfg.DATASET.TRAIN_DATASET,
                                                      test=cfg.DATASET.TEST_DATASET,
                                                      img_size=(cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE),
                                                      normalize=normalize)
    elif SETTING.upper() == "FAS":
        train_dataset, val_dataset = get_FAS_dataset(args=args, cfg=cfg, normalize=normalize, img_size=(cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))

    elif SETTING.upper() == 'ALL':
        train_dataset, val_dataset = get_ALL_dataset(args=args, cfg=cfg, normalize=normalize, img_size=(cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))

    return train_dataset, val_dataset

