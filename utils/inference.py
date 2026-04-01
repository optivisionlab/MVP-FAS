import torchvision.transforms as transforms
import torch
from loaders.make_dataset import RemoveBlackBorders
from torch.nn import functional as F


def infer_model(net, cfg, device, img):
    # metric
    # spoofing ~ is_real = 0 | real ~ is_real = 1
    #     0       1
    # labels = ['spoof', 'live']
    
    transform = transforms.Compose([
        RemoveBlackBorders(), 
        transforms.Resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=cfg.DATASET.Mean, std=cfg.DATASET.Std)
    ])
    
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        outputs = net(img)
    sim = outputs['similarity']
    # print("sim : ", sim)
    prob = F.softmax(sim, dim=-1).cpu().data.numpy()[:, -1].tolist()
    return prob
