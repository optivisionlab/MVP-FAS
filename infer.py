import os
import pandas as pd
import argparse
import torch
from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
from utils.metric import get_HTER_at_thr, get_EER_states
import torchvision.transforms as transforms
from torch.nn import functional as F
from loaders.make_dataset import RemoveBlackBorders
from PIL import Image
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


def get_eta(batch_time, batch_index, loader_len, this_epoch, max_epoch):
    # this epoch start 0
    this_epoch_eta = int(batch_time * (loader_len - (batch_index + 1)))
    left_epoch_eta = int(((max_epoch - (this_epoch + 1)) * batch_time * loader_len))
    eta = this_epoch_eta + left_epoch_eta
    return eta, this_epoch_eta


def create_logger(filename='train_log.log'):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger



def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Entry Fuction")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="choose model")
    parser.add_argument("--backbone", type=str, default="RN50", help="choose subname of model")
    parser.add_argument("--weights", type=str, default='best_model.pt', help='for infer')
    parser.add_argument('--test_csv', type=str, default="test_label.csv")
    parser.add_argument('--root_dir', type=str, default="dataset", help='Root directory of dataset')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--save_path', type=str, default="runs/infer")
    parser.add_argument("--seed", type=int, default=42, help="random seed for training")

    args = parser.parse_args()

    model_name = args.model
    save_name = args.backbone.replace("/", "-")
    checkpoint = args.weights
    
    # --- Setup ----
    os.makedirs(args.save_path, exist_ok=True)
    count = len(os.listdir(os.path.join(args.save_path))) + 1
    save_folder = os.path.join(args.save_path, f"infer_{count}")
    logger_name = f'infer_{save_name}.log'
    createDirectory(save_folder)
    logger = create_logger(os.path.join(save_folder, logger_name))
    logger.info(f">>>>>>>>>>>>> Save Infer to : {save_folder} <<<<<<<<<<<<<<<<<<<")
    
    # --- Device ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device : {device}")
    
    # --- setup -----
    net = get_network(cfg=cfg, args=args, net_name=model_name, device=device, backbone=args.backbone)
    net = load_checkpoint(net, weight_path=args.weights)
    net.to(device)
    
    logger.info("load checkpoint is done!")
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
    
    test_df = pd.read_csv(args.test_csv, usecols=['path', 'is_spoof'])
    preds, preds_score = [], []
    targets, targets_score = [], []
    for path, label in test_df.values:
        img = Image.open(os.path.join(args.root_dir, path))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            outputs = net(img)
        sim = outputs['similarity']
        prob = F.softmax(sim, dim=-1).cpu().data.numpy()[:, -1].tolist()
        logger.info(f"path: {path} - {prob} - {'live' if prob[0] > 0.5 else 'spoof'}")
        preds_score.append(prob[0])
        targets_score.append(int(not label))
        preds.append('live' if prob[0] > 0.5 else 'spoof')
        targets.append('spoof'if label else 'live')
    
    cm = confusion_matrix(targets, preds, labels=['spoof', 'live'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['spoof', 'live'])
    fig, ax = plt.subplots(figsize=(8, 6)) # Optional: set figure size
    disp.plot(ax=ax) # Pass the ax object to the plot method
    plt.title("Confusion Matrix Live/Spoof") # Optional: add a title
    disp.figure_.savefig(os.path.join(save_folder, 'confusion_matrix_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(classification_report(targets, preds, target_names=['spoof', 'live']))
    
    EER_score, threshold, _, _ = get_EER_states(np.array(preds_score), np.array(targets_score))
    HTER_score = get_HTER_at_thr(np.array(preds_score), np.array(targets_score), threshold)

    logger.info(f"EER_score: {EER_score}, HTER_score: {HTER_score}, threshold: {threshold}")
    