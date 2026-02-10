import os
import time
import datetime
import argparse
import random
import cv2
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader

from configs.cfg import _C as cfg

from loaders.make_dataset import get_Dataset
from models.make_network import get_network
from losses.make_losses import get_loss_fucntion
from utils.metric import Metric

from torch.nn import functional as F

def load_ckpt(net, weight_path):
    checkpoint_dict = torch.load(weight_path)
    checkpoint = checkpoint_dict['state_dict']
    net.load_state_dict(checkpoint)
    return net

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(filename='train_log.log'):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def get_eta(batch_time, batch_index, loader_len, this_epoch, max_epoch):
    # this epoch start 0
    this_epoch_eta = int(batch_time * (loader_len - (batch_index + 1)))
    left_epoch_eta = int(((max_epoch - (this_epoch + 1)) * batch_time * loader_len))
    eta = this_epoch_eta + left_epoch_eta
    return eta, this_epoch_eta


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
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    parser.add_argument("--resume", type=bool, default=True, help='resume')
    parser.add_argument("--checkpoint", type=str, default='best_model.pth', help='for resume')
    parser.add_argument("--setting", type=str, default='MCIO', help='DATASET SETTING [MCIO, SFW]')
    parser.add_argument("--train_dataset", type=str, default='MIO', help='TRAIN_DATASET')
    parser.add_argument("--test_dataset", type=str, default='C', help='TEST_DATASET')
    parser.add_argument('--train_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/train_label.txt")
    parser.add_argument('--val_csv', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof/metas/intra_test/test_label.txt")
    parser.add_argument('--root_dir', type=str, default="/data02/manhquang/dataset/celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--save_path', type=str, default="runs/save_model")

    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    now_time = datetime.datetime.now()

    model_name = args.model
    save_name = args.backbone.replace("/", "-")
    batch_size = args.batch_size
    seed = args.seed
    resume = args.resume
    checkpoint = args.checkpoint

    cfg['DATASET']['SETTING'] = args.setting
    cfg['DATASET']['TRAIN_DATASET'] = args.train_dataset
    cfg['DATASET']['TEST_DATASET'] = args.test_dataset

    set_seed(seed, deterministic=False)

    start_epoch = 0
    max_epoch = cfg.TRAIN.EPOCH
    
    # --- Setup ----
    os.makedirs(args.save_path, exist_ok=True)
    count = len(os.listdir(os.path.join(args.save_path))) + 1
    save_folder = os.path.join(args.save_path, f"test_{count}")
    print(f">>>>>>>>>>>>> Save testing to : {save_folder} <<<<<<<<<<<<<<<<<<<")
    save_folder = os.path.join(save_folder, model_name + '_' + save_name)
    createDirectory(save_folder)

    logger_name = f'test_{save_name}.log'
    logger = create_logger(os.path.join(save_folder, logger_name))
    logger.info(
        f'##############################################################################################################\n'
        f'Save testing to : {save_folder}'
        f'Experiment history\n'
        f'save_name: {save_name}\n'
        f'year: {now_time.year} month: {now_time.month} day: {now_time.day} hour: {now_time.hour} min: {now_time.minute}\n'
        f'##############################################################################################################')
    logger.info(cfg)

    best_val_loss = np.inf
    best_HTER = np.inf
    save_periodically = False
    period = 10
    PIN_MEMORY = True
    logger_interval = 10

    lr = cfg.TRAIN.LR

    Similarity_alpha = cfg.TRAIN.SIMILARITY_ALPHA
    # get dataset
    _, val_Dataset = get_Dataset(args, cfg, SETTING=cfg.DATASET.SETTING)
    net = get_network(cfg, net_name=model_name, device=device)
    print("============= Loading checkpoint =============")
    net = load_ckpt(net=net, weight_path=checkpoint)
    net.to(device)
    print("============= Load checkpoint done =============")
    
    CE_loss, val_CE_loss = get_loss_fucntion(cfg, loss_name='CrossEntropy', device=device)

    val_batch_time = None
    batch_time = 0 # None

    val_batch_iterator = iter(DataLoader(val_Dataset, batch_size, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=PIN_MEMORY))
    val_batch_iterator_len = val_batch_iterator.__len__()

    net.eval()
    val_total_loss_history, val_Sim_loss_history = [], []
    with torch.no_grad():
        val_metric = Metric()
        for val_batch_idx, (val_img, val_target) in enumerate(val_batch_iterator):
            val_start_time = time.time()

            val_img = val_img.cuda(device)
            val_Is_real = val_target['Is_real'].cuda(device)
            # val_Domain = val_target['Domain']#.cuda()
            # val_Attack_type = val_target['Attack_type']#.cuda()


            # forward
            # val_output_list, val_feature = net(val_img)
            val_results = net(val_img)
            val_output_list = val_results['similarity']


            val_Similarity_loss = val_CE_loss(val_output_list, val_Is_real).cpu().numpy()

            val_Sim_loss_history.append(val_Similarity_loss)
            val_Similarity_loss_mean = np.asarray(val_Sim_loss_history).mean()


            # metric
            # spoofing | real
            #     0       1
            prob = F.softmax(val_output_list, dim=-1).detach().cpu().numpy()[:, -1].tolist()
            val_acc, val_EER, val_HTER, val_auc, val_threshold, val_ACC_threshold, val_TPR_FPR_rate = val_metric(val_Is_real.cpu().numpy().tolist(),prob)
            if (val_batch_idx + 1) == 410:
                print()

            val_end_time = time.time()
            val_batch_time = val_end_time - val_start_time

            val_eta, val_this_epoch_eta = get_eta(batch_time=val_batch_time, batch_index=val_batch_idx,
                                                    loader_len=val_batch_iterator_len, this_epoch=0,
                                                    max_epoch=1)
            val_eta = val_eta

            val_line = '[Test][{}/{}] || iter: {}/{} || ' \
                        'This_iter: total_loss: {:.4f} || ' \
                        'This_epoch: Sim_loss: {:.4f} HTER: {:.4f} AUC: {:.4f} TPR@FPR: {:.4f} top-1: {:.4f} || ' \
                        'Batchtime: {:.4f} s || this_epoch: {} || ETA: {}'.format(
                1, max_epoch, val_batch_idx + 1, val_batch_iterator_len,
                val_Similarity_loss,
                val_Similarity_loss_mean * Similarity_alpha, val_HTER * 100, val_auc*100, val_TPR_FPR_rate, val_acc,
                val_batch_time, str(datetime.timedelta(seconds=val_this_epoch_eta)),
                str(datetime.timedelta(seconds=val_eta)))
            logger.info(val_line)


    logging.shutdown()








