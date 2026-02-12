import os
import time
import datetime
import argparse
import random

import numpy as np
import logging

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import _C as cfg

from loaders.make_dataset import get_Dataset
from models.make_network import get_network, set_pretrained_setting
from losses.make_losses import get_loss_fucntion
from utils.metric import Metric

from torch.nn import functional as F

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
    
    # VFT        35000
    # oulu        4950
    # Axonlab     2101
    # SiW         1700
    # CASIA        600
    # MSU          280

    parser = argparse.ArgumentParser(description="Entry Fuction")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="choose model")
    parser.add_argument("--backbone", type=str, default="RN50", help="choose subname of model")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    parser.add_argument("--resume", type=bool, default=False, help='resume')
    parser.add_argument("--periodically", type=bool, default=False, help='save periodically')
    parser.add_argument("--checkpoint", type=str, default='best_model.pt', help='for resume')
    parser.add_argument("--setting", type=str, default='fas', help='DATASET SETTING [MCIO, SFW, FAS, ALL]')
    parser.add_argument("--train_dataset", type=str, default='FW', help='TRAIN_DATASET')
    parser.add_argument("--test_dataset", type=str, default='S', help='TEST_DATASET')
    parser.add_argument('--train_csv', type=str, default="train_label.txt")
    parser.add_argument('--val_csv', type=str, default="test_label.txt")
    parser.add_argument('--root_dir', type=str, default="celeba-spoof/CelebA_Spoof_/CelebA_Spoof", help='Root directory of dataset')
    parser.add_argument('--full_dataset_csv', type=str, default="full.csv")
    parser.add_argument("--key_train", type=str, default='VFT,oulu,Axonlab,SiW', help='DATASET TRAIN [VFT, oulu, Axonlab, SiW]')
    parser.add_argument("--key_val", type=str, default='CASIA,MSU', help='DATASET VAL [CASIA, MSU]')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--save_path', type=str, default="runs/save_model")
    parser.add_argument('--num_epochs', type=int, default="number of epochs")

    args = parser.parse_args()
    
    # --- Device ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print("device : ", device)

    now_time = datetime.datetime.now()

    model_name = args.model
    save_name = args.backbone.replace("/", "-")
    batch_size = args.batch_size
    seed = args.seed
    resume = args.resume
    checkpoint = args.checkpoint

    cfg['TRAIN']['EPOCH'] = args.num_epochs
    cfg['DATASET']['SETTING'] = args.setting
    cfg['DATASET']['TRAIN_DATASET'] = args.train_dataset
    cfg['DATASET']['TEST_DATASET'] = args.test_dataset

    set_seed(seed, deterministic=False)

    start_epoch = 0
    max_epoch = cfg.TRAIN.EPOCH
    
    # --- Setup ----
    os.makedirs(args.save_path, exist_ok=True)
    count = len(os.listdir(os.path.join(args.save_path))) + 1
    save_folder = os.path.join(args.save_path, f"train_{count}")
    
    print(f">>>>>>>>>>>>> Save training to : {save_folder} <<<<<<<<<<<<<<<<<<<")
    
    save_folder = os.path.join(save_folder, model_name + '_' + save_name)
    createDirectory(save_folder)
    createDirectory(directory=os.path.join(save_folder, 'weights'))
    cfg.LOG.SAVEDF = save_folder
    reference = './reference'

    logger_name = f'train_{save_name}.log'
    logger = create_logger(os.path.join(save_folder, logger_name))
    
    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(save_folder, "tensorboard-logs"))
    
    logger.info(
        f'##############################################################################################################\n'
        f'Save traning to : {save_folder}'
        f'Experiment history\n'
        f'save_name: {save_name}\n'
        f'year: {now_time.year} month: {now_time.month} day: {now_time.day} hour: {now_time.hour} min: {now_time.minute}\n'
        f'##############################################################################################################')
    logger.info(cfg)

    last_epoch = -1
    validation = True
    best_val_loss = np.inf
    best_HTER = np.inf
    save_periodically = args.periodically
    period = 10
    PIN_MEMORY = True
    logger_interval = 10

    lr = cfg.TRAIN.LR

    Similarity_alpha = cfg.TRAIN.SIMILARITY_ALPHA
    Patch_align_beta = cfg.TRAIN.PATCH_ALIGN_BETA
    # get dataset
    train_Dataset, val_Dataset = get_Dataset(args, cfg, SETTING=cfg.DATASET.SETTING)
    net = get_network(cfg=cfg, args=args, net_name=model_name, device=device, backbone=args.backbone)
    net.to(device)
    
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if resume == True: net, optimizer, last_epoch = set_pretrained_setting(net, optimizer, os.path.join(reference, checkpoint))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
    CE_loss, val_CE_loss = get_loss_fucntion(cfg, loss_name='CrossEntropy', device=device)
    patch_align_CE_loss, val_patch_align_CE_loss = get_loss_fucntion(cfg, loss_name='CrossEntropy', device=device)

    val_batch_time = None
    batch_time = 0 #None

    net.train()
    for epoch in range(start_epoch, max_epoch):
        train_total_loss_history, train_Sim_loss_history = [], []

        batch_iterator = iter(DataLoader(train_Dataset, batch_size, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS,
                                         pin_memory=PIN_MEMORY))
        val_batch_iterator = iter(DataLoader(val_Dataset, batch_size, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS,
                                             pin_memory=PIN_MEMORY))
        batch_iterator_len = batch_iterator.__len__()
        val_batch_iterator_len = val_batch_iterator.__len__()

        for batch_idx, (img, target) in enumerate(batch_iterator):
            start_time = time.time()

            img = img.cuda(device)

            Is_real = target['Is_real'].cuda(device)
            # Domain = target['Domain']#.cuda(device)
            # Attack_type = target['Attack_type']#.cuda(device)

            results = net(img, target)
            output_list = results['similarity']
            patch_alignment_results = results['patch_alignment']
            
            ############################

            optimizer.zero_grad()
            Similarity_loss = CE_loss(output_list, Is_real)
            patch_alignment_loss = patch_align_CE_loss(patch_alignment_results, Is_real)

            loss = (Similarity_loss * Similarity_alpha) + (patch_alignment_loss * Patch_align_beta)
            # backward
            loss.backward()
            optimizer.step()

            train_total_loss_history.append(loss.item())
            train_Sim_loss_history.append(Similarity_loss.item())
            
            total_loss_mean, Similarity_loss_mean = np.asarray(train_total_loss_history).mean(), np.asarray(train_Sim_loss_history).mean()

            end_time = time.time()
            batch_time = end_time - start_time

            eta, this_epoch_eta = get_eta(batch_time=batch_time, batch_index=batch_idx, loader_len=batch_iterator_len,
                                          this_epoch=epoch, max_epoch=max_epoch)

            if val_batch_time is None:
                val_eta, _ = get_eta(batch_time=batch_time, batch_index=-1, loader_len=val_batch_iterator_len,
                                     this_epoch=epoch, max_epoch=max_epoch)
            elif val_batch_time is not None:
                val_eta, _ = get_eta(batch_time=val_batch_time, batch_index=-1, loader_len=val_batch_iterator_len,
                                     this_epoch=epoch, max_epoch=max_epoch)
            eta = eta + val_eta

            line = '[Train] Epoch: {}/{} || Iter: {}/{} || ' \
                   'Iter: total_loss: {:.4f} ||' \
                   'Epoch: total_loss: {:.4f} Sim_loss: {:.4f} || ' \
                   'LR: {:.8f} || Batchtime: {:.4f} s || this_epoch: {}||ETA: {}'.format(
                epoch + 1, max_epoch, batch_idx + 1, batch_iterator_len,
                loss.item(), total_loss_mean, Similarity_loss_mean * Similarity_alpha, lr, batch_time, 
                str(datetime.timedelta(seconds=this_epoch_eta)), str(datetime.timedelta(seconds=eta)))
            
            if batch_idx % logger_interval == 0:
                writer.add_scalar("train/total_loss", total_loss_mean, epoch + 1)
                writer.add_scalar("train/Sim_loss", Similarity_loss_mean * Similarity_alpha, epoch + 1)
                writer.add_scalar("train/LR", lr, epoch + 1)
                logger.info(line)


        if validation == True:
            net.eval()
            val_total_loss_history, val_Sim_loss_history = [], []
            with torch.no_grad():
                val_metric = Metric()
                for val_batch_idx, (val_img, val_target) in enumerate(val_batch_iterator):
                    val_start_time = time.time()

                    val_img = val_img.cuda(device)
                    val_Is_real = val_target['Is_real'].cuda(device)
                    # val_Domain = val_target['Domain']#.cuda(device)
                    # val_Attack_type = val_target['Attack_type']#.cuda(device)

                    # forward
                    val_results = net(val_img)
                    val_output_list = val_results['similarity']

                    val_Similarity_loss = val_CE_loss(val_output_list, val_Is_real).cpu().numpy()

                    val_Sim_loss_history.append(val_Similarity_loss)
                    val_Similarity_loss_mean = np.asarray(val_Sim_loss_history).mean()

                    # metric
                    # spoofing | real
                    #     0       1
                    prob = F.softmax(val_output_list, dim=-1).cpu().data.numpy()[:, -1].tolist()
                    val_acc, val_EER, val_HTER, val_auc, val_threshold, val_ACC_threshold, val_TPR_FPR_rate = val_metric(val_Is_real.cpu().numpy().tolist(),prob)

                    val_end_time = time.time()
                    val_batch_time = val_end_time - val_start_time

                    val_eta, val_this_epoch_eta = get_eta(batch_time=val_batch_time, batch_index=val_batch_idx,
                                                          loader_len=val_batch_iterator_len, this_epoch=epoch,
                                                          max_epoch=max_epoch)
                    eta, _ = get_eta(batch_time=batch_time, batch_index=-1, loader_len=batch_iterator_len,
                                     this_epoch=epoch, max_epoch=max_epoch)
                    val_eta = val_eta + eta

                    val_line = '[VAL] Epoch: {}/{} || iter: {}/{} || ' \
                               'Iter: total_loss: {:.4f} || ' \
                               'Epoch: Sim_loss: {:.4f} HTER: {:.4f} AUC: {:.4f} TPR@FPR: {:.4f} top-1: {:.4f} || ' \
                               'Batchtime: {:.4f} s || this_epoch: {} || ETA: {}'.format(
                        epoch + 1, max_epoch, val_batch_idx + 1, val_batch_iterator_len,
                        val_Similarity_loss,
                        val_Similarity_loss_mean * Similarity_alpha, val_HTER * 100, val_auc*100, val_TPR_FPR_rate, val_acc,
                        val_batch_time, str(datetime.timedelta(seconds=val_this_epoch_eta)),
                        str(datetime.timedelta(seconds=val_eta)))
                    
                    # ------ logs ------ 
                    writer.add_scalar("val/total_loss", val_Similarity_loss, epoch + 1)
                    writer.add_scalar("val/Sim_loss", val_Similarity_loss_mean * Similarity_alpha, epoch + 1)
                    writer.add_scalar("val/HTER", val_HTER * 100, epoch + 1)
                    writer.add_scalar("val/AUC", val_auc * 100, epoch + 1)
                    writer.add_scalar("val/TPR@FPR", val_TPR_FPR_rate, epoch + 1)
                    writer.add_scalar("val/Acc", val_acc, epoch + 1)
                    logger.info(val_line)

                # for best NME
                if (val_HTER) < best_HTER:
                    print('\n')
                    new_update = f'Congratulation Best HTER is updated, best_HTER: {best_HTER*100} upto val_HTER: {val_HTER*100}'
                    logger.info(new_update)
                    best_HTER = val_HTER
                    logger.info('=> saving checkpoint to {}'.format(os.path.join(save_folder, model_name + '_' + save_name + '_best.pt')))

                    # save_threshold = 0.05
                    # if cfg.DATASET.SETTING == 'SFW':
                    #     save_threshold = 0.15
                    # elif cfg.DATASET.SETTING == 'MCIO':
                    #     save_threshold = 0.10
                    #
                    # if best_HTER <= save_threshold:
                    best_ckpt_path = os.path.join(save_folder, 'weights', model_name + '_' + save_name + '_best_ckpt.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'performance': best_HTER * 100,
                        'optimizer': optimizer.state_dict(),
                    }, best_ckpt_path)
                    
                    
                    print ("Save model best checkpoint to: ", best_ckpt_path)

            net.train()


            if ((epoch + 1) % period == 0 and epoch > 0) and save_periodically == True:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'performance': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(save_folder, 'weights', model_name + '_' + save_name + '_epoch_' + str(epoch + 1) + '.pt'))
                print ("Save periodically model checkpoint to: ", os.path.join(save_folder, model_name + '_' + save_name + '_epoch_' + str(epoch + 1) + '.pt'))
            
        
        last_ckp_path = os.path.join(save_folder, 'weights', model_name + '_' + save_name + 'last_ckpt.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'performance': best_val_loss,
            'lr': lr
        }, last_ckp_path)
        print(f"💾 Last checkpoint saved {last_ckp_path}")

        scheduler.step()
        lr = scheduler.state_dict()['_last_lr'][0]

    logging.shutdown()








