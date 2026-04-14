import os
import pandas as pd
import argparse
import torch
from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
from utils.metric import get_HTER_at_thr, get_EER_states, calculate_threshold
import torchvision.transforms as transforms
from torch.nn import functional as F
from loaders.make_dataset import RemoveBlackBorders
from PIL import Image
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import cv2
from ultralytics import YOLO


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
    parser.add_argument("--FAS_SAVE", action='store_true', help='SAVE IMAGE FAS_SAVE model')
    parser.add_argument("--YOLO", action='store_true', help='use YOLO')
    parser.add_argument("--YOLO_SAVE", action='store_true', help='SAVE IMAGE CROP YOLO')
    parser.add_argument("--YOLO_FACE", action='store_true', help='use YOLO_FACE')
    parser.add_argument("--YOLO_FACE_SAVE", action='store_true', help='SAVE IMAGE CROP YOLO_FACE')
    parser.add_argument("--threshold", type=float, default=0.5, help='tune threshold')
    parser.add_argument("--YOLO_DET_MASK", action='store_true', help='user model YOLO_DET_MASK')
    parser.add_argument("--MVP_FAS_FACE_CROP", action='store_true', help='user model MVP_FAS_FACE_CROP')
    parser.add_argument("--weights_mvp_face_crop", type=str, default='weights_mvp_face_crop.pt', help='use weights_mvp_face_crop for infer')
    parser.add_argument("--weights_yolo_det", type=str, default='yolo.pt', help='use weights_yolo_det for infer')
    parser.add_argument("--weights_yolo_det_face", type=str, default='yolo.pt', help='use weights_yolo_det_face for infer')
    parser.add_argument("--weights_yolo_det_mask", type=str, default='yolo.pt', help='use weights_yolo_det_mask for infer')
    
    
    net, net_face_crop, yolo_model, yolo_face_model, yolo_det_mask = None, None, None, None, None
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
    
    # Convert the Namespace object to a dictionary
    args_dict = vars(args)
    # Print the dictionary to see all key-value pairs
    for key, value in args_dict.items():
        logger.info("logs {}: {}".format(key, value))
    
    # --- Device ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device : {device}")
    
    # --- setup -----
    net = get_network(cfg=cfg, args=args, net_name=model_name, device=device, backbone=args.backbone)
    net_face_crop = get_network(cfg=cfg, args=args, net_name=model_name, device=device, backbone=args.backbone)
    
    logger.info(f"load checkpoint: {args.weights}")
    net = load_checkpoint(net, weight_path=args.weights)
    net.to(device)
    logger.info("load checkpoint is done! weight: {}".format(args.weights))
    
    if args.MVP_FAS_FACE_CROP:
        logger.info(f"load checkpoint: {args.weights_mvp_face_crop}")
        net_face_crop = load_checkpoint(net, weight_path=args.weights_mvp_face_crop)
        net_face_crop.to(device)
        logger.info("load checkpoint is done! weight: {}".format(args.weights_mvp_face_crop))
    
    # LOAD YOLO
    if args.YOLO:
        yolo_model = YOLO(args.weights_yolo_det)
    
    if args.YOLO_FACE:
        yolo_face_model = YOLO(args.weights_yolo_det_face)
    
    if args.YOLO_DET_MASK:
        yolo_det_mask = YOLO(args.weights_yolo_det_mask)
    
    test_df = pd.read_csv(args.test_csv, usecols=['path', 'is_spoof'])
    preds, preds_score = [], []
    targets, targets_score = [], []
    logs_csv = []
    createDirectory(os.path.join(save_folder, 'save'))
    for path, label in test_df.values:
        try:
            pred = target = ''
            img = Image.open(os.path.join(args.root_dir, path))
            img = img.convert('RGB')

            prob1 = infer_model(net, cfg, device, img=img)
            logger.info(f"\ninfer step 1: path: {os.path.join(args.root_dir, path)} - {prob1} - {'live' if prob1[0] > args.threshold else 'spoof'}")
            target = 'spoof' if label else 'live'
            pred = 'live' if prob1[0] > args.threshold else 'spoof'
            flag = []
            if prob1[0] > args.threshold:
                if args.YOLO:
                    createDirectory(os.path.join(save_folder, 'crop_image'))
                    results = yolo_model.predict(img, device=device, conf=0.5, classes=[0])
                    for result in results:
                        
                        boxes = result.boxes
                        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                        x_min, y_min, x_max, y_max = map(int, boxes.xyxy[best_idx].cpu().numpy())
                        bbox = [x_min, y_min, x_max, y_max]
                        cropped_img = img.crop(tuple(bbox))
                            
                        prob2 = infer_model(net, cfg, device, img=cropped_img)
                        preds_score.append(prob2[0])
                        pred = 'live' if prob2[0] > args.threshold else 'spoof'
                        
                        if args.YOLO_SAVE and pred != target:
                            cropped_img.save(os.path.join(save_folder, 'crop_image', os.path.basename(path)))
                            logger.info("SAVE CROP IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image', os.path.basename(path))))
                        
                        logger.info(f"infer step 2: path: {os.path.join(args.root_dir, path)} - {prob2} - {'live' if prob2[0] > args.threshold else 'spoof'} \n")

                if args.YOLO_FACE:
                    createDirectory(os.path.join(save_folder, 'crop_image_face'))
                    results_face = yolo_face_model.predict(img, device=device, conf=0.5, classes=[0])
                    
                    for f_result in results_face:
                        
                        f_boxes = f_result.boxes
                        f_best_idx = int(np.argmax(f_boxes.conf.cpu().numpy()))
                        f_x_min, f_y_min, f_x_max, f_y_max = map(int, f_boxes.xyxy[f_best_idx].cpu().numpy())
                        f_bbox = [f_x_min, f_y_min, f_x_max, f_y_max]
                        f_cropped_img = img.crop(tuple(f_bbox))
                        
                        prob3 = infer_model(net_face_crop, cfg, device, img=f_cropped_img)
                        logger.info(f"infer step 3: path: {os.path.join(args.root_dir, path)} - {prob3} - {'live' if prob3[0] > args.threshold else 'spoof'} \n")
                        
                        if prob3[0] > args.threshold:
                            if args.YOLO_DET_MASK:
                                results_mask = yolo_det_mask.predict(img, device=device, conf=0.7)
                                for r_mask in results_mask:
                                    obb = getattr(r_mask, "obb", None)
                                    has_object = obb is not None and getattr(obb, "conf", None) is not None and len(obb.conf) > 0
                                    pred = 'spoof' if has_object else 'live'
                                    preds_score.append(int(not has_object))
                                    logger.info(f"infer step 4: path: {os.path.join(args.root_dir, path)} - {int(not has_object)} - {pred} \n")
                            else:
                                preds_score.append(prob3[0])
                                pred = 'live' if prob3[0] > args.threshold else 'spoof'
                        else:
                            preds_score.append(prob3[0])
                            pred = 'live' if prob3[0] > args.threshold else 'spoof'
                        
                        if args.YOLO_FACE_SAVE and pred != target:
                            f_cropped_img.save(os.path.join(save_folder, 'crop_image_face', os.path.basename(path)))
                            logger.info("SAVE CROP FACE IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image_face', os.path.basename(path))))
                        
                        
                
                
                if not args.YOLO_DET_MASK and not args.YOLO_FACE and not args.YOLO:
                    preds_score.append(prob1[0])
                    pred = 'live' if prob1[0] > args.threshold else 'spoof'

            else:
                preds_score.append(prob1[0])
                pred = 'live' if prob1[0] > args.threshold else 'spoof'
            
            if args.FAS_SAVE and pred != target:
                img.save(os.path.join(save_folder, 'save', os.path.basename(path)))
                logger.info("SAVE IMAGE: path: {}".format(os.path.join(save_folder, 'save', os.path.basename(path))))
            
            logger.info(f"path: {os.path.join(args.root_dir, path)} - results: {pred == target} \n")
            targets_score.append(int(not label))  
            preds.append(pred)
            targets.append(target)
            logs_csv.append([os.path.join(args.root_dir, path), pred, target, pred == target])
        except Exception as e:
            logger.info(f"ERROR: {e} -- PATH : {os.path.join(args.root_dir, path)} --- LABEL: {label}")
    
    cm = confusion_matrix(targets, preds, labels=['live', 'spoof'])
    logger.info(f"confusion matrix: {cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['live', 'spoof'])
    fig, ax = plt.subplots(figsize=(8, 6)) # Optional: set figure size
    disp.plot(ax=ax) # Pass the ax object to the plot method
    plt.title("Confusion Matrix Live/Spoof") # Optional: add a title
    disp.figure_.savefig(os.path.join(save_folder, 'confusion_matrix_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    if len(np.unique(targets)) == 2:
        logger.info(classification_report(targets, preds, target_names=['live', 'spoof']))
    
    EER_score, threshold, _, _ = get_EER_states(np.array(preds_score), np.array(targets_score))
    HTER_score = get_HTER_at_thr(np.array(preds_score), np.array(targets_score), threshold)
    
    acc_threshold, cm_threshold = calculate_threshold(probs=np.array(preds_score), labels=np.array(targets_score), threshold=threshold)
    logger.info(f"acc_threshold: {acc_threshold} - cm_threshold: {cm_threshold}")
    
    df = pd.DataFrame(logs_csv, columns=['path', 'pred', 'target', 'output'])
    df.to_csv(os.path.join(save_folder, 'logs_results.csv'), index=False)
    
    logger.info(f"EER_score: {EER_score}, HTER_score: {HTER_score}, threshold: {threshold}")
    logger.info(f"checkpoint: {args.weights}")
    logger.info(f">>>>>>>>>>>>> Save Infer to : {save_folder} <<<<<<<<<<<<<<<<<<<")
    