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
    logger.info(f"load checkpoint: {args.weights}")
    net = load_checkpoint(net, weight_path=args.weights)
    net.to(device)
    logger.info("load checkpoint is done! weight: {}".format(args.weights))
    
    # LOAD YOLO
    yolo_model = YOLO("yolo26l.pt")
    yolo_face_model = YOLO("runs/weights/yolov12l-face.pt")
    
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
            # img_resize = img.resize((224, 224))
            # img_resize.save("runs/results/input.png")
            # img = cv2.imread(os.path.join(args.root_dir, path))
            # pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            prob1 = infer_model(net, cfg, device, img=img)
            logger.info(f"\ninfer step 1: path: {os.path.join(args.root_dir, path)} - {prob1} - {'live' if prob1[0] > 0.5 else 'spoof'}")
            target = 'spoof' if label else 'live'
            if prob1[0] > 0.5:
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
                        logger.info(f"infer step 2: path: {os.path.join(args.root_dir, path)} - {prob2} - {'live' if prob2[0] > 0.5 else 'spoof'} \n")
                        
                        if prob2[0] > 0.5:
                            if args.YOLO_FACE:
                                createDirectory(os.path.join(save_folder, 'crop_image_face'))
                                results_face = yolo_face_model.predict(cropped_img, device=device, conf=0.5)
                                
                                for f_result in results_face:
                                    
                                    f_boxes = f_result.boxes
                                    f_best_idx = int(np.argmax(f_boxes.conf.cpu().numpy()))
                                    f_x_min, f_y_min, f_x_max, f_y_max = map(int, f_boxes.xyxy[f_best_idx].cpu().numpy())
                                    f_bbox = [f_x_min, f_y_min, f_x_max, f_y_max]
                                    f_cropped_img = cropped_img.crop(tuple(f_bbox))
                                    
                                    prob3 = infer_model(net, cfg, device, img=f_cropped_img)
                                    preds_score.append(prob3[0])
                                    pred = 'live' if prob2[0] > 0.5 else 'spoof'
                                    
                                    if args.YOLO_FACE_SAVE and pred != target:
                                        cropped_img.save(os.path.join(save_folder, 'crop_image', os.path.basename(path)))
                                        f_cropped_img.save(os.path.join(save_folder, 'crop_image_face', os.path.basename(path)))
                                        logger.info("SAVE CROP FACE IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image_face', os.path.basename(path))))
                                        logger.info("SAVE CROP IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image', os.path.basename(path))))
                                    
                                    logger.info(f"infer step 3: path: {os.path.join(args.root_dir, path)} - {prob3} - {'live' if prob3[0] > 0.5 else 'spoof'} \n")
                            else:
                                preds_score.append(prob2[0])
                                pred = 'live' if prob2[0] > 0.5 else 'spoof'
                                if args.YOLO_SAVE and pred != target:
                                    cropped_img.save(os.path.join(save_folder, 'crop_image', os.path.basename(path)))
                                    logger.info("SAVE CROP IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image', os.path.basename(path))))
                                        
                        else:
                            preds_score.append(prob2[0])
                            pred = 'live' if prob2[0] > 0.5 else 'spoof'
                            if args.YOLO_SAVE and pred != target:
                                cropped_img.save(os.path.join(save_folder, 'crop_image', os.path.basename(path)))
                                logger.info("SAVE CROP IMAGE: path: {}".format(os.path.join(save_folder, 'crop_image', os.path.basename(path))))
                            
                else:
                    preds_score.append(prob1[0])
                    pred = 'live' if prob1[0] > 0.5 else 'spoof'

            else:
                preds_score.append(prob1[0])
                pred = 'live' if prob1[0] > 0.5 else 'spoof'
            
            if args.FAS_SAVE and pred != target:
                img.save(os.path.join(save_folder, 'save', os.path.basename(path)))
                logger.info("SAVE IMAGE: path: {}".format(os.path.join(save_folder, 'save', os.path.basename(path))))
              
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

    df = pd.DataFrame(logs_csv, columns=['path', 'pred', 'target', 'output'])
    df.to_csv(os.path.join(save_folder, 'logs_results.csv'), index=False)
    
    logger.info(f"EER_score: {EER_score}, HTER_score: {HTER_score}, threshold: {threshold}")
    logger.info(f"checkpoint: {args.weights}")
    logger.info(f">>>>>>>>>>>>> Save Infer to : {save_folder} <<<<<<<<<<<<<<<<<<<")
    