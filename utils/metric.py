import math
import re
import numpy as np
import torch

try:
    from scipy.integrate import simps
except:
    from scipy.integrate import simpson as simps

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Strips the trailing frame index (frame0, frame1, frame_xx, ...) and extension
# from a frame's image path, so every frame sampled from the same video collapses
# to the same key (e.g. ".../1_flip_frame_0.png" and ".../1_flip_frame_1.png"
# both map to ".../1_flip").
_FRAME_SUFFIX_RE = re.compile(r'_?frame_?\d+(\.[A-Za-z0-9]+)?$', re.IGNORECASE)


def get_video_id(img_path):
    return _FRAME_SUFFIX_RE.sub('', img_path)

def get_threshold(probs, grid_density=10000):
    Min = np.min(probs)
    Max = np.max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thr = Min + i * (Max - Min) / float(grid_density)
        thresholds.append(thr)
    return thresholds

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def get_EER_states(probs, labels, grid_density=10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist < min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    
    confusion_matrix = {
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP
    }
    return ACC, confusion_matrix

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif (FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER, FAR, FRR

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Metric():
    def __init__(self):
        self.labels = []
        self.probs = []
        self.video_ids = []

    def update(self, GT, pred, video_ids=None):
        self.labels.append(GT)
        self.probs.append(pred)
        if video_ids is not None:
            self.video_ids.extend(video_ids)

    def compute(self, ext_threshold=None):
        labels = np.concatenate(self.labels)
        probs = np.concatenate(self.probs)

        if self.video_ids:
            assert len(self.video_ids) == len(probs), \
                "video_ids must be provided for every update() call, or none at all"

            # Step 2+3: gộp per-frame prob/label theo videoID, rồi lấy trung bình
            # prob của các frame cùng video làm prob cuối cùng của video đó.
            prob_dict, label_dict = {}, {}
            for vid, p, l in zip(self.video_ids, probs, labels):
                prob_dict.setdefault(vid, []).append(p)
                label_dict.setdefault(vid, l)

            video_ids = list(prob_dict.keys())
            probs = np.array([sum(prob_dict[vid]) / len(prob_dict[vid]) for vid in video_ids])
            labels = np.array([label_dict[vid] for vid in video_ids])

        acc = ((probs > 0.5) == labels).mean()

        # own-set EER/threshold is always reported for reference, but HTER must be
        # measured at a threshold fixed on a *different* split (ext_threshold) —
        # otherwise it collapses to EER (same probs/labels picked the threshold).
        eer, threshold, _, _ = get_EER_states(probs, labels)
        hter_threshold = threshold if ext_threshold is None else ext_threshold
        hter, far, frr = get_HTER_at_thr(probs, labels, hter_threshold)
        acc_thr, _ = calculate_threshold(probs, labels, hter_threshold)

        auc = -1
        tpr_fpr = -1

        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
            fpr, tpr, _ = roc_curve(labels, probs)
            tpr_filtered = tpr[fpr <= 0.01]
            tpr_fpr = tpr_filtered[-1] if len(tpr_filtered) > 0 else 0

        return acc, eer, hter, auc, threshold, acc_thr, tpr_fpr

