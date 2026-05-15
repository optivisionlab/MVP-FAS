import torchvision.transforms as transforms
import torch
from loaders.make_dataset import RemoveBlackBorders
from torch.nn import functional as F
import numpy as np

def crop_face_with_expand(
    img,
    yolo_face_model,
    device="cuda",
    conf=0.5,
    scale_w=1.3,
    scale_h=1.5,
    square=False
):
    """
    Crop face từ YOLO + mở rộng bbox (phù hợp FAS)

    Args:
        img: PIL Image
        yolo_face_model: model YOLO detect face
        device: cpu/cuda
        conf: confidence threshold
        scale_w: scale chiều ngang
        scale_h: scale chiều dọc
        square: ép bbox thành hình vuông

    Returns:
        cropped_img (PIL) hoặc None nếu không detect được face
        bbox (list) hoặc None
    """

    results = yolo_face_model.predict(
        img,
        device=device,
        conf=conf,
        classes=[0],
        verbose=False
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            return None, None

        boxes = result.boxes
        best_idx = int(np.argmax(boxes.conf.cpu().numpy()))

        x_min, y_min, x_max, y_max = map(int, boxes.xyxy[best_idx].cpu().numpy())

        # ===== expand bbox =====
        w = x_max - x_min
        h = y_max - y_min

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        new_w = w * scale_w
        new_h = h * scale_h

        # optional: ép vuông
        if square:
            side = max(new_w, new_h)
            new_w = new_h = side

        new_x_min = int(cx - new_w / 2)
        new_x_max = int(cx + new_w / 2)
        new_y_min = int(cy - new_h / 2)
        new_y_max = int(cy + new_h / 2)

        # ===== clip boundary =====
        img_w, img_h = img.size

        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(img_w, new_x_max)
        new_y_max = min(img_h, new_y_max)

        bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

        cropped_img = img.crop(tuple(bbox))

        return cropped_img, bbox

    return None, None


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
