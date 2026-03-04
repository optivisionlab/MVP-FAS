export BACKBONE='ViT-B/16'
export WEIGHT='runs/clip/train_28/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt'
export DEVICE='0'
export HOST='0.0.0.0'
export PORT='9999'
export IMG_MODE='.jpg,.png,.JPEG,.jpeg,.jfif'
export IPHONE_MODE='.heic'
python app.py
