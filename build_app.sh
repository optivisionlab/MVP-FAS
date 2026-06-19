export BACKBONE='ViT-B/16'
export WEIGHT='/data/manhquang/dataset/fas/solution/MVP-FAS/runs/weights/MVP_FAS_ViT-B-16_best_ckpt_260618.pt'
export DEVICE='0'
export HOST='0.0.0.0'
export PORT='6688'
export IMG_MODE='.jpg,.png,.JPEG,.jpeg,.jfif'
export IPHONE_MODE='.heic'
python app.py
