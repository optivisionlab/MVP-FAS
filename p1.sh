KEY_TRAIN="oulu,MSU,CASIA"

KEY_VAL="replay_attack"

save_clip=1

python train.py \
    --root_dir "/data/fas" \
    --full_dataset_csv "/data/fas/csv/p1/full-level-data-p1-flip.csv" \
    --key_train $KEY_TRAIN \
    --key_val $KEY_VAL \
    --setting "all" \
    --backbone "ViT-B/16" \
    --batch_size 128 \
    --input_size 224 \
    --setting "all" \
    --gpu_id 0 \
    --num_epochs 300 \
    --save_path "/u01/vision/data/fas/solution/runs/protocol1/clip$save_clip" \
    --checkpoint "/u01/vision/data/fas/solution/runs/celeba/clip1/train_2/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
    --pretrained \
    --supcon_action 