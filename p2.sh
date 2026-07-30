# object
# CASIA_SURF_images    96584
# cefa_image           70295
# wmca_rgb              8280
# Name: count, dtype: int64

KEY_TRAIN=("CASIA_SURF_images,wmca_rgb", "cefa_image,wmca_rgb", "CASIA_SURF_images,cefa_image")

KEY_VAL=("cefa_image", "CASIA_SURF_images", "wmca_rgb")

# for i in "${!KEY_TRAIN[@]}"; do
#     echo ${KEY_TRAIN[i]}
#     echo ${KEY_VAL[i]}
#     echo "=============="
# done

save_clip=1

for i in "${!KEY_TRAIN[@]}"; do
    python train.py \
        --root_dir "/u01/vision/data" \
        --full_dataset_csv "/data/fas/csv/p2/fas_p2.csv" \
        --key_train "${KEY_TRAIN[i]}" \
        --key_val "${KEY_VAL[i]}" \
        --setting "all" \
        --backbone "ViT-B/16" \
        --batch_size 128 \
        --input_size 224 \
        --setting "all" \
        --gpu_id 0 \
        --num_epochs 300 \
        --save_path "/u01/vision/data/fas/solution/runs/protocol2/clip$save_clip" \
        --supcon_action 
done