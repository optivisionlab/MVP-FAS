
csv_list=(
    # /data/fas/csv/vft/reddis_image_260611_live.csv
    # /data/fas/csv/vft/live_data_from_gw_050626.csv
    /data/fas/csv/ibeta/ibeta_review_260602.csv
    # /data/fas/csv/full/backtest-full-level-data-260428.csv
    # /data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv
    # /data/fas/solution/MVP-FAS/logs/ibeta_review/ibeta_review.csv
    # /data/fas/csv/vft/reddis_image_260615_live.csv
    # /data/fas/test2.csv
)

# Iterate over the array safely

clip=17
infer=22
for i in 4; do
    for csv in "${csv_list[@]}"; do
        python infer.py \
        --backbone "ViT-B/16" \
        --weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
        --test_csv "$csv" \
        --root_dir "/data/fas" \
        --input_size 224 \
        --gpu_id 0 \
        --save_path "runs/infer$infer"
    done
done