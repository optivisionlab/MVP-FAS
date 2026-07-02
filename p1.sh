# live_FFHQ_p1        11740
# VEGA_FAS_ITW         7462
# oulu                 4950
# live_FFHQ_p1_lv2     4446
# live_FFHQ_p4         2703
# SiW                  1700
# live_FFHQ_p4_lv2      607
# CASIA                 600
# MSU                   280

for item in "oulu" "SiW" "CASIA" "MSU"
do
python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/vega-fas/cross-domain-fas-p1-260525.csv" \
--key_train "VEGA_FAS_ITW,live_FFHQ_p1,live_FFHQ_p1_lv2,live_FFHQ_p4,live_FFHQ_p4_lv2" \
--key_val $item \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 100 \
--save_path "runs/cross_domain" \
--is_physical 
done