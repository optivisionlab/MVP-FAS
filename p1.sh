# VEGA_FAS_ITW    7462
# oulu            4950
# SiW             1700
# CASIA            600
# MSU              280

python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/vega-fas/cross-domain-fas-p1.csv" \
--key_train "VEGA_FAS_ITW" \
--key_val "oulu" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 100 \
--save_path "runs/cross_domain" 

python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/vega-fas/cross-domain-fas-p1.csv" \
--key_train "VEGA_FAS_ITW" \
--key_val "CASIA" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 100 \
--save_path "runs/cross_domain"

python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/vega-fas/cross-domain-fas-p1.csv" \
--key_train "VEGA_FAS_ITW" \
--key_val "MSU" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 100 \
--save_path "runs/cross_domain"


python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/vega-fas/cross-domain-fas-p1.csv" \
--key_train "VEGA_FAS_ITW" \
--key_val "SiW" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 100 \
--save_path "runs/cross_domain"