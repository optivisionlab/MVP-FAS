# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "oulu,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330,cefa_image,CASIA_SURF_images" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 5 \
# --save_path "runs/clip1" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_C.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "oulu,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330,cefa_image,CASIA_SURF_images" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 5 \
# --save_path "runs/clip1" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_W.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "oulu,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330,cefa_image,CASIA_SURF_images" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 10 \
# --save_path "runs/clip1" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_C.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330,cefa_image,CASIA_SURF_images" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU,oulu" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 5 \
# --save_path "runs/clip1" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_W.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
# --key_train "vft_20260415,oulu,MSU,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
# --key_val "VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 64 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 20 \
# --save_path "runs/clip2" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_C.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
# --key_train "vft_20260415,oulu,MSU,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
# --key_val "VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 64 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 20 \
# --save_path "runs/clip2" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_I.pth" \
# --pretrained \
# --is_physical

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
# --key_train "vft_20260415,oulu,MSU,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
# --key_val "VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 64 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 20 \
# --save_path "runs/clip2" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_M.pth" \
# --pretrained \
# --is_physical
# 

# --key_train "live_FFHQ_p4,VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA,oulu,MSU,celeba,live_FFHQ_p1,vft_20260415,SiW,Axonlab_1,Axonlab_2,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
# I M C O
for char in I M C O
do
# celeba
python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/full/full-level-data-260520.csv" \
--key_train "celeba,oulu,MSU,live_FFHQ_p1,vft_20260415,SiW,Axonlab_1,Axonlab_2,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
--key_val "live_FFHQ_p4,VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 64 \
--input_size 448 \
--setting "all" \
--gpu_id 0 \
--num_epochs 10 \
--save_path "runs/clip10" \
--checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_$char.pth" \
--pretrained \
--is_physical

done

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "oulu,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 15 \
# --save_path "runs/clip1" \
# --is_physical
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_C.pth" \
# --pretrained \

# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260408.csv" \
# --key_train "oulu,Axonlab_1,Axonlab_2,celeba,SiW,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,VFT,vft_hq_260320,VFT_20260324,vft_20260330" \
# --key_val "CASIA,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,MSU" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 128 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 15 \
# --save_path "runs/clip1" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_W.pth" \
# --pretrained \
# --is_physicall