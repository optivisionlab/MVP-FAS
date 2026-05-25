
# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
# --key_train "vft_20260415_crop_face,vft_20260330_crop_face,VFT_20260324_crop_face,vft_hq_260320_crop_face,VFT_hvq_260320_crop_face,cefa_image,CASIA_SURF_images,oulu_crop_face,Axonlab_1_crop_face,Axonlab_2_crop_face,SiW_crop_face,VFT_video_20260226_crop_face,VFT_video_20260313_crop_face,vft_imgs_260312_crop_face,axonlab_live_release_260320_crop_face" \
# --key_val "CASIA_crop_face,MSU_crop_face,VFT_crop_face,VFT_video_1_crop_face,VFT_video_2_crop_face,VFT_video_20260306_crop_face,vft-lq-260320_crop_face,vft_dnt_260320_crop_face,vft_20260327_crop_face" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 32 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 20 \
# --save_path "runs/clip_face" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_C.pth" \
# --pretrained \
# --is_physical


# python train.py \
# --root_dir "/data/fas" \
# --full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
# --key_train "vft_20260415_crop_face,vft_20260330_crop_face,VFT_20260324_crop_face,vft_hq_260320_crop_face,VFT_hvq_260320_crop_face,cefa_image,CASIA_SURF_images,oulu_crop_face,Axonlab_1_crop_face,Axonlab_2_crop_face,SiW_crop_face,VFT_video_20260226_crop_face,VFT_video_20260313_crop_face,vft_imgs_260312_crop_face,axonlab_live_release_260320_crop_face" \
# --key_val "CASIA_crop_face,MSU_crop_face,VFT_crop_face,VFT_video_1_crop_face,VFT_video_2_crop_face,VFT_video_20260306_crop_face,vft-lq-260320_crop_face,vft_dnt_260320_crop_face,vft_20260327_crop_face" \
# --setting "all" \
# --backbone "ViT-B/16" \
# --batch_size 32 \
# --input_size 224 \
# --setting "all" \
# --gpu_id 0 \
# --num_epochs 20 \
# --save_path "runs/clip_face" \
# --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_S.pth" \
# --pretrained \
# --is_physical


python train.py \
--root_dir "/data/fas" \
--full_dataset_csv "/data/fas/csv/full/full-level-data-260415.csv" \
--key_train "vft_20260415_crop_face,vft_20260330_crop_face,VFT_20260324_crop_face,vft_hq_260320_crop_face,VFT_hvq_260320_crop_face,cefa_image,CASIA_SURF_images,oulu_crop_face,Axonlab_1_crop_face,Axonlab_2_crop_face,SiW_crop_face,VFT_video_20260226_crop_face,VFT_video_20260313_crop_face,vft_imgs_260312_crop_face,axonlab_live_release_260320_crop_face" \
--key_val "CASIA_crop_face,MSU_crop_face,VFT_crop_face,VFT_video_1_crop_face,VFT_video_2_crop_face,VFT_video_20260306_crop_face,vft-lq-260320_crop_face,vft_dnt_260320_crop_face,vft_20260327_crop_face" \
--setting "all" \
--backbone "ViT-B/16" \
--batch_size 32 \
--input_size 224 \
--setting "all" \
--gpu_id 0 \
--num_epochs 20 \
--save_path "runs/clip_face" \
--checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_W.pth" \
--pretrained \
--is_physical