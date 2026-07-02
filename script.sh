# --key_train "live_FFHQ_p4,VFT,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,vft_dnt_260320,vft_20260327,CASIA,oulu,MSU,celeba,live_FFHQ_p1,vft_20260415,SiW,Axonlab_1,Axonlab_2,VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,axonlab_live_release_260320,VFT_hvq_260320,vft_hq_260320,VFT_20260324,vft_20260330" \
# I M C O

# train: 
# val: 

KEY_TRAIN="celeba,oulu,MSU,live_FFHQ_p1,vft_20260415,siw_mv2_released_260602,\
VFT_video_20260226,VFT_video_20260313,vft_imgs_260312,VFT_hvq_260320,vft_hq_260320,\
VFT_20260324,vft_20260330,VFT_video_1,VFT_video_2,VFT_video_20260306,vft-lq-260320,\
vft_dnt_260320,vft_20260327,vft_hvq_260529,vft_hoang_260528,vft_duong_260528,\
vft_huy_260610_v3,vft_040626,vft_huy_260610_v2,vft-duong-260610,vft_spoof_260610,\
malegrooming,PrettyGirlsUglyFaces,lookyourbest,JustMyFace,selfie,selfies,SelfiesGoneMild,part1,part2"

KEY_VAL="VFT,live_FFHQ_p4,CASIA,Axonlab_1,Axonlab_2,axonlab_live_release_260320,vft_live_030626,part3,part4"

save_clip=27

# 
for char in C I M O
do
# celeba size 448 vft_imgs_260312
    python train.py \
    --root_dir "/data/fas" \
    --full_dataset_csv "/data/fas/csv/full/full-level-data-260615.csv" \
    --key_train $KEY_TRAIN \
    --key_val $KEY_VAL \
    --setting "all" \
    --backbone "ViT-B/16" \
    --batch_size 128 \
    --input_size 224 \
    --setting "all" \
    --gpu_id 0 \
    --num_epochs 10 \
    --save_path "/u01/vision/data/fas/solution/runs/clip$save_clip" \
    --checkpoint "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_$char.pth" \
    --pretrained \
    --is_physical \
    --supcon_action 
done

# /data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P1_$char.pth
