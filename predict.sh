# for i in {44..56}
# do 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_ViT-B-16last_ckpt_260403.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/ibeta-test/test.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_ViT-B-16last_ckpt_260403.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/ibeta-test/test.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer" \
# --YOLO 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260327.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer" \
# --YOLO \
# --YOLO_FACE

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260327.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer" \
# --YOLO \
# --YOLO_FACE

# done

# for i in 11
# do 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt"

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --YOLO_DET_MASK \
# --weights_yolo_det_mask "/data/fas/solution/MVP-FAS/runs/weights/yolo_detect_canny_20261304.pt" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --YOLO_DET_MASK \
# --weights_yolo_det_mask "/data/fas/solution/MVP-FAS/runs/weights/yolo_detect_canny_20261304.pt" 


# --YOLO_DET_MASK \
# --weights_yolo_det_mask "/data/fas/solution/MVP-FAS/runs/weights/yolo_detect_canny_20261304.pt" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_P2_C.pth" \
# --YOLO_DET_MASK \
# --weights_yolo_det_mask "/data/fas/solution/MVP-FAS/runs/weights/yolo_detect_canny_20261304.pt" 

# --YOLO_DET_MASK \
# --weights_yolo_det_mask "/data/fas/solution/MVP-FAS/runs/weights/yolo_detect_canny_20261304.pt" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "runs/clip1/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE 

# done

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip1/train_14/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip1/train_14/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip1/train_14/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip1/train_19/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip1/train_14/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip1/train_14/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer1" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip1/train_16/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16last_ckpt.pt" 

# --test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \

# for i in 7
# do 

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer2" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/csv/full/backtest-full-level-data-260410.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer2" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer2" 


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer2" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip2/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" 

# done
clip=4
infer=14
for i in 7; do

python infer.py \
--backbone "ViT-B/16" \
--weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
--test_csv "/data/fas/csv/full/backtest-full-level-data-260428.csv" \
--root_dir "/data/fas" \
--input_size 224 \
--gpu_id 0 \
--save_path "runs/infer$infer" \
--YOLO_FACE \
--weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
--MVP_FAS_FACE_CROP \
--weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"

python infer.py \
--backbone "ViT-B/16" \
--weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
--test_csv "/data/fas/solution/MVP-FAS/logs/a_thanh_test/vft_athanh_test_zero_shot.csv" \
--root_dir "/data/fas" \
--input_size 224 \
--gpu_id 0 \
--save_path "runs/infer$infer" \
--YOLO_FACE \
--weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
--MVP_FAS_FACE_CROP \
--weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"

python infer.py \
--backbone "ViT-B/16" \
--weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
--test_csv "/data/fas/solution/MVP-FAS/logs/ibeta_review/ibeta_review.csv" \
--root_dir "/data/fas" \
--input_size 224 \
--gpu_id 0 \
--save_path "runs/infer$infer" \
--YOLO_FACE \
--weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
--MVP_FAS_FACE_CROP \
--weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"

#python infer.py \
#--backbone "ViT-B/16" \
#--weights "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_ViT-B-16_best_ckpt_250508.pt" \
#--test_csv "/data/fas/solution/MVP-FAS/logs/ibeta_review/ibeta_review.csv" \
#--root_dir "/data/fas" \
#--input_size 224 \
#--gpu_id 0 \
#--save_path "runs/infer$infer" \
#--YOLO_FACE \
#--weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
#--MVP_FAS_FACE_CROP \
#--weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/weights/MVP_FAS_ViT-B-16_best_ckpt_250508.pt"

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/solution/MVP-FAS/logs/test-vft/test-vft-260508.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer$infer" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"


# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/FFHQ/csv/live_FFHQ_260508.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer$infer" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip$clip/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"

# python infer.py \
# --backbone "ViT-B/16" \
# --weights "/data/fas/solution/MVP-FAS/runs/clip3/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
# --test_csv "/data/fas/kaggle/kaggle.csv" \
# --root_dir "/data/fas" \
# --input_size 224 \
# --gpu_id 0 \
# --save_path "runs/infer3" \
# --YOLO_FACE \
# --weights_yolo_det_face "/data/fas/solution/MVP-FAS/runs/weights/yolov12l-face.pt" \
# --MVP_FAS_FACE_CROP \
# --weights_mvp_face_crop "/data/fas/solution/MVP-FAS/runs/clip3/train_$i/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt"

done
