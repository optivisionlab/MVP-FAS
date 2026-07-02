
clip="16 18"
for c in $clip; do
    for i in {1..17}; do

    rm /data/fas/solution/MVP-FAS/runs/clip$c/train_$i/MVP_FAS_ViT-B-16/weights/*.pt

    done
done