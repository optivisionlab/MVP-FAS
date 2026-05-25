# Heatmap Visualization cho MVP-FAS

Script để visualize heatmap/attention map của model MVP-FAS, giúp hiểu model đang focus vào đâu khi phân loại live/spoof.

## 📋 Tính năng

- **Grad-CAM**: Hiển thị vùng quan trọng nhất cho quyết định của model
- **Attention Map**: Visualize attention weights từ Vision Transformer
- **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- **Overlay**: Hiển thị heatmap chồng lên ảnh gốc

## 🚀 Cách sử dụng

### 1. Visualize một ảnh đơn

```bash
python visualize_heatmap.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "runs/clip6/train_2/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
    --image "path/to/your/image.jpg" \
    --save_path "runs/visualize" \
    --method "gradcam" \
    --gpu_id 0
```

**Hoặc dùng script demo:**
```bash
bash demo_heatmap.sh
```

### 2. Batch visualize nhiều ảnh

**Từ CSV file:**
```bash
python batch_visualize.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "runs/clip6/train_2/MVP_FAS_ViT-B-16/weights/MVP_FAS_ViT-B-16_best_ckpt.pt" \
    --csv_file "logs/ibeta_review/ibeta_review.csv" \
    --root_dir "/data/fas" \
    --save_path "runs/visualize_batch" \
    --method "gradcam" \
    --max_images 50 \
    --gpu_id 0
```

**Hoặc dùng script demo:**
```bash
bash demo_batch_heatmap.sh
```

**Từ thư mục ảnh:**
```bash
python batch_visualize.py \
    --weights "path/to/weights.pt" \
    --image_dir "path/to/images/" \
    --save_path "runs/visualize_batch" \
    --method "gradcam"
```

## 📊 Output

Mỗi ảnh sẽ tạo ra một visualization gồm 3 phần:

1. **Original Image**: Ảnh gốc
2. **Heatmap**: Bản đồ nhiệt (đỏ = quan trọng, xanh = ít quan trọng)
3. **Overlay**: Heatmap chồng lên ảnh gốc

Kèm theo thông tin:
- Live probability
- Spoof probability
- Prediction (LIVE/SPOOF)
- Confidence score

## 🎨 Visualization Methods

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Hiển thị vùng ảnh có gradient lớn nhất đối với class prediction
- Tốt cho việc hiểu model focus vào đâu để đưa ra quyết định
- **Khuyến nghị sử dụng cho FAS**

```bash
--method gradcam
```

### Attention Map
- Visualize attention weights từ Vision Transformer
- Hiển thị các patch mà model attend to
- Tốt cho việc hiểu cơ chế attention

```bash
--method attention
```

## 📝 Parameters

### visualize_heatmap.py

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | MVP_FAS | Tên model |
| `--backbone` | str | ViT-B/16 | Backbone architecture |
| `--weights` | str | **required** | Path đến model weights |
| `--image` | str | **required** | Path đến ảnh input |
| `--save_path` | str | runs/visualize | Thư mục lưu output |
| `--method` | str | gradcam | Method: gradcam hoặc attention |
| `--gpu_id` | int | 0 | GPU ID |

### batch_visualize.py

Tất cả parameters của `visualize_heatmap.py` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--csv_file` | str | None | CSV file chứa danh sách ảnh |
| `--image_dir` | str | None | Thư mục chứa ảnh |
| `--root_dir` | str | "" | Root directory cho paths trong CSV |
| `--max_images` | int | -1 | Số ảnh tối đa (-1 = all) |

## 💡 Use Cases

### 1. Debug model predictions
Xem model đang focus vào đâu khi predict sai:
```bash
# Visualize các ảnh predict sai
python batch_visualize.py \
    --weights "weights.pt" \
    --csv_file "logs/errors.csv" \
    --save_path "runs/debug_errors"
```

### 2. Phân tích attack types
Xem model phát hiện các loại attack như thế nào:
```bash
# Visualize print attacks
python batch_visualize.py \
    --weights "weights.pt" \
    --csv_file "logs/print_attacks.csv" \
    --save_path "runs/analyze_print"

# Visualize replay attacks
python batch_visualize.py \
    --weights "weights.pt" \
    --csv_file "logs/replay_attacks.csv" \
    --save_path "runs/analyze_replay"
```

### 3. So sánh models
Visualize cùng một ảnh với nhiều checkpoints khác nhau:
```bash
for ckpt in runs/clip6/train_*/MVP_FAS_ViT-B-16/weights/*_best_ckpt.pt; do
    python visualize_heatmap.py \
        --weights "$ckpt" \
        --image "test_image.jpg" \
        --save_path "runs/compare_models/$(basename $(dirname $(dirname $ckpt)))"
done
```

### 4. Chuẩn bị cho iBeta review
Visualize các ảnh trong iBeta test set:
```bash
python batch_visualize.py \
    --weights "best_model.pt" \
    --csv_file "logs/ibeta_review/ibeta_review.csv" \
    --root_dir "/data/fas" \
    --save_path "runs/ibeta_visualization" \
    --max_images -1
```

## 🔍 Interpretation Guide

### Heatmap Colors
- 🔴 **Đỏ (High activation)**: Vùng quan trọng nhất cho prediction
- 🟡 **Vàng (Medium activation)**: Vùng có ảnh hưởng trung bình
- 🔵 **Xanh (Low activation)**: Vùng ít quan trọng

### Good vs Bad Heatmaps

**✅ Good Heatmap (Live face):**
- Focus vào mắt, mũi, miệng (facial features)
- Attention phân bố đều trên khuôn mặt
- Không focus vào background

**✅ Good Heatmap (Spoof detection):**
- Focus vào artifacts: screen edges, print borders, mask edges
- Attention vào vùng có texture khác thường
- Focus vào moiré patterns, reflections

**❌ Bad Heatmap:**
- Focus vào background thay vì face
- Attention quá tập trung vào một điểm nhỏ
- Không có pattern rõ ràng

## 🛠️ Troubleshooting

### Lỗi: "No attention weights captured"
- Model không có attention mechanism hoặc hook không đúng layer
- Thử dùng method `gradcam` thay vì `attention`

### Lỗi: "CUDA out of memory"
- Giảm batch size hoặc image size
- Dùng CPU: `--gpu_id -1`

### Heatmap toàn màu đồng nhất
- Model chưa được train tốt
- Thử với checkpoint khác
- Kiểm tra input image có bị corrupt không

## 📚 References

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Vision Transformer Attention](https://arxiv.org/abs/2010.11929)
- [MVP-FAS Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Multi-View_Slot_Attention_Using_Paraphrased_Texts_for_Face_Anti-Spoofing_ICCV_2025_paper.html)

## 📧 Contact

Nếu có vấn đề, tạo issue hoặc liên hệ team.
