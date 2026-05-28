# Slot Attention Visualization cho MVP-FAS

Visualize Multi-View Slot Attention maps từ model MVP-FAS, tương tự như Figure trong paper ICCV 2025.

## 🎯 Mục đích

Hiển thị attention maps cho từng text prompt (slot) để hiểu model đang focus vào đâu khi phân loại với từng loại attack description.

## 📊 Output Format

Giống như paper figure, mỗi visualization gồm:
- **Input**: Ảnh gốc
- **Slot 1-5**: Attention maps cho các text prompts khác nhau
  - Spoof face
  - Attack face  
  - Fake face
  - Real face
  - Baseline

## 🚀 Cách sử dụng

### 1. Visualize một ảnh đơn

```bash
bash demo_slots.sh
```

Hoặc:

```bash
/data/miniconda3/envs/torch128/bin/python3 visualize_slots.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "path/to/weights.pt" \
    --image "path/to/image.jpg" \
    --save_path "runs/visualize_slots" \
    --prompts "spoof face" "attack face" "fake face" "real face" "baseline"
```

### 2. Visualize nhiều ảnh (multi-row)

```bash
bash demo_slots_multi.sh
```

Hoặc:

```bash
/data/miniconda3/envs/torch128/bin/python3 visualize_slots.py \
    --model MVP_FAS \
    --backbone "ViT-B/16" \
    --weights "path/to/weights.pt" \
    --images "image1.jpg" "image2.jpg" "image3.jpg" \
    --save_path "runs/visualize_slots" \
    --prompts "spoof face" "attack face" "fake face" "real face" "baseline"
```

## 🎨 Customization

### Thay đổi text prompts

Bạn có thể customize các text prompts để visualize attention cho các loại attack khác:

```bash
python3 visualize_slots.py \
    --weights "weights.pt" \
    --image "image.jpg" \
    --prompts "replay attack face" "printed photo face spoof" "3D mask face spoof" "silicone face spoof" "real face"
```

### Available prompts từ model

**Spoof templates:**
- spoof face
- attack face
- fake face
- replay attack face
- printed photo face spoof
- 2D attack face
- silicone face spoof
- latex face spoof
- 3D mask face spoof
- full mask face spoof
- paper mask face spoof
- mannequin face spoof
- makeup attack face spoof
- covered mouth face spoof
- covered eye face spoof
- fake glasses face spoof

**Real templates:**
- real face
- bonafide face
- genuine face
- true face
- live face
- left half of real face
- right half of real face
- left half of live face
- right half of live face

## 📝 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | MVP_FAS | Tên model |
| `--backbone` | str | ViT-B/16 | Backbone architecture |
| `--weights` | str | **required** | Path đến model weights |
| `--image` | str | None | Path đến ảnh input (single) |
| `--images` | str[] | None | Paths đến nhiều ảnh (multi-row) |
| `--save_path` | str | runs/visualize_slots | Thư mục lưu output |
| `--prompts` | str[] | [spoof, attack, fake, real, baseline] | Text prompts cho slots |
| `--gpu_id` | int | 0 | GPU ID |

## 🔍 Cách đọc Slot Attention Maps

### Màu sắc
- 🔴 **Đỏ (High attention)**: Vùng model attend nhiều nhất cho text prompt này
- 🟡 **Vàng (Medium attention)**: Vùng có attention trung bình
- 🔵 **Xanh (Low attention)**: Vùng ít attention

### Interpretation

**Spoof face slots:**
- Nên focus vào artifacts: screen edges, print borders, moiré patterns
- Attention vào vùng có texture bất thường
- Focus vào reflections, glare

**Real face slots:**
- Nên focus vào facial features: eyes, nose, mouth
- Attention phân bố đều trên khuôn mặt
- Focus vào skin texture tự nhiên

**Good vs Bad Attention:**

✅ **Good attention (model học tốt):**
- Spoof slots focus vào artifacts rõ ràng
- Real slots focus vào facial features
- Các slots khác nhau có attention patterns khác nhau

❌ **Bad attention (model chưa học tốt):**
- Tất cả slots có attention giống nhau
- Focus vào background thay vì face
- Không có pattern rõ ràng

## 💡 Use Cases

### 1. Debug model behavior
Xem model đang dùng cues gì để phân loại:
```bash
# Visualize failed predictions
python3 visualize_slots.py \
    --weights "weights.pt" \
    --image "failed_case.jpg" \
    --prompts "spoof face" "attack face" "fake face" "real face" "baseline"
```

### 2. Phân tích attack types
So sánh attention patterns cho các loại attack:
```bash
# Print attack
python3 visualize_slots.py \
    --weights "weights.pt" \
    --image "print_attack.jpg" \
    --prompts "printed photo face spoof" "2D attack face" "real face"

# Replay attack  
python3 visualize_slots.py \
    --weights "weights.pt" \
    --image "replay_attack.jpg" \
    --prompts "replay attack face" "spoof face" "real face"

# 3D mask
python3 visualize_slots.py \
    --weights "weights.pt" \
    --image "mask_attack.jpg" \
    --prompts "3D mask face spoof" "silicone face spoof" "real face"
```

### 3. So sánh models
Visualize cùng ảnh với nhiều checkpoints:
```bash
for ckpt in runs/clip*/train_*/MVP_FAS_ViT-B-16/weights/*_best_ckpt.pt; do
    python3 visualize_slots.py \
        --weights "$ckpt" \
        --image "test.jpg" \
        --save_path "runs/compare_slots/$(basename $(dirname $(dirname $ckpt)))"
done
```

### 4. Paper-style multi-row figure
Tạo figure giống paper với nhiều examples:
```bash
python3 visualize_slots.py \
    --weights "weights.pt" \
    --images "live1.jpg" "spoof1.jpg" "live2.jpg" "spoof2.jpg" \
    --prompts "spoof face" "attack face" "fake face" "real face" "baseline" \
    --save_path "runs/paper_figure"
```

## 🔧 Technical Details

### Slot Attention Mechanism

MVP-FAS sử dụng **SlotAttention_PQTK** với:
- **Query**: Patch features từ image
- **Key/Value**: Text embeddings từ CLIP
- **Iterations**: 3 iterations để refine slots
- **Output**: Attention weights `[batch, num_patches, num_text_features]`

### Extraction Process

1. Hook vào `MVSlot.forward()` để capture attention weights
2. Lấy attention từ iteration cuối cùng
3. Transpose nếu cần: `[batch, num_patches, num_texts]` → `[batch, num_texts, num_patches]`
4. Reshape patches thành 2D grid (14x14 cho ViT-B/16)
5. Normalize và apply colormap

### Attention Shape

- Input image: 224x224
- Patch size: 16x16
- Number of patches: 14x14 = 196
- Number of text features: ~25 (spoof + real templates)
- Attention shape: `[1, 25, 196]`

## 📚 References

- [MVP-FAS Paper (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Multi-View_Slot_Attention_Using_Paraphrased_Texts_for_Face_Anti-Spoofing_ICCV_2025_paper.html)
- [Slot Attention Paper](https://arxiv.org/abs/2006.15055)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)

## 🐛 Troubleshooting

### Lỗi: "Attention shape mismatch"
- Model architecture có thể khác
- Kiểm tra backbone (ViT-B/16 vs RN50)

### Lỗi: "CUDA out of memory"
- Giảm số lượng images trong multi-row
- Dùng CPU: `--gpu_id -1`

### Attention maps toàn màu đồng nhất
- Model chưa converge
- Thử checkpoint khác
- Kiểm tra input image quality

## 📧 Contact

Nếu có vấn đề, tạo issue hoặc liên hệ team.
