# Phân Tích So Sánh: Tại Sao 2.JPG Predict LIVE và 1.JPG Predict SPOOF

## 📊 Tóm Tắt Kết Quả

| Ảnh | Prediction | Confidence | Live Prob | Spoof Prob | Margin |
|-----|------------|------------|-----------|------------|--------|
| **2.JPG** | **LIVE** | **94.21%** | 0.9421 | 0.0579 | 0.8841 |
| **1.JPG** | **SPOOF** | **70.19%** | 0.2981 | 0.7019 | 0.4038 |

### Quan Sát Chính:
- **2.JPG** được predict là LIVE với confidence rất cao (94.21%)
- **1.JPG** được predict là SPOOF với confidence trung bình (70.19%)
- Margin giữa live/spoof prob của 2.JPG (0.8841) cao hơn nhiều so với 1.JPG (0.4038)

---

## 🔍 Phân Tích Chi Tiết

### 1. **Attention Pattern Analysis**

#### Attention Statistics:
Cả 2 ảnh đều có:
- **25 text slots** (spoof templates + real templates)
- **196 patches** (14x14 grid từ ViT-B/16)
- **Entropy tương tự** (~5.278): Attention phân bố đều, không quá tập trung

#### Spoof vs Real Attention Ratio:

| Ảnh | Spoof Slots Mean | Real Slots Mean | Ratio (Real/Spoof) |
|-----|------------------|-----------------|-------------------|
| 2.JPG (LIVE) | 0.039997 | 0.040005 | **1.0002** |
| 1.JPG (SPOOF) | 0.039995 | 0.040007 | **1.0003** |

**⚠️ Vấn đề:** Ratio gần như giống nhau (1.0002 vs 1.0003) - không có sự khác biệt rõ ràng!

### 2. **Top Attended Patches**

#### 2.JPG (LIVE):
Top patches focus vào:
- Patch 0 (row=0, col=0) - **góc trên trái**
- Patch 177 (row=12, col=9) - **dưới phải**
- Patch 173 (row=12, col=5) - **dưới giữa**
- Patch 122 (row=8, col=10) - **giữa phải**
- Patch 143 (row=10, col=3) - **dưới trái**

**Pattern:** Attention phân bố ở **các góc và biên** của ảnh

#### 1.JPG (SPOOF):
Top patches focus vào:
- Patch 115 (row=8, col=3) - **giữa trái**
- Patch 60 (row=4, col=4) - **trên giữa**
- Patch 102 (row=7, col=4) - **giữa**
- Patch 101 (row=7, col=3) - **giữa**
- Patch 84 (row=6, col=0) - **giữa trái**

**Pattern:** Attention tập trung ở **vùng trung tâm** của ảnh

---

## 🎯 Giải Thích Tại Sao Predictions Khác Nhau

### Hypothesis 1: **Spatial Attention Distribution**

**2.JPG (LIVE):**
- Attention focus vào **biên và góc** ảnh
- Có thể model đang tìm kiếm **artifacts ở biên** (screen edges, print borders)
- Không tìm thấy artifacts → predict LIVE

**1.JPG (SPOOF):**
- Attention focus vào **vùng trung tâm** (khuôn mặt)
- Model có thể phát hiện **artifacts trên khuôn mặt** (texture bất thường, moiré patterns)
- Tìm thấy artifacts → predict SPOOF

### Hypothesis 2: **Feature-Level Differences**

Mặc dù attention statistics tương tự, nhưng:
- **Slot 4** có std cao hơn ở cả 2 ảnh (0.001763 vs 0.001570)
- Slot 4 có max attention cao (0.049403 vs 0.049007)
- Slot này có thể là **discriminative slot** quan trọng

### Hypothesis 3: **Confidence Level**

**2.JPG:**
- Confidence rất cao (94.21%)
- Margin lớn (0.8841)
- → Model **rất chắc chắn** đây là LIVE

**1.JPG:**
- Confidence trung bình (70.19%)
- Margin nhỏ hơn (0.4038)
- → Model **ít chắc chắn hơn**, có thể có **ambiguous features**

---

## 🔬 Phân Tích Sâu Hơn (Cần Kiểm Tra)

### 1. **Visual Inspection Needed:**

Để hiểu rõ hơn, cần xem:
- ✅ Slot attention heatmaps (đã có trong `runs/visualize_slots/`)
- ✅ Grad-CAM heatmaps
- ❓ Ảnh gốc để xác định:
  - 1.JPG có phải là spoof thật không?
  - Loại attack gì? (print, replay, mask?)
  - Có artifacts rõ ràng không?

### 2. **Possible Artifacts in 1.JPG:**

Dựa vào attention pattern (focus vào center), có thể:
- **Moiré patterns** trên khuôn mặt (từ screen replay)
- **Print texture** (từ photo print)
- **Unnatural skin texture**
- **Lighting inconsistencies**
- **Edge artifacts** ở boundary khuôn mặt

### 3. **Why 2.JPG is LIVE:**

Dựa vào attention pattern (focus vào edges), có thể:
- **Không có screen edges** (không phải replay)
- **Không có print borders** (không phải print)
- **Natural lighting** và **skin texture**
- **Consistent depth** (không phải 2D)

---

## 📈 Recommendations để Cải Thiện

### 1. **Tăng Discriminative Power:**

```python
# Trong training, có thể:
# 1. Tăng weight cho spoof class
_C.TRAIN.WEIGHTS = [10.0, 1.0]  # [spoof, live]

# 2. Sử dụng focal loss để focus vào hard examples
_C.TRAIN.FOCAL_LOSS = True
_C.TRAIN.FOCAL_GAMMA = 2.0

# 3. Hard negative mining
```

### 2. **Attention Regularization:**

Để slots học được patterns rõ ràng hơn:
- Thêm **diversity loss** để các slots focus vào vùng khác nhau
- Thêm **sparsity loss** để attention tập trung hơn
- Thêm **contrastive loss** giữa spoof và real slots

### 3. **Data Augmentation:**

Để model robust hơn với edge cases:
- Augment với **different lighting conditions**
- Augment với **different angles**
- Augment với **motion blur** (simulate real-world conditions)

---

## 🎨 Visualization Analysis

### Cần Xem Slot Attention Heatmaps:

Từ files đã tạo:
- `runs/visualize_slots/2_slots.png` (LIVE)
- `runs/visualize_slots/1_slots.png` (SPOOF)

**Cần kiểm tra:**
1. **Spoof slots** (slot 0-2) có focus vào artifacts không?
2. **Real slots** (slot 3-4) có focus vào facial features không?
3. Có sự khác biệt rõ ràng giữa 2 ảnh không?

### Cần Xem Grad-CAM:

Chạy:
```bash
# For 2.JPG (LIVE)
python3 visualize_heatmap.py \
    --weights "weights.pt" \
    --image "/data/huyvq/fas/2.JPG" \
    --save_path "runs/analysis"

# For 1.JPG (SPOOF)
python3 visualize_heatmap.py \
    --weights "weights.pt" \
    --image "/data/huyvq/fas/1.JPG" \
    --save_path "runs/analysis"
```

---

## 🏁 Kết Luận

### Tại Sao 2.JPG → LIVE:
1. ✅ **High confidence** (94.21%)
2. ✅ **Attention focus vào edges** - không tìm thấy artifacts
3. ✅ **Large margin** (0.8841) - model rất chắc chắn
4. ✅ Có thể là **genuine live face** với natural features

### Tại Sao 1.JPG → SPOOF:
1. ⚠️ **Medium confidence** (70.19%)
2. ⚠️ **Attention focus vào center** - phát hiện artifacts trên face
3. ⚠️ **Smaller margin** (0.4038) - model ít chắc chắn hơn
4. ⚠️ Có thể có **spoof artifacts** (moiré, texture, lighting)

### Next Steps:
1. 📸 Xem ảnh gốc để verify ground truth
2. 🎨 Xem slot attention heatmaps để hiểu attention patterns
3. 🔥 Xem Grad-CAM để xác định vùng discriminative
4. 🔬 Nếu 1.JPG là false positive, cần retrain với more data
5. 📊 Test trên nhiều ảnh hơn để xác định pattern

---

## 📚 References

- MVP-FAS Paper: [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_Multi-View_Slot_Attention_Using_Paraphrased_Texts_for_Face_Anti-Spoofing_ICCV_2025_paper.html)
- Slot Attention: [NeurIPS 2020](https://arxiv.org/abs/2006.15055)
