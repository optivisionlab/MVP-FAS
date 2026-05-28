# Báo cáo phân tích Source Code MVP-FAS

> Phạm vi: toàn bộ mã nguồn trong [/data/fas/solution/MVP-FAS](.) gồm `train.py`, `test.py`, `infer.py`, `api.py`, `app.py`, `loaders/`, `models/`, `losses/`, `utils/`, `configs/`.
>
> Mỗi mục bên dưới gồm: **vị trí**, **mô tả lỗi/hạn chế**, **mức độ ảnh hưởng**, và **đề xuất sửa**.

---

## 1. Các lỗi nghiêm trọng (BUG — phải sửa)

### 1.1. `api.py` — load nhầm model cho `net2` (mọi inference của "face_crop branch" đang chạy bằng `net1`)
- **Vị trí:** [api.py:80-84](api.py#L80-L84)
- **Code:**
  ```python
  net2 = get_network(cfg=cfg, device=device, backbone=os.getenv("BACKBONE", default="ViT-B/16"))
  net2 = load_checkpoint(net1, weight_path=os.getenv("WEIGHT_FACE", default="best.pt"))
  net2.to(device)
  ```
- **Lỗi:** Truyền `net1` thay vì `net2` vào `load_checkpoint`. Vì `load_checkpoint` mutate state_dict in-place và trả lại đối tượng được truyền vào, nên biến `net2` cuối cùng tham chiếu đến **chính `net1`** (đã bị ghi đè state_dict bằng `WEIGHT_FACE`).
- **Hậu quả:** Khi API gọi `infer_model(net2, ...)` cho luồng "crop face", thực chất vẫn dùng `net1` — phá vỡ hoàn toàn ý đồ 2-stage. Đồng thời `net1` bị overwrite weights của face-crop nên cả 2 stage đều sai.
- **Sửa:**
  ```python
  net2 = load_checkpoint(net2, weight_path=os.getenv("WEIGHT_FACE", default="best.pt"))
  ```

### 1.2. `api.py` — biến `prob` có thể chưa được khởi tạo (UnboundLocalError)
- **Vị trí:** [api.py:46-58](api.py#L46-L58)
- **Lỗi:** Trong nhánh `if prob1[0] > threshold:` chỉ có `if YOLO_Det:` gán `prob`. Nếu `YOLO_Det=False` và `prob1[0] > threshold`, biến `prob` không bao giờ được gán → `UnboundLocalError` khi `return`.
- **Sửa:** Khởi tạo `prob = prob1[0]` trước khối `if YOLO_Det`, hoặc thêm `else: prob = prob1[0]`.

### 1.3. `train.py` — default `--num_epochs` là chuỗi, type=int sẽ broken khi không truyền
- **Vị trí:** [train.py:80](train.py#L80)
- **Code:**
  ```python
  parser.add_argument('--num_epochs', type=int, default="number of epochs")
  ```
- **Lỗi:** `default="number of epochs"` là string, trong khi `type=int`. Argparse không convert default → `args.num_epochs` sẽ là chuỗi nếu user không truyền flag. Sau đó:
  ```python
  cfg['TRAIN']['EPOCH'] = args.num_epochs  # string
  max_epoch = cfg.TRAIN.EPOCH              # string
  for epoch in range(start_epoch, max_epoch):  # TypeError
  ```
- **Sửa:** `default=100, help="number of epochs"`.

### 1.4. `train.py` / `test.py` — sai cách dùng `type=bool` với argparse
- **Vị trí:** [train.py:66](train.py#L66), [test.py:73](test.py#L73)
- **Code:**
  ```python
  parser.add_argument("--periodically", type=bool, default=False, ...)
  parser.add_argument("--resume", type=bool, default=True, ...)
  ```
- **Lỗi:** `type=bool` trong argparse tương đương `bool(str)`, mọi chuỗi non-empty đều thành `True`. Truyền `--periodically False` vẫn → `True`.
- **Sửa:** Dùng `action='store_true'` (đã làm cho `--pretrained`, `--is_physical`), hoặc `argparse.BooleanOptionalAction` (Python ≥3.9).

### 1.5. `models/MVP_FAS.py` — `random.seed()` bị gọi mỗi forward, triệt tiêu randomness
- **Vị trí:** [models/MVP_FAS.py:144](models/MVP_FAS.py#L144)
- **Code:**
  ```python
  def forward(self, input, target=None):
      ...
      random.seed(self.cfg.MODEL.SEED)
  ```
- **Lỗi:** Seed lại module `random` **trong mọi step training/inference**. Điều này phá toàn bộ randomness (data aug, sampling) — tất cả batch sau forward đầu sẽ thấy cùng chuỗi số ngẫu nhiên.
- **Sửa:** Bỏ dòng này, seed chỉ một lần ở đầu chương trình.

### 1.6. `models/MVP_FAS.py` — `.squeeze()` không an toàn khi batch=1
- **Vị trí:** [models/MVP_FAS.py:136](models/MVP_FAS.py#L136)
- **Code:**
  ```python
  patch_activations = patch_activations.squeeze()
  ```
- **Lỗi:** Khi `B=1`, output `[1,1,196]` sẽ bị squeeze về `[196]` thay vì `[1,196]` → bước nhân `patch * patch_activations.unsqueeze(-1)` sẽ broadcast sai shape.
- **Sửa:** `patch_activations = patch_activations.squeeze(1)`.

### 1.7. `models/MVP_FAS.py` — re-encode text prompts mỗi forward (lãng phí cực lớn)
- **Vị trí:** [models/MVP_FAS.py:148-153](models/MVP_FAS.py#L148-L153)
- **Lỗi:** Tokenize + chạy `model.encode_text(...)` cho `spoof_templates`/`real_templates` ở **mọi iteration**. Vì prompts là **hằng**, đây là phí phạm tài nguyên (đặc biệt khi text encoder bị freeze cố định) và còn rò rỉ memory khi training dài.
- **Sửa:** Tính 1 lần trong `__init__` rồi cache, hoặc cache theo eval/train mode.

### 1.8. `infer.py` — `net_face_crop` được dùng dù không load weights → kết quả ngẫu nhiên
- **Vị trí:** [infer.py:206-217](infer.py#L206-L217), [infer.py:264-275](infer.py#L264-L275)
- **Lỗi:**
  - Dòng 206 luôn khởi tạo `net_face_crop` với weight ngẫu nhiên.
  - Chỉ khi `args.MVP_FAS_FACE_CROP=True` mới `load_checkpoint`.
  - Nhưng khi `args.YOLO_FACE=True` (không kèm `MVP_FAS_FACE_CROP`), code vẫn gọi `infer_model(net_face_crop, ...)` ở dòng 268 → suy luận bằng model chưa train.
- **Sửa:** Bỏ flag riêng `MVP_FAS_FACE_CROP`, hoặc bắt buộc load weight khi `YOLO_FACE=True`; thêm assert.

### 1.9. `infer.py` — không xử lý `img_crop` là `None` từ YOLO
- **Vị trí:** [infer.py:267-275](infer.py#L267-L275)
- **Lỗi:** `crop_face_with_expand` có thể trả về `(None, None)` khi không detect được face. Hàm `infer_model` sau đó gọi `transform(img=None)` → crash.
- **Sửa:** Kiểm tra `if img_crop is None: ... continue/handle` (đã làm trong `api.py:49-51`, nhưng `infer.py` thì không).

### 1.10. `utils/loggers/wandb_logger.py` — import sai package (rò rỉ PaddleOCR)
- **Vị trí:** [utils/loggers/wandb_logger.py:3](utils/loggers/wandb_logger.py#L3)
- **Code:**
  ```python
  from ppocr.utils.logging import get_logger
  ```
- **Lỗi:** `ppocr` không có trong project. Đây là tàn dư copy từ PaddleOCR. Mọi import của `Loggers`/`WandbLogger` sẽ throw `ModuleNotFoundError`.
- **Sửa:** `from utils.logging import get_logger`.
- **Liên quan:** [utils/loggers/wandb_logger.py:72](utils/loggers/wandb_logger.py#L72) còn dùng `model_path = os.path.join(self.save_dir, prefix + ".pdparams")` — chỉ Paddle dùng `.pdparams`. Với PyTorch phải đổi thành `.pt`/`.pth`.

### 1.11. `MCIO.py` / `SFW.py` — `Image_Saturation` HSV chạy trên RGB → augmentation bị biến thành R↔B swap
- **Vị trí:** [loaders/MCIO.py:48-53](loaders/MCIO.py#L48-L53), [loaders/SFW.py:48-53](loaders/SFW.py#L48-L53), [loaders/MCIO.py:170](loaders/MCIO.py#L170), [loaders/SFW.py:113](loaders/SFW.py#L113)
- **Phân tích chính xác:**
  - `cv2.imread` → bytes BGR.
  - `cvtColor(COLOR_RGB2BGR)` thực ra chỉ là **swap channel 0↔2** (cùng phép biến đổi với `COLOR_BGR2RGB`). Sau đó data ở dạng RGB. Naming sai nhưng kết quả đúng.
  - **Đường truyền chính (model input) không bị ảnh hưởng** — cả train và val đều thấy RGB chuẩn.
  - Nhưng `Image_Saturation` lại gọi `COLOR_BGR2HSV` trên data đang là RGB → OpenCV coi nhầm R là B → công thức tính Hue chọn sai nhánh. Khi `S` nhân `uniform(0.8, 1.2)` rồi `HSV2BGR` ngược lại, kết quả tương đương **swap R↔B ngẫu nhiên** với cường độ tỉ lệ `|S-1|`. Ví dụ: pixel đỏ `[255,0,0]` với `S×0.8` → output `[51,51,255]` (xanh dương nhạt), với `S=1.0` thì không đổi.
- **Phạm vi tác động:**
  - **Val/test:** không gọi `Image_Saturation` (chỉ khi `is_train=True`) → input model y hệt, **không bị bug**.
  - **Train:** 50% sample qua `Image_Saturation`, augmentation saturation thực chất là "saturation + random R↔B swap".
- **Hệ quả:**
  - Inference với checkpoint cũ + code đã fix → **metric không thay đổi** (val pipeline giống y).
  - Training mới sau fix → augmentation đúng nghĩa, có thể train tốt hơn nhưng không bắt buộc retrain.
- **Sửa:** Thống nhất với `custom_dataset.py`:
  - `cvtColor(..., COLOR_BGR2RGB)` ở `__getitem__` (đặt tên đúng).
  - Trong `Image_Saturation`: `COLOR_RGB2HSV` / `COLOR_HSV2RGB`.

### 1.12. `utils/metric.py` — accuracy dùng threshold cứng 0.5
- **Vị trí:** [utils/metric.py:115](utils/metric.py#L115)
- **Code:**
  ```python
  acc = ((probs > 0.5) == labels).mean()
  ```
- **Lỗi:** Trong khi `hter`, `eer` đều được tính tại threshold tối ưu, riêng `acc` lại dùng 0.5 cứng → log misleading khi dataset imbalance hoặc threshold lệch xa 0.5 (trong logs thực tế threshold thường ~0.1–0.15).
- **Sửa:** dùng `acc_thr` (đã có) hoặc thêm cả hai (acc@0.5 và acc@optimal_thr).

### 1.13. `test.py` / `models/CLIP/clip.py` — `torch.load` không truyền `weights_only` & `map_location`
- **Vị trí:** [test.py:23](test.py#L23), [models/CLIP/clip.py:137](models/CLIP/clip.py#L137), [models/make_network.py:11,23](models/make_network.py#L11)
- **Lỗi:** Từ PyTorch 2.4+ mặc định `weights_only=True` (kèm DeprecationWarning); nếu checkpoint chứa object ngoài tensor (vd `numpy` scalars) sẽ load fail. Ngoài ra không có `map_location` ⇒ load checkpoint train trên GPU x bị bind vào GPU x kể cả khi chạy CPU.
- **Sửa:** `torch.load(path, map_location='cpu', weights_only=False)`.

### 1.14. `train.py` — resume không cập nhật `start_epoch`
- **Vị trí:** [train.py:171](train.py#L171)
- **Code:**
  ```python
  if resume == True: net, optimizer, last_epoch = set_pretrained_setting(net, optimizer, checkpoint)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
  ```
- **Lỗi:** `last_epoch` được pass đúng vào scheduler nhưng **`start_epoch` vẫn = 0**. Vòng `for epoch in range(start_epoch, max_epoch)` chạy lại từ 0. Resume mất tác dụng.
- **Sửa:** `start_epoch = last_epoch + 1`.

---

## 2. Lỗi vừa / Code smell quan trọng

### 2.1. `train.py` — `count = len(os.listdir(...)) + 1` để đặt tên run
- **Vị trí:** [train.py:107](train.py#L107), [test.py:109](test.py#L109), [infer.py:187](infer.py#L187)
- **Hạn chế:** Race-condition (2 train đồng thời = trùng folder), tăng theo bất kỳ file rác nào, vô tình đếm `.DS_Store`/checkpoint hỏng.
- **Sửa:** Dùng timestamp `datetime.now().strftime('%Y%m%d_%H%M%S')`.

### 2.2. `train.py` — log LR sai một epoch
- **Vị trí:** [train.py:232](train.py#L232), [train.py:360-361](train.py#L360-L361)
- **Hạn chế:** `lr` chỉ được cập nhật sau `scheduler.step()` ở cuối epoch, nhưng `writer.add_scalar("train/LR", lr, epoch + 1)` được gọi đầu epoch. Giá trị LR log ra cho epoch n thực ra là LR đã apply ở epoch n-1.
- **Sửa:** `optimizer.param_groups[0]['lr']`, hoặc đẩy phần log LR xuống sau `scheduler.step()`.

### 2.3. `train.py` — `last_ckp_path` thiếu underscore
- **Vị trí:** [train.py:350](train.py#L350)
- **Code:**
  ```python
  last_ckp_path = os.path.join(save_folder, 'weights', model_name + '_' + save_name + 'last_ckpt.pt')
  ```
- **Hạn chế:** `save_name + 'last_ckpt.pt'` → `RN50last_ckpt.pt`. Cosmetic nhưng dễ confuse khi tìm file.

### 2.4. `set_seed` không deterministic và không seed `torch.cuda.manual_seed` cho thiết bị đang dùng
- **Vị trí:** [train.py:21-28](train.py#L21-L28)
- **Hạn chế:** `deterministic=False` luôn được truyền → kết hợp với `cudnn.benchmark` mặc định ⇒ reproducibility không được đảm bảo.

### 2.5. `DataLoader` được tạo lại mỗi epoch
- **Vị trí:** [train.py:180-186](train.py#L180-L186)
- **Hạn chế:** Tạo lại loader trong vòng `for epoch` → chi phí setup `num_workers=16` lặp đi lặp lại. Nên tạo loader 1 lần trước vòng `for epoch`.

### 2.6. Class weight `[5.0, 1.0]` cứng cho mọi setting
- **Vị trí:** [configs/cfg.py:41](configs/cfg.py#L41)
- **Hạn chế:** Comment chỉ rõ `[spoof, live]` nhưng label encoding trong dataset là `Is_real` (0=spoof, 1=live). Tỷ trọng cứng `[5,1]` ép class 0 (spoof) nặng hơn × 5, không hợp lý với mọi dataset (MSU 280 vs VFT 35000 — chênh lệch tỉ lệ rất khác).
- **Sửa:** Tính class weight tự động từ `train_df`, hoặc đưa thành CLI flag.

### 2.7. Duplicate code giữa `infer.py` và `utils/inference.py`
- **Vị trí:** [infer.py:21-98](infer.py#L21-L98) ≡ [utils/inference.py:7-84](utils/inference.py#L7-L84)
- **Hạn chế:** 2 bản copy `crop_face_with_expand` và `infer_model`. Sửa 1 chỗ rất dễ quên chỗ kia (đã thấy `RemoveBlackBorders` có ở `utils/inference.py:94` mà ở `infer.py` lại comment-out).
- **Sửa:** Import từ `utils.inference`.

### 2.8. `infer.py` — mismatch length giữa `preds_score` và `targets_score`
- **Vị trí:** [infer.py:241-296](infer.py#L241-L296)
- **Hạn chế:**
  - Nhánh `YOLO=True` append `prob2[0]`.
  - Nhánh `YOLO_FACE=True` lại append `prob3[0]`.
  - Khi cả hai cùng bật, append 2 lần ⇒ length lệch so với `targets_score` (chỉ append 1 lần ngoài cùng). Khi tính EER/HTER cuối, hai mảng khác độ dài → exception.

### 2.9. `test.py` — val loader `shuffle=True`
- **Vị trí:** [test.py:149](test.py#L149)
- **Hạn chế:** Test set không nên shuffle (gây khó đối chiếu logs, không deterministic).

### 2.10. `train.py` — kiểu dữ liệu loss không nhất quán (numpy/tensor lẫn lộn)
- **Vị trí:** [train.py:270-275](train.py#L270-L275)
- **Hạn chế:** `val_sim_loss = val_CE_loss(...).cpu().numpy()` là `np.ndarray`. `val_loss = (val_sim_loss * Sim_alpha) + (val_patch_alignment_loss * Beta)` → numpy * scalar + tensor * scalar. Kết quả pha trộn kiểu, `.item()` gọi trên cả numpy và tensor. Nên giữ tensor đến hết và `.item()` đồng nhất.

### 2.11. `models/MVP_FAS.py` — `spoof_text_features` build nhưng không dùng (head=`cls`)
- **Vị trí:** [models/MVP_FAS.py:160-161, 182-194](models/MVP_FAS.py#L160-L194)
- **Hạn chế:** `spoof_text_features` chỉ dùng cho `head_type='sim'` (đang không bật) — vẫn được build mọi forward.

### 2.12. `loaders/custom_dataset.py` — xác suất aug bằng `elif` chuỗi khó đọc
- **Vị trí:** [loaders/custom_dataset.py:197-266](loaders/custom_dataset.py#L197-L266)
- **Hạn chế:** Các nhánh `elif prob_value < cfg.TRAIN.X` chỉ chạy đúng khi các threshold tăng dần. Cấu hình hiện tại:
  - Real: 0.1 → 0.3 → 0.5 (Moire→Lux→CeilingLight) → tổng 50% chance áp 1 trong 3 aug.
  - Spoof: 0.2 → 0.4 (Lux → CeilingLight) → 40%.
  - Comment ghi "20%: Lux → real / 20%: CeilingLight(real)" nhưng do `elif`, thực tế CeilingLight là probability `0.5 - 0.3 = 0.2` — chỉ đúng khi user biết cấu trúc này. Rất dễ tinh chỉnh sai.
- **Sửa:** Dùng `np.random.choice` với xác suất rõ ràng; hoặc đặt tên config rõ "until_threshold_x".

### 2.13. `losses/make_losses.py` — typo trong tên hàm public
- **Vị trí:** [losses/make_losses.py:4](losses/make_losses.py#L4)
- **Code:** `def get_loss_fucntion(...)` (thiếu chữ "n"). Các caller `train.py:173-174`, `test.py:144` đều phải dùng tên typo.

### 2.14. `models/MVP_FAS.py` — Class tên `mspt` không khớp `--model MVP_FAS`
- **Vị trí:** [models/MVP_FAS.py:68](models/MVP_FAS.py#L68)
- **Hạn chế:** Tên class viết thường, không PEP8, không khớp tên CLI.

### 2.15. `api.py` — không validate env, fallback im lặng tới `"best.pt"`
- **Vị trí:** [api.py:74-88](api.py#L74-L88)
- **Hạn chế:** Nếu thiếu env var sẽ default về `"best.pt"` ở thư mục hiện hành — silent failure khi không tồn tại file. Không log warning khi env vars rỗng.

### 2.16. `api.py` — field `prob` trong response gây hiểu lầm
- **Vị trí:** [api.py:63](api.py#L63)
- **Hạn chế:** `prob` là xác suất "live", nhưng trả về `"prob": "{:.4f}".format((1-prob)*100)` — số này thực ra là **spoof probability * 100**.
- **Sửa:** Đổi tên `"spoof_score"` / `"live_score"` cho rõ.

### 2.17. `app.py` — API key fallback hard-coded "sherlock"
- **Vị trí:** [app.py:23](app.py#L23)
- **Hạn chế:** Bảo mật: nếu deploy mà quên set env `token`, mọi request gửi với header `token=sherlock` sẽ qua được.
- **Sửa:** Bắt buộc env phải tồn tại; nếu không, raise tại startup.

### 2.18. `models/MVP_FAS.py` — `_freeze_stages` logic dễ nhầm
- **Vị trí:** [models/MVP_FAS.py:93-113](models/MVP_FAS.py#L93-L113)
- **Hạn chế:**
  - `exclude_key=['visual','learnable']` → mọi param có "visual" HOẶC "learnable" trong tên đều requires_grad=True. CLIP có nhiều layer trong text encoder cũng có substring khớp ⇒ freeze map có thể không như ý.
  - In `Finetune layer in backbone: ...` cho mọi layer match — log spam.
- **Sửa:** Liệt kê tên prefix rõ ràng, dùng `n.startswith(...)`.

### 2.19. `script.sh` — vòng for không thực sự sweep
- **Vị trí:** [script.sh:123-143](script.sh#L123-L143)
- **Hạn chế:** Vòng `for char in I M C O` chỉ khác nhau ở weight init (`MVP_FAS_P1_$char.pth`), trainset/valset giữ nguyên. 4 lần train có thể có chủ ý (fine-tune từ 4 checkpoint pretrained khác nhau) nhưng comment không nêu rõ.

### 2.20. `train.py` — `best_val_loss` chưa từng được cập nhật nhưng vẫn lưu vào ckpt
- **Vị trí:** [train.py:143-144](train.py#L143-L144), [train.py:344-356](train.py#L344-L356)
- **Hạn chế:** `best_val_loss = np.inf` rồi không bao giờ gán mới; vẫn ghi vào checkpoint `'performance': best_val_loss` (luôn `inf`).

---

## 3. Lỗi nhỏ / Inconsistency

| # | Vị trí | Vấn đề |
|---|---|---|
| 3.1 | [configs/cfg.py:19](configs/cfg.py#L19) | `_C.DATASET.PATH.ROOT = 'D:/Anti_spoofing/dataset'` — đường dẫn Windows hardcode, nên là `'./dataset'` hoặc lấy từ CLI |
| 3.2 | [test.py:138](test.py#L138) | `get_network(cfg, net_name=model_name, device=device)` thiếu `args=...`, `backbone=...`. Backbone fallback về default `"ViT-B/16"` không khớp với `--backbone` CLI ⇒ Test dùng backbone khác training |
| 3.3 | [train.py:148](train.py#L148) | `logger_interval = 20` khai báo nhưng không dùng |
| 3.4 | [models/MVP_FAS.py:67](models/MVP_FAS.py#L67) | `details = ['a photo of a', 'an image of a']` khai báo nhưng đoạn dùng đã comment-out (line 145-146) — dead code |
| 3.5 | [models/MVP_FAS.py:75](models/MVP_FAS.py#L75) | `self.head_type = 'cls'` hardcode, không thể override qua config |
| 3.6 | [loaders/MCIO.py:73](loaders/MCIO.py#L73) | `file_paths = frame_0_paths + frame_1_paths` → mỗi sample được nhân đôi. Nếu file `frame1` không tồn tại sẽ crash tại `cv2.imread` (trả None) → `Img.shape` |
| 3.7 | [loaders/MCIO.py:99-107](loaders/MCIO.py#L99-L107) | Parse `name.split('_')` cho domain `'C'` rất giòn với mọi pattern tên file. Không try/except, không validate |
| 3.8 | [utils/metric.py:13-19](utils/metric.py#L13-L19) | `get_threshold` tính `min(probs), max(probs)` nhưng không dùng — dead code |
| 3.9 | [utils/metric.py:127-128](utils/metric.py#L127-L128) | `tpr_filtered = tpr[fpr <= 0.01]` — hardcode FPR=1% (thường gọi TPR@FAR1%) — nên đưa vào config |
| 3.10 | [api.py:25](api.py#L25) | `files[0].size > 0` chỉ check file đầu tiên, không kiểm tra mọi file |
| 3.11 | [infer.py:18](infer.py#L18) | Import `from ultralytics import YOLO` ở top-level — bắt buộc cài ultralytics ngay cả khi chạy chế độ không YOLO |
| 3.12 | [train.py:108,112](train.py#L108) | Tạo folder `train_{count}` xong lại nối thêm `model_name + '_' + save_name` ⇒ thư mục lồng 2 lớp không cần thiết |
| 3.13 | Toàn project | Mix `print` + `logger.info` không thống nhất. Có chỗ `print()` rỗng (vd `loaders/custom_dataset.py:282`, `loaders/MCIO.py:196`) — debug rác |
| 3.14 | [models/CLIP/clip.py:61](models/CLIP/clip.py#L61) | Download CLIP qua `urllib.request.urlopen(url)` — không cấu hình proxy/timeout |
| 3.15 | [utils/utils.py:14](utils/utils.py#L14) | `import pyheif` ở top-level. pyheif không còn được maintain trên Linux (cần libheif headers) — fail-to-import nếu user không cần HEIC |
| 3.16 | `.dockerignore` | Chỉ 81 byte — có thể không exclude `runs/`, `logs/`, `dataset/` ⇒ docker context khổng lồ |
| 3.17 | [requirements.txt:49](requirements.txt#L49) | `win_inet_pton==1.1.0` (Windows-only) trong requirements ⇒ pip install trên Linux vô nghĩa |
| 3.18 | [train.py:52-57](train.py#L52-L57) | Comment hardcode về số ảnh từng dataset trong source code — không liên quan logic |

---

## 4. Hạn chế thiết kế (Design issues)

### 4.1. Không tách rõ Train/Val/Test split của dataset "FAS"/"ALL"
- `get_FAS_dataset` và `get_ALL_dataset` đọc CSV của user rồi `shuffle` với seed cố định. Nhưng không có phương án split per-subject (subject-disjoint). FAS rất dễ leak nếu train/val share subject ⇒ HTER tốt giả tạo.

### 4.2. Không có early-stopping / patience
- `train.py` chạy đủ `EPOCH` epoch dù `val_HTER` không cải thiện. Lãng phí.

### 4.3. Không có gradient clipping
- Với CLIP fine-tune LR=1e-5 và batch=18, gradient có thể spike. Thiếu `torch.nn.utils.clip_grad_norm_` ⇒ rủi ro NaN.

### 4.4. Không AMP / mixed precision
- Backbone là CLIP ViT-B/16 (~86M params). Train batch lớn (64-128) trên FP32 rất chậm; nên `torch.cuda.amp.autocast` + `GradScaler`.

### 4.5. Không DDP / multi-GPU
- `torch.nn.DataParallel(net).cuda()` đã bị comment-out. Single-GPU CLIP rất chậm.

### 4.6. Test pipeline không tái dùng `infer_model`
- `test.py` viết lại logic forward / softmax / metric chứ không gọi `utils.inference.infer_model`. Drift cao.

### 4.7. Visualization & Analysis duplicate, không gắn vào main pipeline
- `visualization/` chứa nhiều file đọc/ghi đường dẫn hardcode (`./runs/results/`), không có CLI rõ ràng. Khi train, `utils/visualization.visualize_attn` đang được `import` nhưng phần gọi đã comment-out trong `slot_attention_PQTK.py:66` ⇒ dead code.

### 4.8. Wandb logger được import nhưng không tích hợp
- `utils/loggers/wandb_logger.py` tồn tại nhưng (a) sai import (`ppocr`), (b) không gọi từ `train.py`. Toàn bộ wandb logic là dead code.

### 4.9. Convention label confuse: `Is_real` vs `is_spoof`
- `MCIO`/`SFW` dataset trả `is_real ∈ {0,1}` (1=real).
- `FAS_Dataset` đọc `is_spoof` từ CSV rồi convert `is_real = int(not is_spoof)`.
- Trong `metric.py`, `probs > 0.5 == labels` giả định prob "real". Nhưng `targets_score = int(not label)` trong `infer.py` (label=is_spoof) ⇒ targets_score=1 khi live.
- Logic đảo dấu rải rác khắp code, mỗi nơi convention một kiểu. Rất dễ sai khi mở rộng.

### 4.10. Thiếu unit tests / type hints / docstring
- Không có thư mục `tests/`, không có CI, không có schema cho config, không có docstring chuẩn.

---

## 5. Bảo mật & Vận hành

| # | Vấn đề | Vị trí |
|---|---|---|
| 5.1 | API key fallback `"sherlock"` hard-coded | [app.py:23](app.py#L23) |
| 5.2 | FastAPI `debug=True` mở debug ⇒ leak stacktrace ra client | [app.py:15](app.py#L15) |
| 5.3 | `download_file_from_urls3` ghi file theo tên user-controlled (path traversal nhẹ qua `urlparse.path`) | [utils/utils.py:60-83](utils/utils.py#L60-L83) |
| 5.4 | Không rate-limit `/vft-fas` | [api.py:102](api.py#L102) |
| 5.5 | Logs ghi cả URL người dùng + filename — cần đảm bảo PII compliance | [api.py:127-131](api.py#L127-L131) |

---

## 6. Tổng kết mức độ ưu tiên fix

1. **P0 (chặn release):** 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 1.10, 1.11, 1.14
2. **P1 (sai metric / behavior):** 1.6, 1.7, 1.9, 1.12, 1.13, 2.6, 2.8, 2.11, 2.20
3. **P2 (chất lượng code / efficiency):** 2.1, 2.2, 2.7, 2.10, 2.13, 4.2, 4.3, 4.4
4. **P3 (cosmetic / cleanup):** 2.3, 2.14, các mục mục 3.*, dead code visualization

---

## 7. Khuyến nghị tiếp theo

- Viết unit test cho `Metric.compute` với input giả lập để khoá đúng convention "label=1=real".
- Tách `text_features` ra cache, bật AMP, gradient clip, early-stop.
- Refactor `infer.py` / `utils/inference.py` thành 1 source of truth.
- Đưa hyperparameter (`SIMILARITY_ALPHA`, `PATCH_ALIGN_BETA`, `TRAIN.WEIGHTS`, threshold YOLO,...) ra CLI/Hydra để track theo run trên TensorBoard/W&B.
- Sửa lại `wandb_logger.py` (import + định dạng checkpoint) hoặc xoá nếu không dùng.
- Subject-disjoint split cho `get_ALL_dataset` để tránh leakage.
- Bổ sung CI tối thiểu: `ruff` + `mypy` + chạy `python -c "import api"` để bắt sớm các import sai như `ppocr`.
