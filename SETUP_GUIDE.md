# PlantVillage Conditional DDIM - Setup Guide

Hướng dẫn chi tiết để train conditional DDIM trên dataset PlantVillage với Kaggle.

## Tổng quan

Dự án này đã được cấu hình sẵn để train mô hình conditional DDIM trên dataset PlantVillage với các tính năng:
- ✅ Conditional generation với 38 lớp bệnh cây trồng
- ✅ Tự động split dataset 70/15/15 (train/val/test)
- ✅ Tích hợp Wandb để tracking experiments
- ✅ Tối ưu hóa cho môi trường Kaggle

## Cách sử dụng trên Kaggle

### Bước 1: Tạo Kaggle Notebook mới

1. Truy cập https://www.kaggle.com/
2. Tạo notebook mới
3. Bật GPU (Settings → Accelerator → GPU T4 x2)

### Bước 2: Thêm Dataset

Trong phần "Add data" của notebook:
- Tìm kiếm "PlantVillage" hoặc "plantdisease"
- Thêm dataset có path: `/kaggle/input/plantdisease/PlantVillage`

### Bước 3: Clone Repository

```python
# Cell 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/ddim.git
%cd ddim
```

### Bước 4: Cài đặt Dependencies

```python
# Cell 2: Install
!pip install -q -r requirements.txt
```

### Bước 5: (Tùy chọn) Cấu hình Wandb

**Cách 1: Sử dụng Kaggle Secrets (Khuyến nghị)**
1. Vào Settings của Kaggle
2. Thêm Secret với tên `WANDB_API_KEY` và giá trị là API key từ https://wandb.ai/authorize
3. Enable "Make this secret available to me"

**Cách 2: Login thủ công**
```python
# Cell 3: Wandb login (nếu không dùng secrets)
import wandb
wandb.login()  # Nhập API key khi được yêu cầu
```

### Bước 6: Chạy Training

**Cách đơn giản nhất:**
```bash
# Cell 4: Train
!bash train_kaggle.sh
```

**Hoặc chạy trực tiếp:**
```bash
!python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run \
    --ni \
    --exp /kaggle/working/exp
```

## Tùy chỉnh Cấu hình

### Thay đổi Batch Size (nếu bị GPU OOM)

Chỉnh sửa `configs/plantvillage.yml`:
```yaml
training:
  batch_size: 32  # Giảm từ 64 xuống 32 hoặc 16
```

### Thay đổi Image Size

```yaml
data:
  image_size: 32  # Giảm từ 64 xuống 32 để tiết kiệm memory
```

### Tắt Wandb

```yaml
wandb:
  enabled: False
```

### Thay đổi Learning Rate

```yaml
optim:
  lr: 0.0001  # Giảm từ 0.0002
```

## Kiểm tra Trước Khi Train

Test dataset và config:
```bash
!python test_plantvillage.py --data_root /kaggle/input/plantdisease/PlantVillage
```

Test config mà không cần dataset:
```bash
!python test_plantvillage.py --skip_dataset
```

## Cấu trúc Dataset Yêu cầu

```
/kaggle/input/plantdisease/PlantVillage/
├── Pepper__bell___Bacterial_spot/
│   ├── ảnh1.jpg
│   ├── ảnh2.jpg
│   └── ...
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
└── ... (38 classes tổng cộng)
```

## Theo dõi Training

### Xem Tensorboard
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/exp/tensorboard/plantvillage_run
```

### Xem Wandb
Truy cập: https://wandb.ai/YOUR_USERNAME/ddim-plantvillage

### Xem Logs
```bash
!tail -f /kaggle/working/exp/logs/plantvillage_run/stdout.txt
```

## Checkpoints

Checkpoints được lưu tại:
- `/kaggle/working/exp/logs/plantvillage_run/ckpt.pth` (checkpoint mới nhất)
- `/kaggle/working/exp/logs/plantvillage_run/ckpt_{step}.pth` (checkpoints theo step)

## Resume Training

Nếu training bị gián đoạn:
```bash
!python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run \
    --resume_training \
    --ni \
    --exp /kaggle/working/exp
```

## Sampling (Tạo ảnh mới)

Sau khi train xong:
```bash
!python main.py \
    --config plantvillage.yml \
    --doc plantvillage_run \
    --sample \
    --fid \
    --ni \
    --exp /kaggle/working/exp
```

## Thông số Training Mặc định

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| Image size | 64x64 | Kích thước ảnh |
| Batch size | 64 | Số ảnh mỗi batch |
| Learning rate | 0.0002 | Tốc độ học |
| Num classes | 38 | Số lớp bệnh |
| Train split | 70% | Tỉ lệ dữ liệu train |
| Val split | 15% | Tỉ lệ dữ liệu validation |
| Test split | 15% | Tỉ lệ dữ liệu test |
| Epochs | 1000 | Số epoch tối đa |
| Snapshot freq | 5000 | Lưu checkpoint mỗi 5000 steps |

## Xử lý Lỗi Thường gặp

### GPU Out of Memory
```yaml
# Giảm batch size trong configs/plantvillage.yml
training:
  batch_size: 32  # hoặc 16
```

### Dataset không tìm thấy
```bash
# Kiểm tra path dataset
!ls /kaggle/input/plantdisease/
```

### Wandb không hoạt động
```yaml
# Tắt wandb trong config
wandb:
  enabled: False
```

### Import Error
```bash
# Cài lại dependencies
!pip install --force-reinstall -r requirements.txt
```

## Files Quan trọng

- `configs/plantvillage.yml` - Cấu hình training
- `train_kaggle.sh` - Script training cho Kaggle
- `test_plantvillage.py` - Test dataset và config
- `datasets/plantvillage.py` - Dataset loader
- `models/diffusion.py` - Model architecture (conditional)
- `runners/diffusion.py` - Training loop với wandb
- `requirements.txt` - Dependencies

## Lưu ý Quan trọng

1. **GPU**: Nhớ bật GPU trong Kaggle notebook settings
2. **Time limit**: Kaggle có giới hạn thời gian, nên chia nhỏ training và sử dụng resume
3. **Checkpoints**: Lưu checkpoints thường xuyên để tránh mất dữ liệu
4. **Wandb**: Rất hữu ích để theo dõi training từ xa

## Hỗ trợ

Nếu gặp vấn đề:
1. Chạy `test_plantvillage.py` để kiểm tra setup
2. Xem logs trong `/kaggle/working/exp/logs/plantvillage_run/stdout.txt`
3. Kiểm tra Wandb dashboard để xem training metrics

## Citation

```bibtex
@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2010.02502},
  year={2020}
}
```
