# PlantVillage Conditional DDIM - Changelog

Danh sách các thay đổi để hỗ trợ training conditional DDIM trên PlantVillage dataset.

## Files Mới Được Tạo

### 1. Dataset & Configuration
- **`datasets/plantvillage.py`**
  - Dataset loader cho PlantVillage
  - Tự động split 70/15/15 (train/val/test)
  - Hỗ trợ 38 lớp bệnh cây trồng

- **`configs/plantvillage.yml`**
  - Cấu hình training cho PlantVillage
  - Kích hoạt conditional model
  - Tích hợp Wandb logging

### 2. Scripts & Tools
- **`train_kaggle.sh`**
  - Bash script để chạy training trên Kaggle
  - Tự động setup môi trường
  - Kiểm tra dataset và dependencies

- **`test_plantvillage.py`**
  - Test script để verify dataset loading
  - Kiểm tra config file
  - Test model initialization

- **`requirements.txt`**
  - Dependencies cho Kaggle
  - Bao gồm wandb, PyYAML, tqdm, etc.

### 3. Documentation
- **`README_KAGGLE.md`**
  - Hướng dẫn chi tiết sử dụng trên Kaggle
  - Quick start guide
  - Troubleshooting

- **`SETUP_GUIDE.md`**
  - Hướng dẫn setup chi tiết bằng tiếng Việt
  - Cấu hình và tùy chỉnh
  - Xử lý lỗi thường gặp

## Files Đã Sửa Đổi

### 1. Model Architecture
**`models/diffusion.py`**
- ✅ Thêm conditional mode với class embeddings
- ✅ Thêm parameter `conditional` và `num_classes`
- ✅ Thêm `label_emb` embedding layer cho classes
- ✅ Cập nhật `forward()` để nhận class labels `y`
- ✅ Backward compatible với unconditional models

**Thay đổi:**
```python
# Trước
def forward(self, x, t):
    ...

# Sau
def forward(self, x, t, y=None):
    if self.conditional:
        temb = temb + self.label_emb(y)
    ...
```

### 2. Loss Function
**`functions/losses.py`**
- ✅ Thêm parameter `y` cho class labels
- ✅ Pass labels vào model khi available
- ✅ Backward compatible với unconditional training

**Thay đổi:**
```python
# Trước
def noise_estimation_loss(model, x0, t, e, b, keepdim=False):
    output = model(x, t.float())
    ...

# Sau
def noise_estimation_loss(model, x0, t, e, b, y=None, keepdim=False):
    if y is not None:
        output = model(x, t.float(), y)
    else:
        output = model(x, t.float())
    ...
```

### 3. Training Loop
**`runners/diffusion.py`**
- ✅ Import wandb với fallback nếu không có
- ✅ Initialize wandb trong `train()` method
- ✅ Log metrics lên wandb (loss, epoch, step)
- ✅ Xử lý class labels trong training loop
- ✅ Pass labels vào loss function khi conditional
- ✅ Save checkpoints lên wandb
- ✅ Cleanup wandb khi kết thúc

**Thay đổi chính:**
```python
# Wandb initialization
if use_wandb:
    wandb.init(project=..., ...)

# Conditional training
if is_conditional:
    y = y.to(self.device)
    loss = loss_registry[config.model.type](model, x, t, e, b, y)
else:
    loss = loss_registry[config.model.type](model, x, t, e, b)

# Wandb logging
if use_wandb:
    wandb.log({"loss": loss.item(), ...}, step=step)
```

### 4. Dataset Loader
**`datasets/__init__.py`**
- ✅ Import PlantVillage dataset
- ✅ Thêm PlantVillage case trong `get_dataset()`
- ✅ Hỗ trợ custom `data_root` từ config
- ✅ Tạo train/test datasets với đúng transforms

**Thay đổi:**
```python
from datasets.plantvillage import PlantVillage

elif config.data.dataset == "PlantVillage":
    data_root = getattr(config.data, "data_root", ...)
    dataset = PlantVillage(root=data_root, split='train', ...)
    test_dataset = PlantVillage(root=data_root, split='test', ...)
```

## Tính Năng Mới

### 1. Conditional Generation
- Model có thể generate ảnh dựa trên class label
- 38 lớp bệnh cây trồng từ PlantVillage
- Class embedding được thêm vào timestep embedding

### 2. Wandb Integration
- Automatic logging của loss, epoch, step
- Save checkpoints lên wandb
- Config tracking
- Tag và notes cho experiments

### 3. Flexible Data Split
- Configurable train/val/test ratios
- Reproducible splits với fixed seed
- Automatic class detection

### 4. Kaggle Optimization
- Pre-configured paths cho Kaggle
- Auto environment setup
- GPU detection và optimization
- Checkpoint saving trong Kaggle working directory

## Backward Compatibility

Tất cả thay đổi đều backward compatible:
- Unconditional models vẫn hoạt động bình thường
- Config cũ (CIFAR10, CelebA, LSUN) không bị ảnh hưởng
- Wandb là optional (có thể tắt hoặc không cài)

## Testing

Để test các thay đổi:

```bash
# Test config
python test_plantvillage.py --skip_dataset

# Test dataset (cần có PlantVillage dataset)
python test_plantvillage.py --data_root /path/to/PlantVillage

# Test training (dry run)
python main.py --config plantvillage.yml --doc test --ni --exp /tmp/test
```

## Migration Guide

Để sử dụng với dataset khác:

1. Tạo dataset loader mới theo mẫu `plantvillage.py`
2. Thêm vào `datasets/__init__.py`
3. Tạo config file mới
4. Set `conditional: True` và `num_classes` trong config

## Summary

- ✅ 4 files mới
- ✅ 4 files sửa đổi
- ✅ 100% backward compatible
- ✅ Đầy đủ documentation
- ✅ Test scripts
- ✅ Ready for Kaggle

Giờ bạn chỉ cần:
1. Git clone trên Kaggle
2. Chạy `bash train_kaggle.sh`
3. Theo dõi training trên Wandb
