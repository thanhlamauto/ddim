# Hướng dẫn Resume Training với Tmux

## Cách 1: Sử dụng Python Script (Khuyến nghị)

```bash
cd /workspace/ddim
python resume_tmux.py
```

### Tùy chọn:
```bash
# Đặt target steps khác
python resume_tmux.py --target_steps 200000

# Tạo session với tên khác
python resume_tmux.py --session my_training

# Tạo session nhưng không attach
python resume_tmux.py --no_attach
```

## Cách 2: Sử dụng Bash Script

```bash
cd /workspace/ddim
chmod +x resume_in_tmux.sh
./resume_in_tmux.sh
```

## Cách 3: Lệnh thủ công

```bash
# Tạo tmux session
tmux new-session -s plantvillage_training

# Trong tmux, chạy training
cd /workspace/ddim
python resume_training.py

# Detach khỏi session: Ctrl+B rồi nhấn D
```

---

## Các lệnh Tmux quan trọng

### Quản lý Session

```bash
# Xem danh sách sessions
tmux ls

# Attach vào session
tmux attach -t plantvillage_training

# Detach khỏi session (trong tmux)
Ctrl+B, rồi nhấn D

# Kill session
tmux kill-session -t plantvillage_training

# Rename session
tmux rename-session -t old_name new_name
```

### Trong Tmux Session

```bash
# Detach
Ctrl+B, D

# Scroll/xem log
Ctrl+B, [
# (dùng mũi tên hoặc Page Up/Down để scroll, nhấn Q để thoát)

# Tạo window mới
Ctrl+B, C

# Chuyển window
Ctrl+B, N (next)
Ctrl+B, P (previous)
Ctrl+B, 0-9 (số window)

# Split pane ngang
Ctrl+B, "

# Split pane dọc
Ctrl+B, %

# Chuyển giữa các pane
Ctrl+B, arrow keys
```

---

## Theo dõi Training

### Xem log real-time (trong tmux)
Training logs sẽ hiện trực tiếp trong tmux session

### Xem log từ file
```bash
# Xem log cuối
tail -f /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt

# Xem 50 dòng cuối
tail -n 50 /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt

# Xem FID scores
grep "Val FID" /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt
```

### Kiểm tra tiến độ
```bash
# Xem step hiện tại
tail /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt | grep "Step"

# Xem checkpoint
ls -lh /workspace/ddim/exp/logs/plantvillage_100steps/*.pth
```

---

## Workflow khuyến nghị

1. **Start training trong tmux:**
   ```bash
   cd /workspace/ddim
   python resume_tmux.py
   ```

2. **Detach để training chạy background:**
   - Nhấn `Ctrl+B`, sau đó nhấn `D`

3. **Đóng terminal/disconnect SSH:**
   - Training vẫn tiếp tục chạy trong tmux

4. **Quay lại kiểm tra:**
   ```bash
   tmux attach -t plantvillage_training
   ```

5. **Xem log nhanh (không cần attach):**
   ```bash
   tail -f /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt
   ```

---

## Troubleshooting

### Session không attach được
```bash
# List all sessions
tmux ls

# Force attach
tmux attach -t plantvillage_training -d
```

### Muốn dừng training
```bash
# Attach vào session
tmux attach -t plantvillage_training

# Trong session, nhấn Ctrl+C để dừng training

# Hoặc kill session từ bên ngoài
tmux kill-session -t plantvillage_training
```

### Training bị lỗi
```bash
# Xem log
tail -100 /workspace/ddim/exp/logs/plantvillage_100steps/stdout.txt

# Attach vào session để xem trực tiếp
tmux attach -t plantvillage_training
```

### Tmux chưa cài đặt
```bash
apt-get update
apt-get install -y tmux
```

---

## Thông tin Training hiện tại

- **Current step:** 100,000
- **Target step:** 160,000
- **Remaining:** 60,000 steps
- **Current FID:** 64.19
- **Checkpoint:** `/workspace/ddim/exp/logs/plantvillage_100steps/ckpt.pth`

---

## Quick Reference Card

| Tác vụ | Lệnh |
|--------|------|
| Start training | `python resume_tmux.py` |
| Detach | `Ctrl+B, D` |
| Attach lại | `tmux attach -t plantvillage_training` |
| Xem log | `tail -f .../stdout.txt` |
| Kill session | `tmux kill-session -t plantvillage_training` |
| List sessions | `tmux ls` |
| Scroll trong tmux | `Ctrl+B, [` |
