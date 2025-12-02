# Đăng ký ảnh y tế biến dạng dựa trên VoxelMorph

Repo này được xây dựng dựa trên code từ [VoxelMorph (dev-tensorflow branch)](https://github.com/voxelmorph/voxelmorph/tree/dev-tensorflow), triển khai đăng ký ảnh y tế biến dạng (scan-to-scan hoặc scan-to-atlas) với tùy chọn sử dụng giám sát bằng phân đoạn. Nó tích hợp phương pháp cốt lõi từ bài báo [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231) — hỗ trợ huấn luyện không giám sát và bán giám sát.

---

## 📄 Giới thiệu & Mục tiêu

VoxelMorph đưa bài toán đăng ký ảnh thành một hàm học được: một mạng CNN nhận cặp ảnh (moving + fixed) và sinh ra **trường biến dạng (warp)** để căn chỉnh chúng. Sau khi huấn luyện, mô hình có thể đăng ký cặp ảnh mới chỉ bằng một lượt warp, nhanh hơn rất nhiều so với các phương pháp tối ưu truyền thống.

Hai chiến lược huấn luyện:

- **Không giám sát (Unsupervised)**: chỉ sử dụng cường độ ảnh (MSE, NCC) + điều chuẩn mượt mà.
- **Bán giám sát (Semi-supervised)**: khi có phân đoạn giải phẫu tại thời điểm huấn luyện, có thể thêm loss dựa trên phân đoạn để cải thiện độ chính xác.

Repo này mở rộng code chính thức để hỗ trợ:

- Huấn luyện với giám sát phân đoạn
- Đánh giá (Dice) trên tập kiểm tra
- Đăng ký cặp ảnh mới
- Trực quan hóa trường biến dạng

---

## 📁 Cấu trúc thư mục

├── scripts/
│ ├── train_seg.py # Huấn luyện mô hình sử dụng dữ liệu phân đoạn
| ├── train.py # Huấn luyện mô hình không sử dụng dữ liệu phân đoạn
| ├── test_seg.py # Đánh giá mô hình trên các cặp ảnh với mô hình sử dụng phân đoạn
│ ├── test.py # Đánh giá mô hình trên các cặp ảnh với mô hình không sử dụng phân đoạn
| ├── register_seg.py # Đăng ký một cặp ảnh, xuất warp và ảnh đã đăng ký với mô hình sử dụng phân đoạn  
│ ├── register.py # Đăng ký một cặp ảnh, xuất warp và ảnh đã đăng ký với mô hình không sử dụng phân đoạn
├── voxelmorph/ # Các file code mô hình chính: mạng neuron, hàm loss ....
├── utils/ # Hàm hỗ trợ (trực quan hóa, I/O, metrics)
├── notebook/
│ ├── visualize.ipynb # File demo đăng ký ảnh và hiển thị kết quả trên các model đã train
├── setup.py
└── README.md

## 🚀 Huấn luyện

Huấn luyện mô hình đăng ký (không giám sát hoặc bán giám sát):

Ví dụ cho mô hình semi-supervised

```bash
python scripts/train_seg.py \
  --img-list path/to/img_list.txt \
  --seg-list path/to/seg_list.txt \
  --model-dir path/to/model.h5
  --nb-labels 4 \
  --epochs 10 \
  --steps-per-epoch 100 \
  --alpha 0.01 \
  --lambda 0.01 \
  --image-loss mse
```

Ví dụ cho mô hình unsupervised

```bash
python scripts/train.py \
  --img-list path/to/img_list.txt \
  --model-dir path/to/model.h5 \
  --epochs 10 \
  --steps-per-epoch 100 \
  --lambda 0.01 \
  --image-loss ncc
```

## ✅ Kiểm tra / Đánh giá

Tính chỉ số dice score và thời gian đăng ký trung bình trên các cặp ảnh test sử dụng ground-truth phân đoạn:

Ví dụ cho mô hình semi-supervised

```bash
python scripts/test_seg.py \
  --model path/to/trained_model.h5 \
  --pairs path/to/pairs.txt \
  --img-suffix _norm.nii.gz \
  --seg-suffix _seg.nii.gz \
  --nb-labels 4
```

Ví dụ cho mô hình unsupervised

```bash
python scripts/test.py \
  --model path/to/trained_model.h5 \
  --pairs path/to/pairs.txt \
  --img-suffix _norm.nii.gz \
  --seg-suffix _seg.nii.gz \
```

In ra kết quả trung bình và độ lệch chuẩn của thời gian đăng ký và dice score

## 🔄 Đăng ký / Inference trên cặp ảnh mới

Đăng ký một cặp ảnh mới với mô hình đã huấn luyện:

Ví dụ cho mô hình semi-supervised

```bash
python scripts/register_seg.py \
  --moving path/to/moving.nii.gz \
  --fixed path/to/fixed.nii.gz \
  --model path/to/trained_model.h5 \
  --moved output/moved.nii.gz \
  --warp output/warp.nii.gz \
  --nb-labels 4
```

Ví dụ cho mô hình unsupervised

```bash
python scripts/register.py \
  --moving path/to/moving.nii.gz \
  --fixed path/to/fixed.nii.gz \
  --model path/to/trained_model.h5 \
  --moved output/moved.nii.gz \
  --warp output/warp.nii.gz \
```

## 🧩 Dữ liệu sử dụng

Trong repo này, các thí nghiệm được thực hiện trên tập dữ liệu OASIS (2D slices) — một tập con từ [**Neurite-OASIS dataset**](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md), được phát hành bởi nhóm nghiên cứu MIT CSAIL.

### 📚 Mô tả dữ liệu

- Dữ liệu gốc: MRI não người cao tuổi từ bộ **OASIS** (Open Access Series of Imaging Studies).
- Phiên bản sử dụng: các lát cắt 2D (slice) được trích từ thể tích 3D, có định dạng `.nii.gz`.
- Mỗi đối tượng gồm hai loại file:
  - `slice_norm.nii.gz`: ảnh cường độ đã được chuẩn hóa (đầu vào cho mô hình).
  - `slice_seg4.nii.gz`: ảnh phân đoạn gồm 4 vùng giải phẫu (dùng cho supervision và đánh giá).

### ⚙️ Chia tập Train/Test

- Tổng cộng: **414 ảnh** từ các đối tượng khác nhau.
- Chia thành:
  - **Train set**: 80% dùng để huấn luyện mô hình.
  - **Test set**: 20% dùng để đánh giá mô hình (Dice, thời gian đăng ký,…).
