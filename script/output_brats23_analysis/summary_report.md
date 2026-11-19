# Báo Cáo Phân Tích Tập Dữ Liệu BraTS 2023 GLI TrainingData

- **Đường dẫn dữ liệu:** `/home/cuc.buithi/BRATS/BraTS2023_TrainVal/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
- **Tổng số ca hợp lệ:** **1251 / 1251** ca (thiếu file: 0, lỗi đọc: 0)
- **Số lượng bệnh nhân riêng biệt:** **1133** bệnh nhân (phân bố số lần chụp: `{'1_scan': 1015, '2_scan': 118}`, timepoints: `{'000': 1133, '001': 118}`)
- **Tổng dung lượng trên đĩa (5 file `.nii.gz`/ca):** **12.47 GB** (trung bình 10.21 MB/ca)

## 1. Thông số Không gian & Tính nhất quán NIfTI
- **Kích thước ảnh (Shape):** `{'240x240x155': 1251}`
- **Voxel spacing (mm):** `{'1.0x1.0x1.0': 1251}`
- **Hệ trục (Orientation):** `{'LPS': 1251}`
- **Kiểu dữ liệu mask (`seg`):** `{'float32': 1251}`
- **Các tập giá trị nhãn xuất hiện:** `{'0,1,2,3': 1180, '0,2,3': 37, '0,1,2': 27, '0,2': 6, '0,1,3': 1}`
- **Lỗi không nhất quán nội tại ca:** shape mismatch = 0, spacing mismatch = 0, axcodes mismatch = 0, nhãn lạ ngoài {0,1,2,3} = 0

## 2. Mất cân bằng lớp (Class Imbalance) & Ca thiếu nhãn
| Thành phần | Tổng số voxels | % Toàn volume | % Thể tích não (foreground) |
| :--- | ---: | ---: | ---: |
| Background (0) | 11,048,872,504 | 98.925% | - |
| NCR (1) | 17,896,396 | 0.160% | 0.968% |
| ED (2) | 75,328,509 | 0.674% | 4.076% |
| ET (3) | 26,830,591 | 0.240% | 1.452% |
| WT (1+2+3) | 120,055,496 | 1.075% | 6.496% |
| TC (1+3) | 44,726,987 | 0.400% | 2.420% |

- **Số ca không có Whole Tumor (WT = 0):** 0
- **Số ca thiếu Enhancing Tumor (ET = 0):** **33** (2.64%)
- **Số ca thiếu Necrotic Core (NCR = 0):** **43** (3.44%)
- **Số ca thiếu Edema (ED = 0):** **1** (0.08%)

## 3. Phân bố Thể tích (cm³)
| Vùng | Mean ± Std | Median (IQR: P25 - P75) | Min - Max |
| :--- | ---: | ---: | ---: |
| Não (Union Foreground) | 1477.29 ± 150.29 | 1469.21 (1370.84 - 1573.35) | 823.39 - 2008.08 |
| Whole Tumor (WT) | 95.97 ± 57.32 | 89.33 (48.26 - 136.70) | 2.81 - 361.78 |
| Tumor Core (TC) | 35.75 ± 30.00 | 28.27 (12.14 - 51.71) | 0.00 - 190.19 |
| Enhancing Tumor (ET) | 21.45 ± 18.06 | 17.34 (7.35 - 31.42) | 0.00 - 111.25 |
| NCR (Label 1) | 14.31 ± 20.65 | 7.37 (1.86 - 19.33) | 0.00 - 189.15 |
| Edema (Label 2) | 60.21 ± 42.26 | 52.31 (25.63 - 85.81) | 0.00 - 216.41 |

## 4. Hình thái khối u, Bounding Box & Thành phần liên thông (CCA)
- **Bounding box WT (X × Y × Z) trung vị:** 60 × 83 × 67 voxels (P95: 94 × 122 × 92)
- **Số lát cắt axial chứa u (WT):** trung vị 66 (min 11, max 109)
- **Tỉ lệ ca có toàn bộ khối u nằm gọn trong kích thước patch vuông:** `le_64`: 17.1%, `le_96`: 71.6%, `le_128`: 96.6%, `le_160`: 99.8%
- **WT có nhiều hơn 1 thành phần liên thông:** 891 ca
- **Phân tích cụm nhỏ vùng ET (liên quan hậu xử lý CCA):** có cụm < 50 vox: 900 ca; < 100 vox: 915 ca; < 200 vox: 931 ca; ca mà cụm ET lớn nhất < 100 vox (rủi ro bị xóa sạch nếu CCA ngưỡng 100): **9 ca**

## 5. Thống kê Cường độ tín hiệu 4 Kênh MRI (trên vùng Foreground > 0)
| Kênh | Dtype | % Voxel nền (=0) | Mean ± Std | Median | P99 | P99.5 | Max | Tỉ lệ tương phản ET/FG | Tỉ lệ tương phản ED/FG |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `t1n` | int16/float32 | 83.46% | 897.1 ± 233.6 | 405.0 | 571.0 | 587.0 | 150107.9 | 0.96x | 1.02x |
| `t1c` | int16/float32 | 83.46% | 1489.4 ± 502.5 | 444.0 | 1030.0 | 1209.0 | 2120537.0 | 1.57x | 0.94x |
| `t2w` | int16/float32 | 83.46% | 1483.9 ± 802.3 | 474.0 | 1472.0 | 1568.0 | 4563633.0 | 1.37x | 1.29x |
| `t2f` | int16/float32 | 83.48% | 610.1 ± 211.0 | 307.0 | 593.5 | 641.0 | 612366.4 | 1.74x | 1.65x |

## 6. Kiểm tra độ cân bằng của Split (`brats23_patient_split_70_10_20.json`)
| Tập | Số ca | Số bệnh nhân | WT Mean (cm³) | WT Median (cm³) | TC Mean (cm³) | ET Mean (cm³) | % Ca thiếu ET |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 876/876 | 793 | 96.03 | 90.42 | 36.21 | 21.45 | 2.63% |
| `val` | 126/126 | 114 | 96.93 | 93.47 | 33.82 | 21.32 | 2.38% |
| `test` | 249/249 | 226 | 95.27 | 87.54 | 35.13 | 21.50 | 2.81% |

- **Kiểm tra trùng bệnh nhân giữa các tập:** `train ∩ val` = **0**, `train ∩ test` = **0**, `val ∩ test` = **0**
- **Kiểm định đồng nhất Kruskal-Wallis:** `brain_cm3`: p=0.6220, `wt_cm3`: p=0.9678, `tc_cm3`: p=0.6189, `et_cm3`: p=0.9363, `ncr_cm3`: p=0.2747, `ed_cm3`: p=0.7061
- **Kiểm định đồng nhất Chi-square:** `has_et`: p=0.9695, `has_ncr`: p=0.9752, `is_multi_scan`: p=0.9843

## 7. Phân tích Cặp chụp lặp & Nhóm độ khó (Chuẩn bị cho Paper)
- **Số bệnh nhân có 2 lần chụp (`-000` & `-001`):** **118** bệnh nhân
  - Biến thiên tuyệt đối giữa `-000` và `-001` (Median [IQR]): WT = 16.60 (6.04–32.65) cm³, ET = 4.35 (1.75–11.38) cm³
- **Phân tầng kích thước WT theo Tertiles:** Small (≤ 63.44 cm³): 417 ca | Medium (63.44–118.37 cm³): 417 ca | Large (> 118.37 cm³): 417 ca
- **Nhóm ca khó:** WT < 20 cm³ = 81 ca | ET nhỏ (< 2 cm³, > 0) = 87 ca | WT đa ổ (> 1 cụm liên thông) = 891 ca
