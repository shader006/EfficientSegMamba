# Benchmark Tốc Độ Đọc Dữ Liệu (`all/data`)

- **Thư mục dữ liệu:** `/home/cuc.buithi/BRATS/all/data`
- **Số ca đo:** `10` ca | **Lặp lại:** `3` lần/ca (lấy best-of-n) | **Patch size:** `128^3`

| Hạng | Phương thức đọc | Mean (ms/ca) | Median (ms/ca) | Std (ms) | Nhanh hơn `.npz` |
| :---: | :--- | ---: | ---: | ---: | ---: |
| #1 | RAM cache slice (patch) | **10.26** | 10.42 | 0.52 | 49.33x (x1.00 vs #1) |
| #2 | /dev/shm .npy mmap (patch) | **13.24** | 12.98 | 1.06 | 38.23x (x1.29 vs #1) |
| #3 | NFS .npy mmap (patch) | **17.37** | 16.82 | 1.66 | 29.13x (x1.69 vs #1) |
| #4 | NFS .npy full read | **28.91** | 23.44 | 19.67 | 17.51x (x2.82 vs #1) |
| #5 | /dev/shm .npy full read | **34.14** | 18.68 | 33.30 | 14.82x (x3.33 vs #1) |
| #6 | NFS .npz compressed (full) | **506.03** | 510.78 | 79.02 | 1.00x (x49.33 vs #1) |
