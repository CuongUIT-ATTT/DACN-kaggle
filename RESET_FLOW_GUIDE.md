# Hướng Dẫn Reset Flow - Xóa Toàn Bộ File Output

**Ngày tạo:** 11 tháng 3, 2026  
**Mục đích:** Xóa tất cả file đã tạo để chạy lại flow từ đầu

---

## 🎯 Mục Đích

Document này hướng dẫn cách xóa sạch tất cả file output đã được tạo ra trong quá trình xử lý để bạn có thể:
- Chạy lại flow từ đầu một cách sạch sẽ
- Giải phóng dung lượng ổ cứng
- Debug/test với dữ liệu mới
- Khắc phục lỗi từ các lần chạy trước

---

## 📁 Cấu Trúc Thư Mục Output

Các thư mục chứa file output được tạo ra:

```
VISION/
├── tmp/
│   ├── cwe20cfa/
│   │   ├── cpg/          ← CPG binary và JSON files
│   │   ├── source/       ← Source code files (.c)
│   │   ├── input/        ← Input processed files
│   │   ├── model/        ← Model files
│   │   └── w2v/          ← Word2Vec files
│   ├── tokens/           ← Token files
│   └── joern_temp_script*.sc  ← Joern temporary scripts
├── workspace/            ← Joern workspace
└── /tmp/
    ├── joern_test*/      ← Test directories
    └── joern-default*.semantics  ← Joern semantics files
```

---

## 🧹 Lệnh Reset - Xóa Tất Cả File Output

### **Option 1: Reset Hoàn Toàn (Recommended)**

Xóa tất cả file output và temporary files:

```bash
cd /home/cuong/DACN/VISION

# Xóa tất cả file trong các thư mục output
rm -rf tmp/cwe20cfa/cpg/* \
       tmp/cwe20cfa/source/* \
       tmp/cwe20cfa/input/* \
       tmp/cwe20cfa/model/* \
       tmp/cwe20cfa/w2v/* \
       tmp/tokens/* \
       workspace/

# Xóa Joern temporary scripts
rm -f tmp/joern_temp_script*.sc

# Xóa Joern semantics files
rm -f /tmp/joern-default*.semantics

# Xóa test directories
rm -rf /tmp/joern_test*

echo "✓ Reset hoàn tất! Tất cả file output đã được xóa."
```

### **Option 2: Reset Từng Phần**

#### 2.1. Chỉ xóa CPG files
```bash
cd /home/cuong/DACN/VISION
rm -rf tmp/cwe20cfa/cpg/*
echo "✓ Đã xóa CPG files"
```

#### 2.2. Chỉ xóa Source files
```bash
cd /home/cuong/DACN/VISION
rm -rf tmp/cwe20cfa/source/*
echo "✓ Đã xóa Source files"
```

#### 2.3. Chỉ xóa Joern workspace
```bash
cd /home/cuong/DACN/VISION
rm -rf workspace/
echo "✓ Đã xóa Joern workspace"
```

#### 2.4. Chỉ xóa Temporary scripts
```bash
cd /home/cuong/DACN/VISION
rm -f tmp/joern_temp_script*.sc
rm -f /tmp/joern-default*.semantics
echo "✓ Đã xóa Temporary scripts"
```

---

## 🔍 Kiểm Tra Kết Quả

Sau khi chạy lệnh reset, kiểm tra xem đã xóa sạch chưa:

```bash
cd /home/cuong/DACN/VISION

echo "=== Kiểm tra thư mục output ==="
for dir in tmp/cwe20cfa/cpg tmp/cwe20cfa/source tmp/cwe20cfa/input tmp/cwe20cfa/model tmp/cwe20cfa/w2v tmp/tokens; do
    count=$(ls -A "$dir" 2>/dev/null | wc -l)
    echo "$dir: $count files"
done

echo ""
echo "=== Kiểm tra workspace ==="
ls -la workspace/ 2>/dev/null || echo "✓ workspace/ đã bị xóa"

echo ""
echo "=== Kiểm tra temp scripts ==="
ls -la tmp/joern_temp_script*.sc 2>/dev/null || echo "✓ Không còn temp scripts"
```

**Kết quả mong đợi:**
```
tmp/cwe20cfa/cpg: 0 files
tmp/cwe20cfa/source: 0 files
tmp/cwe20cfa/input: 0 files
tmp/cwe20cfa/model: 0 files
tmp/cwe20cfa/w2v: 0 files
tmp/tokens: 0 files

✓ workspace/ đã bị xóa
✓ Không còn temp scripts
```

---

## 📝 Script Tự Động

Tạo script để reset nhanh:

```bash
# Tạo script reset.sh
cat > reset_flow.sh << 'EOF'
#!/bin/bash

echo "=== Bắt đầu reset flow ==="

cd /home/cuong/DACN/VISION

# Xóa output directories
echo "Đang xóa output directories..."
rm -rf tmp/cwe20cfa/cpg/* \
       tmp/cwe20cfa/source/* \
       tmp/cwe20cfa/input/* \
       tmp/cwe20cfa/model/* \
       tmp/cwe20cfa/w2v/* \
       tmp/tokens/* \
       workspace/

# Xóa temporary files
echo "Đang xóa temporary files..."
rm -f tmp/joern_temp_script*.sc
rm -f /tmp/joern-default*.semantics
rm -rf /tmp/joern_test*

# Verify
echo ""
echo "=== Kiểm tra kết quả ==="
for dir in tmp/cwe20cfa/cpg tmp/cwe20cfa/source tmp/cwe20cfa/input tmp/cwe20cfa/model tmp/cwe20cfa/w2v tmp/tokens; do
    count=$(ls -A "$dir" 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "✓ $dir: sạch"
    else
        echo "✗ $dir: còn $count files"
    fi
done

echo ""
echo "✅ Reset hoàn tất!"
EOF

# Cho phép thực thi
chmod +x reset_flow.sh
```

**Sử dụng:**
```bash
./reset_flow.sh
```

---

## ⚠️ Lưu Ý Quan Trọng

### 1. **Backup trước khi reset**
Nếu có file quan trọng, backup trước:
```bash
# Backup dataset đã tạo
cp -r datasets/cwe20cfa/*.pkl /path/to/backup/

# Backup model files (nếu có)
cp -r tmp/cwe20cfa/model/* /path/to/backup/models/
```

### 2. **Không xóa nhầm dataset gốc**
Các lệnh reset **KHÔNG** xóa:
- ✅ `datasets/cwe20cfa/raw/` - Dataset gốc
- ✅ `joern/joern-cli/` - Joern installation
- ✅ `joern/graph-for-funcs.sc` - Joern script
- ✅ Python scripts (.py files)

### 3. **Kiểm tra trước khi chạy**
Luôn chạy với `echo` hoặc `ls` để xem trước:
```bash
# Xem file sẽ bị xóa
ls -la tmp/cwe20cfa/cpg/
ls -la tmp/cwe20cfa/source/

# Sau đó mới xóa
rm -rf tmp/cwe20cfa/cpg/*
```

### 4. **Conda environment**
Reset **KHÔNG** ảnh hưởng đến:
- ✅ Conda environment `gear`
- ✅ Installed packages
- ✅ Python dependencies

---

## 🚀 Workflow Sau Khi Reset

Sau khi reset xong, chạy lại flow từ đầu:

```bash
# 1. Activate conda environment
conda activate gear

# 2. Chạy graph2cpg để tạo CPG
python graph2cpg.py -d test

# HOẶC

# 2. Chạy generate counterexample dataset
python generate_counterexample_dataset.py -d test
```

---

## 📊 Ước Tính Dung Lượng Giải Phóng

Tùy thuộc vào số lượng examples đã xử lý:

| Thư mục | Ước tính dung lượng |
|---------|---------------------|
| `tmp/cwe20cfa/cpg/` | ~100-500 MB |
| `tmp/cwe20cfa/source/` | ~1-10 MB |
| `workspace/` | ~50-200 MB |
| Temporary files | ~1-5 MB |
| **Tổng cộng** | **~150-700 MB** |

---

## 🛠️ Troubleshooting

### Vấn đề: Permission denied
```bash
# Giải pháp: Thêm sudo (nếu cần)
sudo rm -rf tmp/cwe20cfa/cpg/*
```

### Vấn đề: File đang được sử dụng
```bash
# Giải pháp: Kill process đang dùng
pkill -9 joern
pkill -9 python

# Sau đó mới xóa
rm -rf tmp/cwe20cfa/cpg/*
```

### Vấn đề: Thư mục không tồn tại
```bash
# Giải pháp: Tạo lại thư mục
mkdir -p tmp/cwe20cfa/{cpg,source,input,model,w2v}
mkdir -p tmp/tokens
```

---

## ✅ Checklist Trước Khi Chạy Lại

- [ ] Đã backup file quan trọng (nếu có)
- [ ] Đã xóa tất cả file output
- [ ] Đã verify các thư mục đã sạch
- [ ] Đã activate conda environment
- [ ] Joern đang hoạt động bình thường
- [ ] File `graph-for-funcs.sc` đã được sửa lỗi (xóa dòng import `dataflowengine`)

---

## 📚 Tài Liệu Liên Quan

- [FIX_RACE_CONDITION_REPORT.md](FIX_RACE_CONDITION_REPORT.md) - Báo cáo sửa lỗi Race Condition
- [README.md](README.md) - Hướng dẫn sử dụng project

---

**Lưu ý cuối:** Luôn đảm bảo bạn đang ở đúng thư mục và hiểu rõ lệnh trước khi thực thi để tránh xóa nhầm dữ liệu quan trọng! 🔒
