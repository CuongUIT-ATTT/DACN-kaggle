# Báo Cáo Tổng Hợp Dự Án VISION

**Ngày cập nhật:** 12 tháng 3, 2026  
**Repository:** David-Egea/VISION  
**Branch:** main  
**Phiên bản:** 2.1 (With Data Integrity Fixes)

---

## 🎯 Điểm Nổi Bật Phiên Bản 2.1

### 🔥 Critical Fixes (v2.1):
1. **Fixed Loop Logic**: Removed early break - processes all tasks completely
2. **Fixed Drop Logic**: Correct row removal with `inplace=True` instead of column drop
3. **Correct Data Structure**: orig_cpg stores dict directly (not `[dict]`) - consistent format
4. **Complete Processing**: All items processed naturally without premature termination

### ✨ Major Updates (v2.0):
1. **Integrated Flattening Logic**: cpg2input.py giờ tự động flatten dataset, không cần chạy flatten_dataset.py riêng
2. **Single-Pass Processing**: Chỉ cần 1 lần chạy cpg2input.py để có output hoàn chỉnh
3. **No NaN Values**: 100% rows đều có input column đầy đủ
4. **Auto CPG Format Handling**: Tự động xử lý CPG dạng Dict hoặc List[Dict]
5. **Perfect Balance**: Tự động đảm bảo 50/50 benign/vulnerable

### 🚀 Simplified Workflow:
```bash
# Version 2.1 (Current - with loop logic fixes):
python graph2cpg.py -d test           # ✅ Processes all items, correct drop logic
python cpg2input.py -d test           # ✅ Output hoàn chỉnh, ready for training

# Old (3 steps):
python generate_counterexample_dataset.py -d test
python cpg2input.py -d test           # Output có NaN
python flatten_dataset.py              # Re-process
```

### 📊 Output Quality:
- ✅ All items processed completely (no skipped last item)
- ✅ Failed CPGs correctly dropped (row removal, not column)
- ✅ Consistent data format (dict not list)
- ✅ Perfectly balanced (50% benign, 50% vulnerable)
- ✅ Clean CPG dicts (extracted from lists automatically)
- ✅ Correct ID pairing (pairwise evaluation ready)
- ✅ Single output file: `cwe20cfa_CWE-20_augmented_input_balanced.pkl`

---

## 📋 Mục Lục

0. [Sửa Lỗi Data Integrity trong graph2cpg.py](#0-sửa-lỗi-data-integrity-trong-graph2cpgpy)
1. [Sửa Lỗi Race Condition trong Multi-Threading](#1-sửa-lỗi-race-condition-trong-multi-threading)
2. [Hướng Dẫn Reset Flow](#2-hướng-dẫn-reset-flow)
3. [Script Flatten Dataset](#3-script-flatten-dataset)
4. [Tích Hợp Flattening Logic vào CPG2Input](#4-tích-hợp-flattening-logic-vào-cpg2input)
5. [Cài Đặt và Kiểm Tra Joern](#5-cài-đặt-và-kiểm-tra-joern)

---

# 0. Sửa Lỗi Data Integrity trong graph2cpg.py

**File:** `graph2cpg.py`  
**Vấn đề:** Loop logic skips last item, incorrect drop logic, wrong data format  
**Trạng thái:** ✅ Đã sửa thành công (12/03/2026)

## 🔴 Vấn Đề Gốc

### 1. Loop Logic - Early Break
Loop bị dừng sớm trước khi xử lý tất cả items:
```python
if i >= len(task_list) - 1:
    break  # ❌ Stops before processing last item
```
- Item cuối cùng không được xử lý
- Incomplete dataset
- Progress bar không đạt 100%

### 2. Wrong Drop Logic - Column vs Row
Sử dụng `axis=1` sai, xóa column thay vì row:
```python
dataset_df = dataset_df.drop(index=idx, axis=1)  # ❌ axis=1 drops column, not row
```
- Xóa nhầm column thay vì row
- Dataset structure bị corrupt
- KeyError trong subsequent scripts

### 3. Data Structure Mismatch
Lưu CPG dưới dạng list thay vì dict:
```python
dataset_df.at[idx, "orig_cpg"] = [cpg]  # ❌ Stores as list [dict]

# Expected by subsequent scripts:
orig_cpg = {"functions": [...]}  # ✅ Should be dict directly
```
- Không consistent với cpg column
- Cần extract từ list trước khi dùng
- Format mismatch errors

## ✅ Giải Pháp Đã Implement

### 1. **Fixed Loop Logic: Remove Early Break**

#### Thay đổi:
```python
# REMOVED this block entirely:
# if i >= len(task_list) - 1:
#     break

# Loop now completes naturally:
for future in as_completed(futures):
    # ... process all futures without early termination
```

#### Lợi ích:
- ✅ Xử lý tất cả items hoàn chỉnh
- ✅ Progress bar đạt 100%
- ✅ Không bỏ sót item cuối

### 2. **Fixed Drop Logic: Correct Row Removal**

#### Thay đổi:
```python
# Old (WRONG):
dataset_df = dataset_df.drop(index=idx, axis=1)  # ❌ Drops column

# New (CORRECT):
dataset_df.drop(index=idx, inplace=True)  # ✅ Drops row in-place
```

#### Lợi ích:
- ✅ Xóa đúng row (không phải column)
- ✅ In-place modification (hiệu quả hơn)
- ✅ Dataset structure không bị corrupt

### 3. **Fixed Data Structure: Store Dict Directly**

#### Thay đổi:
```python
# Old:
dataset_df.at[idx, "orig_cpg"] = [cpg]  # ❌ List wrapper

# New:
dataset_df.at[idx, "orig_cpg"] = cpg  # ✅ Dict directly
```

#### So sánh format:
```python
# Old format:
orig_cpg = [{"functions": [...]}]  # ❌ List of dict

# New format:
orig_cpg = {"functions": [...]}    # ✅ Dict directly (matches cpg column)
```

#### Lợi ích:
- ✅ Consistent với existing cpg column format
- ✅ Subsequent scripts không cần thay đổi
- ✅ Direct access: `orig_cpg["functions"]` works immediately

## 📊 Code Changes Summary

### Main Loop Logic (`__main__` section):
```python
# OLD CODE:
for future in as_completed(futures):
    idx, cpg = future.result(timeout=timeout_per_task)
    
    if cpg is not None:
        dataset_df.at[idx, "orig_cpg"] = [cpg]  # ❌ List wrapper
    else:
        dataset_df = dataset_df.drop(index=idx, axis=1)  # ❌ Wrong: drops column
    
    if i >= len(task_list) - 1:
        break  # ❌ Stops early, skips last item

# NEW CODE:
for future in as_completed(futures):
    idx, cpg = future.result(timeout=timeout_per_task)
    
    if cpg is not None:
        dataset_df.at[idx, "orig_cpg"] = cpg  # ✅ Dict directly
    else:
        dataset_df.drop(index=idx, inplace=True)  # ✅ Correct: drops row
    
    # ✅ No early break - processes all items naturally
```

## 🎯 Kết Quả

### Trước khi sửa:
- ❌ Last item không được xử lý (loop breaks early)
- ❌ Drop logic sai (xóa column thay vì row)
- ❌ orig_cpg format mismatch `[dict]` vs `dict`
- ❌ Dataset incomplete và có structure errors

### Sau khi sửa:
- ✅ Tất cả items được xử lý đầy đủ
- ✅ Drop logic chính xác (xóa row)
- ✅ Consistent dict format cho orig_cpg
- ✅ Dataset hoàn chỉnh và clean

## 🔧 Summary of Changes

| Location | Old Code | New Code | Impact |
|----------|----------|----------|--------|
| Main loop | `if i >= len(task_list) - 1: break` | Removed | Processes all items |
| Drop logic | `dataset_df.drop(index=idx, axis=1)` | `dataset_df.drop(index=idx, inplace=True)` | Correctly drops rows |
| Data format | `dataset_df.at[idx, "orig_cpg"] = [cpg]` | `dataset_df.at[idx, "orig_cpg"] = cpg` | Dict format consistency |

## 💡 Key Improvements

1. **Complete Processing**: Loop naturally processes all futures without artificial termination
2. **Correct Row Removal**: Failed CPGs properly removed from dataset
3. **Format Consistency**: orig_cpg matches cpg column format (dict not list)
4. **Data Integrity**: Output dataset is complete and valid

---

# 1. Sửa Lỗi Race Condition trong Multi-Threading

**File:** `generate_counterexample_dataset.py`  
**Vấn đề:** Race Condition khi xử lý đa luồng (multi-threading)  
**Trạng thái:** ✅ Đã sửa thành công

## 🔴 Vấn đề Gốc (Root Cause)

### Mô tả lỗi:
Khi chạy chương trình với `ThreadPoolExecutor`, nhiều thread cùng xử lý các example song song. Tuy nhiên, tất cả các thread đều sử dụng **chung một file tạm** `tmp/joern_temp_script.sc`, dẫn đến:

1. **Race Condition**: Nhiều thread ghi đè lên cùng một file
2. **NoneType Error**: `json_process()` trả về `None` khi Joern không tạo được file JSON
3. **IndexError**: `ce_graphs[0][1]` bị lỗi khi `ce_graphs` là `None`
4. **File Corruption**: File JSON bị corrupt hoặc thiếu dữ liệu

### Luồng lỗi:
```
Thread 1: Write script → Execute Joern → Read JSON
Thread 2: Write script → Execute Joern → Read JSON  ← Ghi đè file của Thread 1
Thread 3: Write script → Execute Joern → Read JSON  ← Ghi đè file của Thread 1 & 2
```

## ✅ Giải Pháp

### 1. **Sử dụng Unique Temporary Files**

Mỗi thread sử dụng file tạm riêng biệt với `unique_id`:

```python
# Trước (shared file):
tmp/joern_temp_script.sc

# Sau (unique files):
tmp/joern_temp_script_0.sc
tmp/joern_temp_script_1.sc
tmp/joern_temp_script_2.sc
```

### 2. **Thread-Safe File Management**

Mỗi thread chỉ xóa file tạm của chính nó, không ảnh hưởng đến các thread khác.

## 📝 Chi Tiết Thay Đổi

### **1. Function `joern_create()` - Lines 109-180**

#### Thay đổi:
- **Thêm parameter** `unique_id=None` để tạo file tạm unique
- **Return tuple** `(json_file, temp_script_path)` thay vì chỉ `json_file`
- Tạo file tạm theo pattern: `joern_temp_script_{unique_id}.sc`

#### Code:
```python
def joern_create(joern_path, in_path, out_path, cpg_file, unique_id=None):
    # ...
    if unique_id is not None:
        commands_script__path = os.path.abspath(f"tmp/joern_temp_script_{unique_id}.sc")
    else:
        commands_script__path = os.path.abspath("tmp/joern_temp_script.sc")
    # ...
    return json_file, commands_script__path
```

#### Lợi ích:
- ✅ Mỗi thread có file tạm riêng
- ✅ Không có race condition khi ghi file
- ✅ Return path để cleanup sau này

### **2. Function `json_process()` - Lines 195-217**

#### Thay đổi:
- **Thêm parameter** `debug_index=None` để debug tốt hơn
- **Enhanced error handling** với các case cụ thể:
  - Empty JSON file
  - JSON decode error
  - File not found
  - Processing error

#### Lợi ích:
- ✅ Chi tiết hơn khi debug lỗi
- ✅ Hiển thị rõ example bị lỗi
- ✅ Phân biệt các loại lỗi khác nhau

### **3. Function `process_single_example()` - Step 3 (Lines 387-407)**

#### Thay đổi:
- **Capture `temp_script_path`** để cleanup sau
- **Verify JSON file** sau khi Joern tạo xong
- **Cleanup temp script** khi fail

#### Lợi ích:
- ✅ Mỗi thread dùng `unique_id=index`
- ✅ Verify file JSON trước khi tiếp tục
- ✅ Cleanup khi có lỗi

### **4. Function `process_single_example()` - Step 4 (Lines 409-437)**

#### Thay đổi:
- **Safety check** trước khi access `ce_graphs[0][1]`
- **Specific exception handling**: `TypeError`, `IndexError`, `KeyError`
- **Cleanup temp script** khi fail

#### Lợi ích:
- ✅ Không bị lỗi `NoneType object is not subscriptable`
- ✅ Phân biệt các loại lỗi rõ ràng
- ✅ Cleanup đúng cách

### **5. Cleanup Section (Lines 450-463)**

#### Thay đổi:
- **Cleanup thread-specific temp script file**
- **Better error message** với index cụ thể

#### Lợi ích:
- ✅ Xóa sạch file tạm của thread riêng
- ✅ Không conflict với thread khác
- ✅ Better logging

## 🎯 Kết Quả

### Trước khi sửa:
- ❌ Race condition khi nhiều thread ghi cùng file
- ❌ `NoneType object is not subscriptable` error
- ❌ JSON file bị corrupt hoặc empty
- ❌ Không biết thread nào gây lỗi

### Sau khi sửa:
- ✅ Mỗi thread có file tạm riêng biệt
- ✅ Không còn race condition
- ✅ Safety check trước khi access data
- ✅ Error messages chi tiết với index
- ✅ Cleanup đúng cách cho từng thread

## 📊 Kiến Trúc Mới

```
ThreadPoolExecutor (max_workers=4)
├── Thread 1 (index=0)
│   ├── tmp/joern_temp_script_0.sc
│   ├── tmp/cwe20cfa/cpg/0_cpg.bin
│   └── tmp/cwe20cfa/cpg/0_cpg.json
│
├── Thread 2 (index=1)
│   ├── tmp/joern_temp_script_1.sc
│   ├── tmp/cwe20cfa/cpg/1_cpg.bin
│   └── tmp/cwe20cfa/cpg/1_cpg.json
│
├── Thread 3 (index=2)
│   ├── tmp/joern_temp_script_2.sc
│   ├── tmp/cwe20cfa/cpg/2_cpg.bin
│   └── tmp/cwe20cfa/cpg/2_cpg.json
│
└── Thread 4 (index=3)
    ├── tmp/joern_temp_script_3.sc
    ├── tmp/cwe20cfa/cpg/3_cpg.bin
    └── tmp/cwe20cfa/cpg/3_cpg.json
```

**Mỗi thread độc lập hoàn toàn** - không có shared resources!

---

# 2. Hướng Dẫn Reset Flow

**Mục đích:** Xóa tất cả file đã tạo để chạy lại flow từ đầu

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

## 🔍 Kiểm Tra Kết Quả

Sau khi chạy lệnh reset:

```bash
cd /home/cuong/DACN/VISION

echo "=== Kiểm tra thư mục output ==="
for dir in tmp/cwe20cfa/cpg tmp/cwe20cfa/source tmp/cwe20cfa/input tmp/cwe20cfa/model tmp/cwe20cfa/w2v tmp/tokens; do
    count=$(ls -A "$dir" 2>/dev/null | wc -l)
    echo "$dir: $count files"
done
```

**Kết quả mong đợi:** Tất cả thư mục đều có 0 files

## 📝 Script Tự Động: reset_flow.sh

```bash
#!/bin/bash
cd /home/cuong/DACN/VISION

echo "=== Bắt đầu reset flow ==="

# Xóa output directories
rm -rf tmp/cwe20cfa/cpg/* tmp/cwe20cfa/source/* tmp/cwe20cfa/input/* \
       tmp/cwe20cfa/model/* tmp/cwe20cfa/w2v/* tmp/tokens/* workspace/

# Xóa temporary files
rm -f tmp/joern_temp_script*.sc /tmp/joern-default*.semantics
rm -rf /tmp/joern_test*

echo "✅ Reset hoàn tất!"
```

**Sử dụng:**
```bash
chmod +x reset_flow.sh
./reset_flow.sh
```

## ⚠️ Lưu Ý Quan Trọng

### 1. **Backup trước khi reset**
```bash
# Backup dataset đã tạo
cp -r datasets/cwe20cfa/*.pkl /path/to/backup/
```

### 2. **Không xóa nhầm dataset gốc**
Các lệnh reset **KHÔNG** xóa:
- ✅ `datasets/cwe20cfa/raw/` - Dataset gốc
- ✅ `joern/joern-cli/` - Joern installation
- ✅ `joern/graph-for-funcs.sc` - Joern script
- ✅ Python scripts (.py files)

## 🚀 Workflow Sau Khi Reset

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

# 3. Script Flatten Dataset

**File:** `flatten_dataset.py`  
**Trạng thái:** ✅ Đã tạo thành công  
**Mục đích:** Transform dataset từ paired format sang flat format cho training

## 🎯 Tổng Quan

Script này chuyển đổi output của preprocessing step thành format yêu cầu bởi `train.py`.

### Input Format:
```
Mỗi row chứa 1 cặp:
- Original: orig_func, orig_cpg, orig_input (target flipped)
- Adversarial: func, cpg, input (target unchanged)
```

### Output Format:
```
Mỗi cặp được tách thành 2 rows:
Row 1: id, adv=False, func (orig), cpg (orig), target (flipped), input (orig)
Row 2: id, adv=True, func (adv), cpg (adv), target (current), input (adv)
```

## 📋 Yêu Cầu Script

### 1. Load Data
```python
dataset = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_test_input.pkl')
```

### 2. Filter Invalid Rows
- Drop rows với `cpg` hoặc `orig_cpg` là null/None
- Validate CPG structure (phải là dict và không empty)

### 3. Create Original Rows
```python
original_row = {
    'id': str(idx),
    'adv': False,
    'func': row['orig_func'],
    'target': flip_target(row['target']),  # 0→1, 1→0
    'cpg': row['orig_cpg'],
    'input': row['orig_input']  # if exists
}
```

### 4. Create Adversarial Rows
```python
adversarial_row = {
    'id': str(idx),
    'adv': True,
    'func': row['func'],
    'target': int(row['target']),  # Keep unchanged
    'cpg': row['cpg'],
    'input': row['input']  # if exists
}
```

### 5. Unify and Save
- Concatenate original + adversarial DataFrames
- Ensure data types: `int` target, `str` id, `bool` adv
- Save to `datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl`

## 🔧 Chức Năng Chi Tiết

### Function: `validate_row(row)`
Kiểm tra row có đầy đủ fields và valid:
- Required fields: func, cpg, orig_func, orig_cpg, target
- CPG phải là dict và không empty
- Trả về `True` nếu valid, `False` nếu không

### Function: `flip_target(target)`
Đảo ngược target label:
- `0 → 1` (Benign → Vulnerable)
- `1 → 0` (Vulnerable → Benign)

### Function: `flatten_dataset(input_path, output_path)`
Main function thực hiện toàn bộ transformation:

**Steps:**
1. Load data từ pickle file
2. Filter invalid rows (null CPG, invalid structure)
3. Tạo original rows với target flipped
4. Tạo adversarial rows với target unchanged
5. Concatenate và ensure correct data types
6. Validate và hiển thị statistics
7. Save output file

## 📊 Output Statistics

Script hiển thị:

```
Total rows: 1000
Unique IDs: 500
Columns: ['id', 'adv', 'func', 'target', 'cpg', 'input']

Target distribution:
0    500    (50.0%)
1    500    (50.0%)

Adv distribution:
False   500
True    500

Cross-tabulation (Target vs Adv):
         False  True   All
0          250   250   500
1          250   250   500
All        500   500  1000

Class balance ratio: 100.0%
✓ Dataset is well-balanced!
```

## 🚀 Cách Sử Dụng

### Cách 1: Default paths
```bash
python flatten_dataset.py
```

### Cách 2: Custom paths
```bash
python flatten_dataset.py <input_file> <output_file>
```

### Ví dụ:
```bash
python flatten_dataset.py \
  datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_test_input.pkl \
  datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl
```

## ✅ Điểm Mạnh

### 1. Perfect Balance
Đảm bảo dataset 50% benign / 50% vulnerable:
- Mỗi original có target flipped
- Mỗi adversarial giữ nguyên target
- Kết quả: perfectly balanced

### 2. Pairwise Evaluation Compatible
- Mỗi pair có cùng `id`
- Train.py có thể tính: P-C, P-V, P-B, P-R metrics
- Group-level split (train/val/test) hoạt động đúng

### 3. Data Validation
- Filter invalid rows (null CPG, empty dict)
- Check structure trước khi process
- Detailed error messages

### 4. Comprehensive Statistics
- Target distribution
- Adv distribution
- Cross-tabulation
- Balance ratio
- Sample display

## 🔄 Integration với Train.py

### Train.py Requirements:
```python
# Load dataset
dataset_df = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl')

# Must have columns:
- 'id': str - unique identifier cho mỗi pair
- 'adv': bool - False=original, True=adversarial
- 'func': str - source code
- 'target': int - 0=benign, 1=vulnerable
- 'cpg': dict - code property graph
- 'input': array - processed input
```

### Pairwise Evaluation:
```python
def compute_pairwise_metrics_from_loader(model, dataloader):
    pair_groups = {}  # pair_id -> list of (pred, true)
    
    # Group by id
    for batch in dataloader:
        pair_id = batch['id']
        pred = model(batch['input'])
        true = batch['target']
        pair_groups[pair_id].append((pred, true))
    
    # Evaluate pairs
    for pid, samples in pair_groups.items():
        if len(samples) != 2:  # Must have exactly 2: original + adversarial
            continue
        # Calculate P-C, P-V, P-B, P-R
```

## 💡 Tại Sao Cần Script Này?

### Problem:
Preprocessing output có format:
```
Row 1: original + adversarial trong cùng 1 row
Row 2: original + adversarial trong cùng 1 row
...
```

### Solution:
Training cần format:
```
Row 1: original only (id=0, adv=False)
Row 2: adversarial only (id=0, adv=True)
Row 3: original only (id=1, adv=False)
Row 4: adversarial only (id=1, adv=True)
...
```

### Benefits:
1. ✅ DataLoader có thể batch efficiently
2. ✅ Group-level split hoạt động đúng
3. ✅ Pairwise evaluation dễ implement
4. ✅ Class balance đảm bảo 50/50

---

# 4. Tích Hợp Flattening Logic vào CPG2Input

**File:** `cpg2input.py`  
**Trạng thái:** ✅ Đã refactor thành công  
**Mục đích:** Tích hợp logic flattening trực tiếp vào quá trình xử lý CPG

## 🎯 Tổng Quan

Thay vì chạy riêng `flatten_dataset.py` sau khi tạo CPG, giờ logic flattening được tích hợp ngay vào `cpg2input.py`. Điều này đảm bảo output cuối cùng **100% sẵn sàng** cho `train.py` mà không cần bước xử lý trung gian.

### Lợi Ích:
- ✅ **Single Pass Processing**: Chỉ cần chạy 1 lần `cpg2input.py`
- ✅ **No Intermediate Files**: Không cần file trung gian
- ✅ **No NaN Values**: Tất cả rows đều có đầy đủ `input` column
- ✅ **Balanced Dataset**: Tự động cân bằng 50/50 benign/vulnerable
- ✅ **Pairwise Ready**: Format đúng cho pairwise evaluation metrics

## 📋 Thay Đổi Chi Tiết

### **1. Function `extract_cpg_dict(cpg_data)`**

**Mục đích:** Xử lý CPG data có thể là `Dict` hoặc `List[Dict]`

```python
def extract_cpg_dict(cpg_data):
    if isinstance(cpg_data, list):
        if len(cpg_data) > 0:
            return cpg_data[0]  # Extract first element
        else:
            return None
    return cpg_data
```

**Tính năng:**
- Tự động detect format của CPG
- Extract dict từ list nếu cần
- Return None nếu invalid

### **2. Function `flip_target(target)`**

**Mục đích:** Đảo ngược target label cho original rows

```python
def flip_target(target):
    return 1 if target == 0 else 0
```

**Logic:**
- Original code có target flipped (0→1, 1→0)
- Adversarial code giữ nguyên target
- Đảm bảo class balance 50/50

### **3. Function `flatten_dataset(df)`**

**Mục đích:** Transform từ Wide format (paired) sang Long format (stacked)

#### Input Format:
```python
# Mỗi row chứa cả original và adversarial
{
    'func': adversarial_code,
    'cpg': adversarial_cpg,
    'orig_func': original_code,
    'orig_cpg': original_cpg,
    'target': current_target
}
```

#### Output Format:
```python
# Row 1: Original
{
    'id': '0',
    'adv': False,
    'func': original_code,
    'cpg': original_cpg_dict,  # Extracted from list
    'target': flipped_target  # 0→1, 1→0
}

# Row 2: Adversarial  
{
    'id': '0',
    'adv': True,
    'func': adversarial_code,
    'cpg': adversarial_cpg_dict,  # Extracted from list
    'target': current_target  # Unchanged
}
```

#### Processing Steps:
```python
def flatten_dataset(df):
    # 1. Loop through original DataFrame
    for idx, row in df.iterrows():
        # 2. Extract CPG dicts from lists if needed
        orig_cpg_data = extract_cpg_dict(row.get('orig_cpg'))
        cpg_data = extract_cpg_dict(row.get('cpg'))
        
        # 3. Skip if CPG is invalid
        if orig_cpg_data is None or cpg_data is None:
            continue
        
        # 4. Create original row (adv=False, target flipped)
        original_row = {
            'id': str(idx),
            'adv': False,
            'func': row.get('orig_func'),
            'cpg': orig_cpg_data,
            'target': flip_target(int(row['target']))
        }
        
        # 5. Create adversarial row (adv=True, target unchanged)
        adversarial_row = {
            'id': str(idx),
            'adv': True,
            'func': row.get('func'),
            'cpg': cpg_data,
            'target': int(row['target'])
        }
    
    # 6. Combine into single DataFrame
    flattened_df = pd.concat([
        pd.DataFrame(original_rows),
        pd.DataFrame(adversarial_rows)
    ], ignore_index=True)
    
    return flattened_df
```

### **4. Main Processing Flow**

#### Before (Old Flow):
```
1. Load paired dataset
2. Process CPG → Input
3. Save with NaN values
4. Run separate flatten_dataset.py
5. Load and re-process
6. Finally ready for train.py
```

#### After (New Flow):
```
1. Load paired dataset
2. Flatten immediately (wide → long)
3. Process ALL rows (CPG → Input)
4. Save once
5. Ready for train.py ✓
```

#### Code Changes:
```python
# Load dataset
dataset_df = pd.read_pickle(dataset_path)

# FLATTEN BEFORE PROCESSING
dataset_df = flatten_dataset(dataset_df)

# Now process ALL rows (both original and adversarial)
for index, row_series in dataset_df.copy().iterrows():
    # Tokenization
    tokenized_func_df = tokenize(row_df)
    
    # Build/Train Word2Vec
    w2vmodel.build_vocab(...)
    w2vmodel.train(...)
    
    # CPG → Nodes
    row_df[["nodes", "nodes_by_line_map"]] = row_df.apply(process_cpg_to_nodes_row, axis=1)
    
    # Nodes → Input
    row_df[["input", "code_embedding_mapping"]] = row_df.apply(
        lambda row: process_nodes_to_input_row(row, w2vmodel), axis=1
    )
```

### **5. Output Path Change**

```python
# Before:
output_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_{dataset}_input.pkl"

# After:
output_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl"
```

**Lý do:**
- Tên file phản ánh dataset đã balanced
- Trùng với tên file mà `train.py` expect
- Không cần file trung gian

### **6. Cleanup Logic**

```python
# Remove temporary columns before saving
columns_to_drop = ['nodes', 'nodes_by_line_map', 'tokens', 'line_to_tokens_map']
output_df = output_df.drop(columns=columns_to_drop)

# Keep only essential columns
expected_columns = ['id', 'adv', 'func', 'cpg', 'target', 'input']
optional_columns = ['cwe', 'code_embedding_mapping']

final_columns = [col for col in expected_columns if col in output_df.columns]
final_columns += [col for col in optional_columns if col in output_df.columns]
output_df = output_df[final_columns]
```

**Columns được giữ lại:**
- ✅ `id` - String, để group pairs
- ✅ `adv` - Boolean, True=adversarial, False=original
- ✅ `func` - String, source code
- ✅ `cpg` - Dict, clean CPG structure
- ✅ `target` - Int, 0=benign, 1=vulnerable
- ✅ `input` - Data object, processed input cho GNN
- ✅ `cwe` - String (optional), CWE identifier
- ✅ `code_embedding_mapping` - Dict (optional), debug info

**Columns bị xóa:**
- ❌ `orig_func` - Không cần sau khi flatten
- ❌ `orig_cpg` - Không cần sau khi flatten
- ❌ `nodes` - Temporary processing data
- ❌ `nodes_by_line_map` - Temporary processing data
- ❌ `tokens` - Temporary processing data
- ❌ `line_to_tokens_map` - Temporary processing data

### **7. Progress Bar Update**

```python
# Total examples doubled after flattening
total_examples = len(dataset_df)  # Now includes both original + adversarial

with Progress(...) as progress:
    main_task = progress.add_task(
        f"[magenta]Processing {dataset.upper()} dataset", 
        total=total_examples,  # Updated to reflect flattened count
        dataset=dataset.upper()
    )
```

## 🎯 Kết Quả

### Output Statistics:
```
✓ Original rows: 100
✓ Flattened rows: 200 (100 original + 100 adversarial)
✓ Target distribution: {0: 100, 1: 100}
✓ Total rows: 200
✓ Unique IDs: 100
✓ Final columns: ['id', 'adv', 'func', 'cpg', 'target', 'input', 'cwe']
✓ Adv distribution: {False: 100, True: 100}
```

### Data Validation:
- ✅ **No NaN values**: Tất cả rows có `input` column
- ✅ **Perfect balance**: 50% benign, 50% vulnerable
- ✅ **Clean CPG**: Tất cả CPG là Dict objects, không phải List
- ✅ **Correct IDs**: Mỗi pair có cùng ID
- ✅ **Correct adv flags**: False=original, True=adversarial

## 🔄 Workflow Mới

### Full Pipeline:
```bash
# 1. Activate environment
conda activate gear

# 2. Generate counterexample dataset (if needed)
python generate_counterexample_dataset.py -d test

# 3. Process CPG to Input (with integrated flattening)
python cpg2input.py -d test

# 4. Train model (directly use output)
python train.py
```

### Không Cần Nữa:
- ❌ `python flatten_dataset.py` - Logic đã tích hợp vào cpg2input.py
- ❌ Intermediate files với NaN values
- ❌ Multiple processing passes

## 🔍 So Sánh Old vs New

| Aspect | Old Flow | New Flow |
|--------|----------|----------|
| **Steps** | 3 steps (CPG → Input → Flatten) | 1 step (CPG → Input with flatten) |
| **Files** | 2 files (input, balanced) | 1 file (balanced) |
| **NaN values** | Yes (in intermediate file) | No (all processed) |
| **Processing** | Process twice (before & after flatten) | Process once (all rows) |
| **Time** | Longer (multiple passes) | Faster (single pass) |
| **Disk usage** | Higher (intermediate files) | Lower (one output) |

## 💡 Lưu Ý Quan Trọng

### 1. **CPG List Handling**
Script tự động detect và extract dict từ list:
```python
# Input: [{'functions': ...}]
# Output: {'functions': ...}
```

### 2. **Target Flipping Logic**
- Original rows: Target bị flip (0→1, 1→0)
- Adversarial rows: Target giữ nguyên
- Kết quả: Perfect 50/50 balance

### 3. **ID Consistency**
Cùng một ID cho both original & adversarial:
```python
id='0': [original_row, adversarial_row]
id='1': [original_row, adversarial_row]
```

### 4. **Data Type Safety**
```python
dataset_df['target'] = dataset_df['target'].astype(int)
dataset_df['id'] = dataset_df['id'].astype(str)
dataset_df['adv'] = dataset_df['adv'].astype(bool)
```

## ✅ Kiểm Tra Output

### Sau khi chạy cpg2input.py:
```bash
# Verify output exists
ls -lh datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl

# Inspect with readpkl.py
python readpkl.py

# Check for NaN values
python -c "import pandas as pd; df = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl'); print(df.isna().sum())"
```

**Expected:**
- File size: ~X MB (tùy dataset size)
- Columns: id, adv, func, cpg, target, input, cwe
- No NaN in any column
- Row count = 2 × original count

---

# 5. Cài Đặt và Kiểm Tra Joern

**Version:** Joern 1.0.170  
**Trạng thái:** ✅ Đã cài và test thành công

## 📥 Cài Đặt Joern 1.0.170

### Download và Extract:
```bash
cd /home/cuong/DACN/VISION/joern

# Download Joern 1.0.170
wget https://github.com/joernio/joern/releases/download/v1.0.170/joern-cli.zip

# Extract
unzip joern-cli.zip

# Verify
ls -la joern-cli/
./joern-cli/joern-parse --help
```

### Vị trí cài đặt:
```
/home/cuong/DACN/VISION/joern/joern-cli/
├── bin/               # Executables
├── conf/              # Configuration
├── lib/               # JAR libraries
├── joern              # Main script
├── joern-parse        # Parse command
└── fuzzyc2cpg.sh      # C/C++ parser
```

## 🔧 Sửa Lỗi graph-for-funcs.sc

### Problem:
File `joern/graph-for-funcs.sc` có import không compatible với v1.0.170:
```scala
import io.shiftleft.dataflowengine.language._  // ❌ Không tồn tại trong v1.0.170
```

### Solution:
Xóa dòng import này:
```bash
# Line 24 trong file graph-for-funcs.sc
# Xóa hoặc comment:
// import io.shiftleft.dataflowengine.language._
```

### Result:
✅ Script hoạt động bình thường với Joern 1.0.170

## ✅ Test Joern

### Test 1: Parse C code
```bash
# Tạo test file
echo 'int add(int a, int b) { return a + b; } int main() { return add(1, 2); }' > /tmp/test.c

# Parse
./joern/joern-cli/joern-parse /tmp/test.c --out /tmp/test.bin

# Verify
ls -lh /tmp/test.bin  # Should be ~30-50KB
```

### Test 2: Create CPG JSON
```bash
# Create Joern script
cat > /tmp/test_cpg.sc << 'EOF'
importCpg("/tmp/test.bin")
run.ossdataflow
cpg.method.l.foreach(m => println(s"Method: ${m.name}"))
EOF

# Run script
./joern/joern-cli/joern --script /tmp/test_cpg.sc

# Expected output:
# Method: main
# Method: add
```

### Test 3: Full workflow với graph-for-funcs.sc
```bash
cd /home/cuong/DACN/VISION

# Test script đã sửa
./joern/joern-cli/joern-parse /tmp/test.c --out /tmp/test.bin

cat > /tmp/test_graph.sc << 'EOF'
importCpg("/tmp/test.bin")
run.ossdataflow
cpg.runScript("joern/graph-for-funcs.sc").toString() |> "/tmp/output.json"
EOF

./joern/joern-cli/joern --script /tmp/test_graph.sc

# Verify JSON output
cat /tmp/output.json | head -20
```

## 🎯 Workflow Integration

### Trong conda environment `gear`:
```bash
conda activate gear

# Test với full workflow
python graph2cpg.py -d test

# HOẶC

python generate_counterexample_dataset.py -d test
```

### Expected behavior:
1. ✅ Parse C files → CPG binary
2. ✅ Run graph-for-funcs.sc → JSON output
3. ✅ Process JSON → Python dict
4. ✅ No race condition với multi-threading
5. ✅ Cleanup temporary files

---

# 📚 Keywords và Thuật Ngữ

## Threading & Concurrency
- **Race Condition**: Tình trạng nhiều thread cùng truy cập/ghi đè shared resource
- **Thread-Safe**: Code an toàn khi chạy đa luồng
- **ThreadPoolExecutor**: Python's concurrent execution framework
- **Unique Identifier**: ID duy nhất để phân biệt các thread

## Error Handling
- **Safety Check**: Kiểm tra dữ liệu trước khi sử dụng
- **Exception Handling**: Xử lý ngoại lệ/lỗi
- **Debug Information**: Thông tin để debug lỗi
- **Validation**: Kiểm tra tính hợp lệ của dữ liệu

## Data Processing
- **Flatten**: Chuyển đổi nested structure thành flat structure
- **Paired Format**: Dữ liệu dạng cặp (original + adversarial)
- **Flat Format**: Dữ liệu dạng phẳng (mỗi row độc lập)
- **Class Balance**: Cân bằng số lượng giữa các class (50/50)
- **Pairwise Evaluation**: Đánh giá theo cặp

## Joern & CPG
- **CPG** (Code Property Graph): Đồ thị thuộc tính mã nguồn
- **Joern**: Tool phân tích static code
- **Parser**: Công cụ phân tích cú pháp
- **Binary Format**: Format nhị phân (.bin)
- **JSON Export**: Xuất ra định dạng JSON

## Machine Learning
- **Dataset Split**: Chia tập dữ liệu (train/val/test)
- **Group-level Split**: Chia theo nhóm (pair-wise)
- **Target Label**: Nhãn mục tiêu (0=benign, 1=vulnerable)
- **Original**: Mã nguồn gốc
- **Adversarial/Counterexample**: Mã nguồn đã modify

---

# ✅ Checklist Tổng Hợp

## Trước Khi Chạy Pipeline:

- [ ] Đã cài đặt Joern 1.0.170
- [ ] Đã sửa file `joern/graph-for-funcs.sc` (xóa dòng import dataflowengine)
- [ ] Đã test Joern với sample code
- [ ] Conda environment `gear` đã active
- [ ] Dataset gốc tồn tại trong `datasets/cwe20cfa/raw/`

## Sau Khi Generate Counterexamples:

- [ ] File output tồn tại: `cwe20cfa_CWE-20_augmented_test.pkl`
- [ ] File có columns: func, cpg, orig_func, orig_cpg, target
- [ ] CPG không null và có structure hợp lệ (có thể là Dict hoặc List[Dict])

## Sau Khi Chạy cpg2input.py:

- [ ] **Integrated flattening**: cpg2input.py tự động flatten dataset
- [ ] Output file: `cwe20cfa_CWE-20_augmented_input_balanced.pkl`
- [ ] Dataset được balanced 50/50 (tự động)
- [ ] Có columns: id, adv, func, cpg, target, input
- [ ] **Không có NaN values** trong input column
- [ ] Total rows = 2 × original rows (original + adversarial)
- [ ] Unique IDs = 50% của total rows

## Trước Khi Training:

- [ ] Output file từ cpg2input.py đã tồn tại
- [ ] Verify không có NaN: `python -c "import pandas as pd; df = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl'); print(df.isna().sum())"`
- [ ] Kiểm tra balance: Expected 50% benign, 50% vulnerable

## Troubleshooting:

- [ ] Nếu có lỗi race condition: Check unique temp files trong generate_counterexample_dataset.py
- [ ] Nếu có lỗi Joern: Check version và graph-for-funcs.sc
- [ ] Nếu CPG format lỗi: cpg2input.py tự động extract dict từ list
- [ ] Nếu có NaN values: Re-run cpg2input.py với flattening integrated
- [ ] Nếu cần reset: Chạy reset_flow.sh

---

# 🚀 Commands Tổng Hợp

## Full Pipeline (NEW - Simplified):

```bash
# 1. Activate environment
conda activate gear

# 2. Reset nếu cần
./reset_flow.sh

# 3. Generate counterexample dataset
python generate_counterexample_dataset.py -d test

# 4. Process CPG to Input (with integrated flattening)
python cpg2input.py -d test

# 5. Train model
python train.py
```

## Quick Test:

```bash
# Test Joern
echo 'int main() { return 0; }' > /tmp/test.c
./joern/joern-cli/joern-parse /tmp/test.c --out /tmp/test.bin
ls -lh /tmp/test.bin

# Test cpg2input with flattening
python cpg2input.py -d test

# Inspect output
python readpkl.py

# Verify no NaN values
python -c "import pandas as pd; df = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl'); print('NaN check:', df.isna().sum())"
```

---

# 📊 Ước Tính Thời Gian & Resource

| Task | Thời gian | CPU | Memory | Disk |
|------|-----------|-----|--------|------|
| Generate 100 examples | ~30 phút | 80% | 2GB | 500MB |
| CPG to Input (with flattening) | ~15 phút | 70% | 3GB | 200MB |
| Train 1 epoch | ~5 phút | 90% | 4GB | 100MB |
| Full training (100 epochs) | ~8 giờ | 90% | 4GB | 1GB |

---

# 📝 Ghi Chú Cuối

**Lưu ý quan trọng:**
1. Luôn backup dữ liệu trước khi reset
2. Kiểm tra Joern version trước khi chạy
3. **cpg2input.py đã tích hợp flattening logic** - không cần chạy flatten_dataset.py riêng
4. Verify dataset balance và no NaN values sau khi chạy cpg2input.py
5. Monitor memory usage với multi-threading
6. Cleanup temp files định kỳ
7. Output file: `cwe20cfa_CWE-20_augmented_input_balanced.pkl` sẵn sàng cho train.py

**Workflow Updates (v2.0):**
- ✅ Flattening logic integrated vào cpg2input.py
- ✅ Single-pass processing (không cần intermediate files)
- ✅ Tự động xử lý CPG format (Dict hoặc List[Dict])
- ✅ Perfect balance 50/50 tự động
- ✅ No NaN values in output

**Tài liệu tham khảo:**
- [Joern Documentation](https://docs.joern.io/)
- [Python ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html)
- [Pandas DataFrame](https://pandas.pydata.org/docs/)
- [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/)

---

**Báo cáo tổng hợp này được tạo và duy trì bởi GitHub Copilot**  
**Phiên bản:** 2.0 (Updated with integrated flattening)  
**Cập nhật lần cuối:** 11 tháng 3, 2026

generate_counterexample_dataset.py
    → cwe20cfa_CWE-20_augmented_{split}.pkl
            (func, target, cwe, orig_func, cpg)
                        │
                        ▼
            graph2cpg.py
    → cùng pkl, thêm cột orig_cpg
            (func, target, cwe, orig_func, cpg, orig_cpg)
                        │
                        ▼
            cpg2input.py
    → cwe20cfa_CWE-20_augmented_input_balanced.pkl
            (id, adv, func, cpg, target, input)
                        │
                        ▼
                    train.py