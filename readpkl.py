import pickle
import pandas as pd
import numpy as np

file_path = '/home/cuong/DACN/VISION/datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_train.pkl'

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    print("=" * 80)
    print("THÔNG TIN FILE PICKLE")
    print("=" * 80)
    print(f"File path: {file_path}")
    print(f"Kiểu dữ liệu: {type(data)}")
    print()
    
    # Xử lý theo kiểu dữ liệu
    if isinstance(data, pd.DataFrame):
        print("-" * 80)
        print("PANDAS DATAFRAME")
        print("-" * 80)
        print(f"Shape: {data.shape} (rows, columns)")
        print(f"Số lượng rows: {len(data)}")
        print(f"Số lượng columns: {len(data.columns)}")
        print(f"\nCác columns:\n{list(data.columns)}")
        
        print("\n" + "-" * 80)
        print("THÔNG TIN CHI TIẾT")
        print("-" * 80)
        data.info()
        
        print("\n" + "-" * 80)
        print("5 HÀNG ĐẦU TIÊN")
        print("-" * 80)
        print(data.head())
        
        print("\n" + "-" * 80)
        print("THỐNG KÊ MÔ TẢ (CHO CÁC COLUMNS SỐ)")
        print("-" * 80)
        print(data.describe())
        
        # Kiểm tra các columns đặc biệt
        if 'target' in data.columns:
            print("\n" + "-" * 80)
            print("PHÂN PHỐI TARGET")
            print("-" * 80)
            print(data['target'].value_counts())
        
        if 'cpg' in data.columns:
            print("\n" + "-" * 80)
            print("THÔNG TIN CPG COLUMN")
            print("-" * 80)
            print(f"Kiểu dữ liệu CPG: {type(data['cpg'].iloc[0]) if len(data) > 0 else 'N/A'}")
            if len(data) > 0:
                sample_cpg = data['cpg'].iloc[0]
                if isinstance(sample_cpg, dict):
                    print(f"CPG keys: {list(sample_cpg.keys())}")
                    print(f"\nMẫu CPG đầu tiên:")
                    for key, value in list(sample_cpg.items())[:3]:
                        print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        if 'input' in data.columns:
            print("\n" + "-" * 80)
            print("THÔNG TIN INPUT COLUMN")
            print("-" * 80)
            print(f"Kiểu dữ liệu Input: {type(data['input'].iloc[0]) if len(data) > 0 else 'N/A'}")
            if len(data) > 0 and isinstance(data['input'].iloc[0], (list, np.ndarray)):
                print(f"Shape của input đầu tiên: {np.array(data['input'].iloc[0]).shape}")
    
    elif isinstance(data, (list, tuple)):
        print("-" * 80)
        print("LIST/TUPLE")
        print("-" * 80)
        print(f"Số lượng phần tử: {len(data)}")
        if len(data) > 0:
            print(f"Kiểu phần tử đầu tiên: {type(data[0])}")
            print(f"\n3 phần tử đầu tiên:")
            for i, item in enumerate(data[:3]):
                print(f"\n[{i}]: {type(item)}")
                if isinstance(item, dict):
                    print(f"     Keys: {list(item.keys())}")
                else:
                    print(f"     {str(item)[:200]}")
    
    elif isinstance(data, dict):
        print("-" * 80)
        print("DICTIONARY")
        print("-" * 80)
        print(f"Số lượng keys: {len(data)}")
        print(f"Keys: {list(data.keys())}")
        print("\nNội dung:")
        for key, value in list(data.items())[:10]:
            print(f"\n{key}: {type(value)}")
            if isinstance(value, (list, np.ndarray)):
                print(f"  Length/Shape: {len(value) if isinstance(value, list) else value.shape}")
            print(f"  Value: {str(value)[:200]}")
    
    else:
        print("-" * 80)
        print("DỮ LIỆU KHÁC")
        print("-" * 80)
        print(data)
    
    print("\n" + "=" * 80)
    print("ĐỌC FILE THÀNH CÔNG!")
    print("=" * 80)
    
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
except pickle.UnpicklingError:
    print("❌ Error: The file content is not a valid pickle format.")
except EOFError:
    print("❌ Error: The file is incomplete or corrupted.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
# import pandas as pd
# import numpy as np

# # 1. Load data
# file_path = 'datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_test_input.pkl'
# raw_df = pd.read_pickle(file_path)

# processed_rows = []
# for idx, row in raw_df.iterrows():
#     # 1. Dòng Counterexample (Dữ liệu đã sửa qua LLM)
#     processed_rows.append({
#         'id': str(idx), 
#         'adv': True, 
#         'func': row['func'], 
#         'target': int(row['target']), 
#         'input': row['input'], 
#         'cpg': row['cpg']
#     })
    
#     # 2. Dòng Original (Dữ liệu gốc)
#     # TẠM THỜI: Nếu chưa có 'orig_input', ta kiểm tra 'orig_cpg' 
#     # Nhưng lưu ý: Để train được GNN, bạn CẦN chạy cpg2input cho bản gốc trước.
#     if 'orig_input' in row and pd.notna(row['orig_input']):
#         orig_target = 0 if int(row['target']) == 1 else 1
#         processed_rows.append({
#             'id': str(idx), 
#             'adv': False, 
#             'func': row['orig_func'], 
#             'target': orig_target,
#             'input': row['orig_input'], 
#             'cpg': row['orig_cpg']
#         })
#     else:
#         # Cảnh báo để bạn biết tại sao bản gốc không xuất hiện
#         print(f"Cảnh báo: Hàng {idx} thiếu 'orig_input', bản gốc sẽ không được thêm vào tập Train.")

# # Tạo lại DataFrame mới
# dataset_df = pd.DataFrame(processed_rows)

# # Kiểm tra thử kết quả
# print("\nThông tin Dataset sau khi chuyển đổi:")
# print(dataset_df.info())
# print("\n5 hàng đầu tiên:")
# print(dataset_df[['id', 'adv', 'target']].head())