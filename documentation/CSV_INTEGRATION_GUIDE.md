# CSV 元件庫整合指南

## 概述

LogicConeMiner 已成功整合 CSV 元件庫支援，實現了您要求的兩個關鍵功能：
1. 使用 CSV 檔案定義標準元件資訊
2. 將非標準元件（Macro）視為邊界節點處理

## 主要改進

### 1. CSV 元件庫定義

**檔案格式：`cell_library.csv`**
```csv
cell_name,cell_type,input_pins,output_pins,is_sequential,clock_pin,data_pin
AND2,combinational,"A,B",Y,false,,
DFF,sequential,"D,CLK",Q,true,CLK,D
```

**欄位說明：**
- `cell_name`: 元件名稱
- `cell_type`: 元件類型（combinational/sequential）
- `input_pins`: 輸入埠名稱（逗號分隔）
- `output_pins`: 輸出埠名稱（逗號分隔）
- `is_sequential`: 是否為時序元件（true/false）
- `clock_pin`: 時鐘埠名稱（時序元件用）
- `data_pin`: 資料埠名稱（時序元件用）

### 2. Macro 處理策略

**自動檢測：**
- 任何不在 CSV 檔案中定義的元件自動視為 Macro
- 系統會記錄並報告檢測到的 Macro

**邊界處理：**
- Macro 輸入埠 → 視為偽 PO（前級邏輯的終點）
- Macro 輸出埠 → 視為偽 PI（後級邏輯的起點）
- 這樣確保邏輯錐分析不會跨越 Macro 邊界

## 使用方法

### 基本用法（使用 CSV 元件庫）
```bash
python3 cone_finder.py \
  --netlist design.v \
  --cell_library cell_library.csv \
  --n_in 4 --n_out 2 --n_depth 10
```

### 不指定 CSV（降級模式）
```bash
python3 cone_finder.py \
  --netlist design.v \
  --n_in 4 --n_out 2 --n_depth 10
```
如果沒有指定 `--cell_library`，程式會自動載入內建的基本元件定義。

## 實際測試結果

### 測試電路：`testing/test_with_macro.v`
包含：
- 標準邏輯閘：AND2, OR2, NAND2, XOR2, OR3
- Macro 元件：CPU_CORE, MEMORY_CTRL

### 執行結果：
```
從 cell_library.csv 載入 26 個標準元件定義
檢測到 Macro: CPU_CORE (實例: cpu_inst)
檢測到 Macro: MEMORY_CTRL (實例: mem_ctrl)
找到 2 個組合邏輯區塊
完成！發現 8 個邏輯錐
```

**關鍵觀察：**
- ✅ 正確識別標準元件和 Macro
- ✅ Macro 被正確處理為邊界，產生多個分離的組合邏輯區塊
- ✅ 邏輯錐分析不跨越 Macro 邊界

## CSV 檔案管理

### 創建新的 CSV 檔案
```python
from csv_cell_library import CSVCellLibrary

lib = CSVCellLibrary()
lib.export_template_csv("my_cells.csv")
```

### 擴展現有定義
您可以編輯 `cell_library.csv` 添加更多標準元件：
```csv
# 新增複雜邏輯閘
AOI21,combinational,"A1,A2,B",Y,false,,
OAI22,combinational,"A1,A2,B1,B2",Y,false,,

# 新增更多時序元件
SDFF,sequential,"D,CLK,SE,SI",Q,true,CLK,D
```

## 相容性保證

### 向後相容
- 原有的所有功能完全保留
- 無 CSV 檔案時自動降級為內建定義
- 所有原有的命令列參數維持不變

### 降級機制
1. **CSV 解析失敗** → 使用內建定義
2. **元件不在 CSV 中** → 使用啟發式規則 + Macro 處理
3. **埠名稱不匹配** → 降級為通用埠名稱

## 技術實作細節

### 核心類別

1. **CSVCellLibrary**
   - 載入和管理 CSV 元件定義
   - 提供元件查詢和驗證功能

2. **MacroHandler**
   - 檢測非標準元件
   - 將 Macro 轉換為邊界節點

3. **Enhanced VerilogParser**
   - 整合 CSV 元件庫支援
   - 智慧型元件處理策略

### 處理流程
```
Verilog 解析 → 元件分類 → 標準元件處理 + Macro 邊界處理 → 圖建構 → 錐分析
```

## 效益總結

### 對於機密性
- ✅ 無需提供完整的 Liberty 檔案
- ✅ 只需要基本的埠定義資訊
- ✅ 可自訂元件定義不洩露設計細節

### 對於實用性
- ✅ 自動處理階層化設計中的 Macro
- ✅ 正確的邊界切割不影響分析精度
- ✅ 簡單的 CSV 格式易於維護

### 對於擴展性
- ✅ 支援任意數量的自訂標準元件
- ✅ 靈活的降級機制確保robustness
- ✅ 模組化設計便於未來擴展

這個方案完美解決了您提出的兩個需求，既保護了機密資訊，又提供了實用的 Macro 處理能力！