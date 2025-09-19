# LogicConeMiner

基於門級 Verilog netlists 的邏輯錐探勘工具，支援 CSV 元件庫定義與 Macro 邊界處理。

## 功能特性

- **自動邊界檢測**：自動在時序元件（FF/Latch）和 PI/PO 處切割
- **k-feasible cuts 演算法**：用於單輸出錐的高效枚舉
- **多輸出錐支援**：基於支撐共享的群組化策略
- **約束驗證**：支援輸入數量、輸出數量、深度約束
- **連通性檢查**：確保錐形成單一連通元件
- **去重機制**：基於簽名的重複錐過濾
- **🆕 CSV 元件庫**：使用 CSV 檔案定義標準元件，保護機密資訊
- **🆕 Macro 支援**：自動檢測非標準元件，視為邊界節點處理
- **🆕 多層級降級**：CSV → 內建 → 啟發式，確保 robustness

## 使用方法

### 基本使用

```bash
# 不使用 CSV 元件庫（降級模式）
python3 cone_finder.py --netlist input.v --n_in 4 --n_out 2 --n_depth 10

# 使用 CSV 元件庫（推薦）
python3 cone_finder.py \
  --netlist input.v \
  --cell_library cell_library.csv \
  --n_in 4 --n_out 2 --n_depth 10
```

### 完整參數

```bash
python3 cone_finder.py \
  --netlist circuit.v \
  --cell_library cell_library.csv \
  --n_in 4 \
  --n_out 2 \
  --n_depth 10 \
  --cmp_in "<=" \
  --cmp_out "<=" \
  --cmp_depth "<=" \
  --count_inverters_in_depth true \
  --max_cuts_per_node 150 \
  --out_dir results/
```

### 參數說明

- `--netlist`: 輸入的 Verilog netlist 檔案
- `--cell_library`: 🆕 CSV 元件庫檔案（可選）
- `--n_in`: 錐的最大輸入數量
- `--n_out`: 錐的最大輸出數量
- `--n_depth`: 錐的最大深度（閘數）
- `--cmp_in/out/depth`: 比較運算子（`<=` 或 `==`）
- `--count_inverters_in_depth`: 是否將反相器計入深度（預設 true）
- `--max_cuts_per_node`: 每節點最大 cut 數量（記憶體控制）
- `--out_dir`: 輸出目錄（預設 "results"）

## CSV 元件庫格式

創建 `cell_library.csv` 檔案定義標準元件：

```csv
cell_name,cell_type,input_pins,output_pins,is_sequential,clock_pin,data_pin
AND2,combinational,"A,B",Y,false,,
DFF,sequential,"D,CLK",Q,true,CLK,D
INV,combinational,A,Y,false,,
NAND2,combinational,"A,B",Y,false,,
OR2,combinational,"A,B",Y,false,,
```

### 欄位說明

- `cell_name`: 元件名稱（如 AND2, DFF）
- `cell_type`: 元件類型（combinational/sequential）
- `input_pins`: 輸入埠名稱，逗號分隔（如 "A,B"）
- `output_pins`: 輸出埠名稱，逗號分隔（如 "Y"）
- `is_sequential`: 是否為時序元件（true/false）
- `clock_pin`: 時鐘埠名稱（時序元件使用）
- `data_pin`: 資料埠名稱（時序元件使用）

## Macro 處理

系統會自動檢測不在 CSV 檔案中定義的元件：

- **Macro 輸入埠** → 視為偽 PO（前級邏輯終點）
- **Macro 輸出埠** → 視為偽 PI（後級邏輯起點）

這確保邏輯錐分析不會跨越 Macro 邊界，適合處理階層化設計。

## 輸出檔案

### cones.jsonl
每行一個錐的 JSON 記錄：
```json
{
  "cone_id": "hash",
  "block_id": 0,
  "roots": ["u1.Y", "u2.Y"],
  "leaves": ["a", "b"],
  "depth": 3,
  "num_nodes": 5,
  "num_edges": 6,
  "connected": true,
  "signature": "abc123..."
}
```

### summary.json
統計摘要：
```json
{
  "total_cones": 26,
  "total_blocks": 1,
  "distribution": {
    "by_depth": {"1": 3, "2": 4, "3": 18},
    "by_inputs": {"1": 1, "2": 25},
    "by_outputs": {"1": 6, "2": 20}
  }
}
```

## 演算法概述

1. **Verilog 解析**：解析結構化 netlist，識別邏輯閘和時序元件
2. **時序切割**：在 FF/Latch/PI/PO 處切割，形成組合邏輯區塊
3. **單輸出錐枚舉**：使用 k-feasible cuts 動態規劃演算法
4. **多輸出錐枚舉**：基於支撐共享的群組化
5. **連通性驗證**：確保錐形成單一連通元件
6. **去重處理**：使用簽名機制避免重複錐

## 測試

### 使用測試腳本（推薦）
```bash
cd testing
./run_tests.sh
```

### 手動測試

#### 標準測試電路
```bash
python3 cone_finder.py --netlist testing/test_circuit.v --n_in 3 --n_out 2 --n_depth 5
```

#### 測試 CSV 元件庫與 Macro 支援
```bash
python3 cone_finder.py \
  --netlist testing/test_with_macro.v \
  --cell_library cell_library.csv \
  --n_in 4 --n_out 2 --n_depth 10
```

預期輸出：
```
從 cell_library.csv 載入 26 個標準元件定義
檢測到 Macro: CPU_CORE (實例: cpu_inst)
檢測到 Macro: MEMORY_CTRL (實例: mem_ctrl)
找到 2 個組合邏輯區塊
完成！發現 8 個邏輯錐
```

## 限制與假設

- 支援結構化 Verilog，不支援行為式描述
- 非標準元件自動視為 Macro（需 CSV 元件庫精確定義）
- 不支援三態邏輯和多驅動網路
- 組合迴圈會被檢測但未自動處理

## 專案結構

```
LogicConeMiner/
├── cone_finder.py              # 主程式
├── csv_cell_library.py         # CSV 元件庫管理與 Macro 處理
├── cell_library.csv            # 標準元件定義
├── README.md                   # 專案說明
├── testing/                    # 測試相關檔案
│   ├── test_circuit.v          # 基本測試電路
│   ├── test_with_macro.v       # Macro 測試電路
│   ├── run_tests.sh            # 自動化測試腳本
│   ├── test_results/           # 基本測試結果
│   └── test_macro_results/     # Macro 測試結果
└── documentation/              # 說明文件
    ├── DESIGN_GUIDE.md         # 程式設計指南
    ├── CSV_INTEGRATION_GUIDE.md # CSV 整合說明
    ├── CLAUDE.md               # Claude 專案指令
    └── logic_cone_spec_*.md    # 規格文件
```

## 更新日誌

### v2.0 (最新版本)
- ✅ 新增 CSV 元件庫支援
- ✅ 實作 Macro 自動檢測與邊界處理
- ✅ 多層級降級機制（CSV → 內建 → 啟發式）
- ✅ 保護元件庫機密資訊
- ✅ 完整測試與文件

### v1.0 (基礎版本)
- ✅ k-feasible cuts 演算法
- ✅ 多輸出錐支援
- ✅ 時序切割與連通性檢查