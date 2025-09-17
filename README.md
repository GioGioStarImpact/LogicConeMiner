# LogicConeMiner

基於門級 Verilog netlists 的邏輯錐探勘工具。

## 功能特性

- **自動邊界檢測**：自動在時序元件（FF/Latch）和 PI/PO 處切割
- **k-feasible cuts 演算法**：用於單輸出錐的高效枚舉
- **多輸出錐支援**：基於支撐共享的群組化策略
- **約束驗證**：支援輸入數量、輸出數量、深度約束
- **連通性檢查**：確保錐形成單一連通元件
- **去重機制**：基於簽名的重複錐過濾

## 使用方法

### 基本使用

```bash
python3 cone_finder.py --netlist input.v --n_in 4 --n_out 2 --n_depth 10
```

### 完整參數

```bash
python3 cone_finder.py \
  --netlist circuit.v \
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
- `--n_in`: 錐的最大輸入數量
- `--n_out`: 錐的最大輸出數量
- `--n_depth`: 錐的最大深度（閘數）
- `--cmp_in/out/depth`: 比較運算子（`<=` 或 `==`）
- `--count_inverters_in_depth`: 是否將反相器計入深度（預設 true）
- `--max_cuts_per_node`: 每節點最大 cut 數量（記憶體控制）
- `--out_dir`: 輸出目錄（預設 "results"）

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

使用提供的測試電路：
```bash
python3 cone_finder.py --netlist test_circuit.v --n_in 3 --n_out 2 --n_depth 5
```

## 限制與假設

- 支援結構化 Verilog，不支援行為式描述
- 時序元件識別基於關鍵字模式匹配
- 假設標準邏輯閘命名規範（AND2, OR2 等）
- 不支援三態邏輯和多驅動網路
- 組合迴圈會被檢測但未自動處理