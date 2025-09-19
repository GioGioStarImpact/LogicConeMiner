# LogicConeMiner 測試

此目錄包含 LogicConeMiner 的測試檔案、測試腳本和測試結果。

## 測試檔案

### 測試電路

- **`test_circuit.v`** - 基本測試電路
  - 包含標準邏輯閘：AND2, OR2, NAND2
  - 用於測試基本邏輯錐探勘功能

- **`test_with_macro.v`** - 包含 Macro 的測試電路
  - 標準邏輯閘：AND2, OR2, NAND2, XOR2, OR3
  - Macro 元件：CPU_CORE, MEMORY_CTRL
  - 用於測試 CSV 元件庫和 Macro 邊界處理

### 測試腳本

- **`run_tests.sh`** - 自動化測試腳本
  - 執行所有測試案例
  - 生成測試結果摘要

## 執行測試

### 方法 1：使用測試腳本（推薦）

```bash
cd testing
./run_tests.sh
```

### 方法 2：手動執行

```bash
# 基本測試
python3 ../cone_finder.py \
  --netlist testing/test_circuit.v \
  --n_in 3 --n_out 2 --n_depth 5 \
  --out_dir testing/test_results

# Macro 測試
python3 ../cone_finder.py \
  --netlist testing/test_with_macro.v \
  --cell_library cell_library.csv \
  --n_in 4 --n_out 2 --n_depth 10 \
  --out_dir testing/test_macro_results
```

## 測試結果

測試結果會保存在以下目錄：

- `test_results/` - 基本測試結果
  - `cones.jsonl` - 發現的邏輯錐詳細資訊
  - `summary.json` - 統計摘要

- `test_macro_results/` - Macro 測試結果
  - `cones.jsonl` - 發現的邏輯錐詳細資訊
  - `summary.json` - 統計摘要

## 預期結果

### 基本測試
- 應該發現多個組合邏輯錐
- 所有錐都應該通過連通性檢查

### Macro 測試
- 應該正確識別標準元件和 Macro
- Macro 應該被視為邊界，產生分離的邏輯區塊
- 預期輸出：
  ```
  從 cell_library.csv 載入 26 個標準元件定義
  檢測到 Macro: CPU_CORE (實例: cpu_inst)
  檢測到 Macro: MEMORY_CTRL (實例: mem_ctrl)
  找到 2 個組合邏輯區塊
  完成！發現 8 個邏輯錐
  ```

## 故障排除

如果測試失敗，請檢查：

1. Python 3 是否正確安裝
2. 所有必要檔案是否存在
3. 檔案權限是否正確
4. 輸出目錄是否可寫入

如需詳細除錯資訊，可以在指令中加入 `-v` 或 `--verbose` 參數。